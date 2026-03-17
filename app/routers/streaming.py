"""
Роутер для realtime стриминга видео с одновременной обработкой видео с YOLOService
"""

from fastapi import APIRouter, WebSocket
from fastapi.websockets import WebSocketDisconnect

import asyncio

from ..celery_app import get_yolo_service


router = APIRouter(
    prefix="/ws",
    tags=["streaming"],
)


@router.websocket("/detect", name="streaming:detect")
async def detect(websocket: WebSocket, target_fps: int = 30):
    await websocket.accept()
    min_frame_interval = 1.0 / target_fps
    
    yolo_service = get_yolo_service()
    frame_queue = asyncio.Queue(maxsize=2)  # Уменьшаем размер очереди
    last_process_time = 0
    
    async def receive_frames():
        try:
            while True:
                data = await websocket.receive_bytes()
                
                # Не блокируемся на полной очереди
                if frame_queue.full():
                    try:
                        frame_queue.get_nowait()  # Удаляем старый кадр
                    except asyncio.QueueEmpty:
                        pass
                
                await frame_queue.put({
                    'data': data,
                    'timestamp': asyncio.get_event_loop().time()
                })
        except WebSocketDisconnect:
            pass
    
    async def process_frames():
        nonlocal last_process_time
        loop = asyncio.get_event_loop()
        
        while True:
            try:
                frame = await frame_queue.get()
                
                # Контроль FPS
                current_time = loop.time()
                time_since_last = current_time - last_process_time
                if time_since_last < min_frame_interval:
                    await asyncio.sleep(min_frame_interval - time_since_last)
                
                # Оптимизация: батч-обработка если возможно
                results = await loop.run_in_executor(None, yolo_service.predict, frame['data'])
                
                await websocket.send_json({
                    'objects': results,
                    'processing_time': loop.time() - frame['timestamp'],
                    'fps': 1.0 / (loop.time() - last_process_time) if last_process_time else target_fps
                })
                
                last_process_time = loop.time()
                
            except WebSocketDisconnect:
                break
            except Exception as e:
                print(f"Error: {e}")
    
    await asyncio.gather(receive_frames(), process_frames())