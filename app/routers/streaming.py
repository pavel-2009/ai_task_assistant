"""
Роутер для realtime стриминга видео с одновременной обработкой видео с YOLOService
"""

from fastapi import APIRouter, Depends, WebSocket
from fastapi.websockets import WebSocketDisconnect

import asyncio
import logging

from app.ml.cv.detection.yolo_onnx_service import YoloONNXService
from app.ml.cv.detection.yolo_service import YoloService
from app.services import get_yolo


router = APIRouter(
    prefix="/ws",
    tags=["streaming"],
)

logger = logging.getLogger(__name__)


@router.websocket("/detect", name="streaming:detect")
async def detect(
    websocket: WebSocket,
    target_fps: int = 30,
    yolo_service: YoloService | YoloONNXService = Depends(get_yolo),
):
    await websocket.accept()
    min_frame_interval = 1.0 / target_fps

    frame_queue = asyncio.Queue(maxsize=2)  # Уменьшаем размер очереди
    last_process_time = 0
    close_event = asyncio.Event()  # Сигнал о закрытии соединения
    
    async def receive_frames():
        try:
            while close_event.is_set() == False:
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
            logger.info("WebSocket disconnected in receive_frames")
            await frame_queue.put(None)  # Сигнал для process_frames о завершении
        except Exception as e:
            logger.error(f"Error in receive_frames: {e}")
        finally:
            close_event.set()  # Устанавливаем сигнал о закрытии при любом исключении или отключении
    
    async def process_frames():
        nonlocal last_process_time
        loop = asyncio.get_event_loop()
        
        while not close_event.is_set():
            try:
                # Используем timeout, чтобы периодически проверять close_event
                try: 
                    frame = await asyncio.wait_for(frame_queue.get(), timeout=0.1)
                    
                    if frame is None:  # Получили сигнал о завершении
                        break
                except asyncio.TimeoutError:
                    continue  # Нет кадра, продолжаем проверку close_event
                
                # Контроль FPS
                current_time = loop.time()
                time_since_last = current_time - last_process_time
                
                    
                if close_event.is_set():
                    break  # Выходим из цикла, если соединение закрыто
                
                
                results = await yolo_service.predict_async(frame['data'])
                
                # Если прошло меньше времени, чем нужно для достижения target_fps, ждем
                if time_since_last < min_frame_interval and last_process_time > 0:
                    try:
                        await asyncio.wait_for(
                            asyncio.sleep(min_frame_interval - time_since_last),
                            timeout=0.1  # Периодически просыпаемся проверять close_event
                        )
                    except asyncio.TimeoutError:
                        pass  # Просто проверяем условие цикла
                
                if not close_event.is_set():
                
                    await websocket.send_json({
                        'objects': results,
                        'processing_time': loop.time() - frame['timestamp'],
                        'fps': 1.0 / (loop.time() - last_process_time) if last_process_time else target_fps
                    })
                    
                last_process_time = loop.time()
                
                
            except WebSocketDisconnect:
                logger.info("WebSocket disconnected in process_frames")
                break
            except Exception as e:
                logger.error(f"Ошибка при обработке кадра: {e}", exc_info=True)
                
                # Очищаем очередь при ошибке, чтобы не обрабатывать устаревшие кадры
                while not frame_queue.empty():
                    try:
                        frame_queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                
                try:
                    await websocket.send_json({'error': str(e)})
                except WebSocketDisconnect:
                    break
    
    # Создаем задачи явно для правильного graceful shutdown
    receive_task = asyncio.create_task(receive_frames())
    process_task = asyncio.create_task(process_frames())
    
    try:
        await asyncio.gather(receive_task, process_task)
    except WebSocketDisconnect:
        pass
    finally:
        # Graceful shutdown: отменяем только созданные задачи
        close_event.set()
        
        for task in [receive_task, process_task]:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
