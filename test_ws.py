"""
Простой скрипт для тестирования веб-сокетов.
"""

import asyncio
import websockets

from pathlib import Path


IMAGE_FILE = Path(__file__).parent / "avatars" / "0_0.jpg"

with open(IMAGE_FILE, "rb") as f:
    image_bytes = f.read()


async def test_websocket():
    uri = "ws://localhost:8000/ws/detect"
    
    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")
        
        total_start_time = asyncio.get_event_loop().time()
        
        # Отправляем и получаем 100 раз
        for i in range(100):
            start_time = asyncio.get_event_loop().time()
            
            await websocket.send(image_bytes)
            print(f"[{i+1}] Image sent")
            
            response = await websocket.recv()
            print(f"[{i+1}] Received:", response)
            end_time = asyncio.get_event_loop().time()
            print(f"[{i+1}] Time taken: {end_time - start_time:.4f} seconds\n")
        
        total_end_time = asyncio.get_event_loop().time()
        print(f"Total time: {total_end_time - total_start_time:.2f} seconds")
            
            
if __name__ == "__main__":
    asyncio.run(test_websocket())