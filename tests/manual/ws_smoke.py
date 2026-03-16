"""
Ручной smoke-скрипт для проверки websocket-детекции.
Запускать только при поднятом docker-compose (api + celery + redis).
"""

import asyncio
import io

import websockets
from PIL import Image


def _build_test_image_bytes() -> bytes:
    image = Image.new("RGB", (320, 240), color=(120, 120, 120))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()


async def run_websocket_smoke() -> None:
    uri = "ws://localhost:8000/ws/detect"
    image_bytes = _build_test_image_bytes()

    async with websockets.connect(uri) as websocket:
        print("Connected to WebSocket")

        total_start_time = asyncio.get_event_loop().time()

        for i in range(10):
            start_time = asyncio.get_event_loop().time()

            await websocket.send(image_bytes)
            response = await websocket.recv()

            elapsed = asyncio.get_event_loop().time() - start_time
            print(f"[{i + 1}] response={response}")
            print(f"[{i + 1}] elapsed={elapsed:.4f}s")

        total_elapsed = asyncio.get_event_loop().time() - total_start_time
        print(f"Total elapsed: {total_elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(run_websocket_smoke())
