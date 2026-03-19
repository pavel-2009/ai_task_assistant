"""
Сервис для сегментации изображений
"""

import torch
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101
from PIL import Image
import io
import numpy as np
import cv2


class SegmentationService:
    """Сервис сегментации изображений"""
    
    def __init__(self):
        """Инициализация модели и трансформаций"""
        self.model = deeplabv3_resnet101(pretrained=True).eval()
        self.transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        
    def segment_image(self, image: bytes) -> bytes:
        """Сегментация входящего изображения с наложением маски на исходное"""
        
        image_pil = Image.open(io.BytesIO(image)).convert("RGB")
        original_size = image_pil.size
        
        image_tensor = self.transform(image_pil).unsqueeze(0)
    
        with torch.no_grad():
            output: torch.Tensor = self.model(image_tensor)['out'][0]
            
        mask = output.argmax(0).cpu().numpy().astype(np.uint8)
        mask_resized = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)
        
        binary_mask = (mask_resized > 0).astype(np.uint8) * 255
        
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        image_array = np.array(image_pil)
        
        overlay = image_array.copy()
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), -1)
        
        alpha = 0.3
        result = cv2.addWeighted(image_array, 1 - alpha, overlay, alpha, 0)
        
        cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
        
        result_pil = Image.fromarray(result)
        output_image_bytes = io.BytesIO()
        result_pil.save(output_image_bytes, format='PNG')
        
        return output_image_bytes.getvalue()
            
        
            
        