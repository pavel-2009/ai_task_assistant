"""
Сервис с получением эмбеддингов для текста
"""

from transformers import AutoTokenizer, AutoModel
import numpy as np
import torch


class EmbeddingService:
    """Сервис для получения эмбеддингов текста"""
    
    def __init__(self):
        self.model: AutoModel = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        
        
    def encode(self, text: str) -> np.ndarray:
        """Получить эмбеддинг для текста"""
        
        inputs_tokenized = self.tokenizer(
            text,
            padding=True, # Добавляем паддинг, чтобы все входные данные были одинаковой длины
            truncation=True, # Обрезаем текст, если он слишком длинный для модели
            return_tensors="pt" # Возвращаем тензоры PyTorch
        )
        with torch.no_grad():
            outputs = self.model(**inputs_tokenized)
            
        embeddings = outputs.last_hidden_state[:, 0, :].numpy() # Получаем эмбеддинг для CLS токена
        
        return embeddings.squeeze()  # Преобразуем из (1, 384) в (384,)
    
    
def get_embedding_service() -> EmbeddingService:
    """Получить экземпляр сервиса для получения эмбеддингов"""
    return EmbeddingService()


def test_cosine_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """Тестовая функция для проверки косинусного сходства между двумя эмбеддингами"""
    dot_product = np.dot(embedding1, embedding2)
    norm1 = np.linalg.norm(embedding1)
    norm2 = np.linalg.norm(embedding2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    cosine_similarity = dot_product / (norm1 * norm2)
    
    return cosine_similarity


if __name__ == "__main__":
    
    service = get_embedding_service()
    
    text1 = "Использования модели для получения эмбеддингов текста"
    text2 = "Здравствуйте, как у вас дела?"
    
    embedding1 = service.encode(text1)
    embedding2 = service.encode(text2)
    
    similarity = test_cosine_similarity(embedding1, embedding2)
    
    print(f"Косинусное сходство между текстами: {similarity:.4f}")