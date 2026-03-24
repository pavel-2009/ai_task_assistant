"""
Сервис для извлечения именованных сущностей из текста с помощью модели NER (Named Entity Recognition).
"""

import spacy
from spacy.tokens import Span
import logging
from typing import List, Dict, Any


from functools import lru_cache


logger = logging.getLogger(__name__)


class NerService:
    """
    Сервис для извлечения именованных сущностей из текста с помощью модели NER (Named Entity Recognition).
    Фокусируется на технологиях (PRODUCT), компаниях (ORG) и произведениях (WORK_OF_ART).
    """
    
    # Технологии, которые могут быть ошибочно классифицированы
    BLACKLIST = {"apple", "google", "microsoft", "amazon", "facebook", "twitter", "instagram", "whatsapp"}
    
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        """Инициализация сервиса с загрузкой модели NER."""
        
        self.nlp = spacy.load(model_name)
        self._add_special_cases()
        
        if not Span.has_extension("confidence"):
            Span.set_extension("confidence", default=0.8)
        
        logger.info(f"NER модель '{model_name}' загружена успешно.")
        
        
    @property
    def is_ready(self) -> bool:
        """Проверяет состояние модели. Используется в healthcheck"""
        
        return self.nlp is not None
        
        
    def _add_special_cases(self) -> None:
        # 1. Создаем компонент EntityRuler правильно
        if "entity_ruler" not in self.nlp.pipe_names:
            ruler = self.nlp.add_pipe("entity_ruler", before="ner")
        else:
            ruler = self.nlp.get_pipe("entity_ruler")

        # 2. Определяем паттерны. 
        patterns = [
            {"label": "PRODUCT", "pattern": "C++"},
            {"label": "PRODUCT", "pattern": "C#"},
            {"label": "PRODUCT", "pattern": "F#"},
            {"label": "PRODUCT", "pattern": "Node.js"},
            {"label": "PRODUCT", "pattern": "React.js"},
            {"label": "PRODUCT", "pattern": "Vue.js"},
            {"label": "PRODUCT", "pattern": "Angular.js"},
            {"label": "PRODUCT", "pattern": "FastAPI"},
            {"label": "PRODUCT", "pattern": "PostgreSQL"},
        ]

        ruler.add_patterns(patterns)
    
    
    def extract_technologies(self, text: str) -> List[tuple[str, float]]:
        """Извлекает технологии из текста"""
        
        processed = self.nlp(text)
        
        results = set()
        
        for ent in processed.ents:
            
            if ent.label_ not in {"PRODUCT", "ORG", "WORK_OF_ART"}:
                continue
            
            if ent.text.lower() in self.BLACKLIST:
                continue
            # Устанавливаем confidence для сущностей, извлеченных с помощью EntityRuler
            if ent.text in {"C++", "C#", "F#", "Node.js", "React.js", "Vue.js", "Angular.js"}:
                ent._.confidence = 0.95
            results.add((ent.text.lower(), ent._.confidence))
            
        return list(set(results))
    
    
    @lru_cache(maxsize=32)
    def tag_task(self, text: str) -> Dict[str, Any]:
        """Автоматическое тегирование задачи технологиями"""
        
        technologies = self.extract_technologies(text)
        confidence = [conf for _, conf in technologies]
        
        return {
            "technologies": technologies,
            "confidence": sum(confidence) / len(confidence) if confidence else 0
        }
        
        
# Пример использования:
if __name__ == "__main__":
    service = NerService()
    
    sample_text = "We need to develop a web application using React.js and Node.js, and also consider using C++ for performance-critical components. Also, we might want to integrate with Google services and FastAPI."
    
    result = service.tag_task(sample_text)
    
    print("Extracted Technologies:", result["technologies"])
    print("Average Confidence:", result["confidence"])
        