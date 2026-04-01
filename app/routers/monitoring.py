"""
Роутер для мониторинга моделей машинного обучения.
"""

from fastapi import APIRouter, HTTPException, status
from app.schemas import DriftReportResponse, DriftHistoryResponse


router = APIRouter(
    prefix="/monitoring",
    tags=['Monitoring']
)


@router.get("/drift/report", response_model=DriftReportResponse)
async def get_drift_report(request):
    """
    Возвращает текущий статус из DriftDetector.get_status()
    """
    drift_detector = getattr(request.app.state, "drift_detector", None)
    
    if not drift_detector:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Ошибка при загрузке drift detector"
        )
    
    result = drift_detector.get_status()
    return DriftReportResponse(
        status=result.get("status", "unknown"),
        detail=result.get("detail")
    )


@router.get("/drift/history", response_model=DriftHistoryResponse)
async def get_drift_history(request):
    """
    Возвращает историю дрейфов из Redis (ключи drift_alert:*)
    """
    redis_client = getattr(request.app.state, "redis_client", None)
    
    if not redis_client:
        return DriftHistoryResponse(history={}, count=0, status="redis_unavailable")
    
    try:
        # Получить все ключи drift_alert:*
        keys = await redis_client.keys("drift_alert:*")
        
        history = {}
        for key in keys:
            try:
                value = await redis_client.get(key)
                key_str = key.decode() if isinstance(key, bytes) else key
                value_str = value.decode() if isinstance(value, bytes) else value
                history[key_str] = value_str
            except Exception:
                continue
        
        return DriftHistoryResponse(history=history, count=len(history), status="ok")
    except Exception as e:
        # Redis недоступен, возвращаем пустую историю
        return DriftHistoryResponse(
            history={}, count=0, status="redis_unavailable", error=str(e)
        )
