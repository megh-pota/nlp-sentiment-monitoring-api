from pydantic import BaseModel, Field
from typing import Dict, Any


class SentimentRequest(BaseModel):
    text: str = Field(..., example="This movie was amazing!")


class SentimentResponse(BaseModel):
    sentiment: str
    confidence: float
    drift: Dict[str, Any]   # ✅ ADD THIS
