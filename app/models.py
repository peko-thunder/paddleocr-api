from pydantic import BaseModel
from typing import Optional


class BoundingBox(BaseModel):
    """Bounding box coordinates for detected text."""

    points: list[list[float]]


class OCRResult(BaseModel):
    """Individual OCR result."""

    text: str
    confidence: float
    bounding_box: BoundingBox


class OCRResponse(BaseModel):
    """OCR API response."""

    success: bool
    results: list[OCRResult]
    full_text: str
    message: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    ocr_ready: bool
    version: str
