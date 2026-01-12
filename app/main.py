from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from app.models import BoundingBox, HealthResponse, OCRResponse, OCRResult
from app.ocr_service import OCRService

ocr_service: OCRService | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle management."""
    global ocr_service
    print("Initializing OCR service...")
    ocr_service = OCRService(lang="japan")
    print("OCR service initialized successfully")
    yield
    ocr_service = None


app = FastAPI(
    title="PaddleOCR API",
    description="Japanese OCR Web API powered by PaddleOCR",
    version="1.0.0",
    lifespan=lifespan,
)

ALLOWED_CONTENT_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/bmp",
    "image/webp",
]


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint (simple health check)."""
    return HealthResponse(
        status="ok",
        ocr_ready=ocr_service is not None and ocr_service.is_ready,
        version="1.0.0",
    )


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Detailed health check."""
    return HealthResponse(
        status="ok",
        ocr_ready=ocr_service is not None and ocr_service.is_ready,
        version="1.0.0",
    )


@app.post("/ocr", response_model=OCRResponse)
async def perform_ocr(file: UploadFile = File(...)):
    """
    Perform OCR on uploaded image file.

    - **file**: Image file (JPEG, PNG, GIF, BMP, WebP)

    Returns:
        OCRResponse: Recognized text and coordinate information
    """
    if ocr_service is None:
        raise HTTPException(status_code=503, detail="OCR service is not initialized")

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. "
            f"Allowed types: {', '.join(ALLOWED_CONTENT_TYPES)}",
        )

    try:
        contents = await file.read()

        ocr_results = ocr_service.process_image(contents)

        results: list[OCRResult] = []
        texts: list[str] = []

        for item in ocr_results:
            if item is None:
                continue

            bbox, (text, confidence) = item
            results.append(
                OCRResult(
                    text=text,
                    confidence=confidence,
                    bounding_box=BoundingBox(points=bbox),
                )
            )
            texts.append(text)

        return OCRResponse(
            success=True,
            results=results,
            full_text="\n".join(texts),
            message=f"Detected {len(results)} text regions",
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OCR processing failed: {e!s}")
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
