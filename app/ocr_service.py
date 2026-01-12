import io

import numpy as np
from paddleocr import PaddleOCR
from PIL import Image


class OCRService:
    """PaddleOCR wrapper service."""

    def __init__(self, lang: str = "japan"):
        """
        Initialize OCR service.

        Args:
            lang: Language setting ("japan", "ch", "en", etc.)
        """
        self.ocr = PaddleOCR(
            lang=lang,
            use_textline_orientation=True,
        )
        self._initialized = True

    def process_image(self, image_bytes: bytes) -> list:
        """
        Perform OCR on image bytes.

        Args:
            image_bytes: Binary image data

        Returns:
            List of OCR results in format: [[bbox, (text, confidence)], ...]
        """
        image = Image.open(io.BytesIO(image_bytes))

        if image.mode != "RGB":
            image = image.convert("RGB")

        image_array = np.array(image)

        result = self.ocr.predict(image_array)

        # Convert new API format to legacy format for compatibility
        if not result:
            return []

        # predict() returns a list, get first element
        item = result[0] if result else None
        if not item:
            return []

        # New API returns dict with rec_texts, rec_scores, rec_polys
        if isinstance(item, dict):
            texts = item.get("rec_texts", [])
            scores = item.get("rec_scores", [])
            polys = item.get("rec_polys", [])

            legacy_results = []
            for i, text in enumerate(texts):
                score = scores[i] if i < len(scores) else 0.0
                poly = polys[i].tolist() if i < len(polys) else []
                legacy_results.append([poly, (text, score)])

            return legacy_results

        # Fallback for old API format (list of [bbox, (text, conf)])
        return item if item else []

    @property
    def is_ready(self) -> bool:
        """Check if OCR service is initialized."""
        return self._initialized
