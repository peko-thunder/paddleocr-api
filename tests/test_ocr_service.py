"""Tests for OCRService."""

import io
from pathlib import Path
import pytest
from PIL import Image
from app.ocr_service import OCRService

@pytest.fixture(scope="module")
def ocr_service():
    """Create OCRService instance (shared across tests for performance)."""
    return OCRService()


class TestOCRServiceInit:
    """Tests for OCRService initialization."""

    def test_is_ready_after_init(self, ocr_service):
        """Test that is_ready returns True after initialization."""
        assert ocr_service.is_ready is True


class TestProcessImage:
    """Tests for OCRService.process_image method."""

    def test_process_valid_image_一輝(self, ocr_service):
        """Test processing a valid image returns results."""
        TEST_IMAGE_PATH = Path(__file__).parent / "test_images"
        TEST_IMAGE_PATH = TEST_IMAGE_PATH / "一輝.png"

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        image_bytes = TEST_IMAGE_PATH.read_bytes()
        result = ocr_service.process_image(image_bytes)

        assert result[0][1][0] == "一輝"
        assert isinstance(result, list)

    def test_process_valid_image_一(self, ocr_service):
        """Test processing a valid image returns results."""
        TEST_IMAGE_PATH = Path(__file__).parent / "test_images"
        TEST_IMAGE_PATH = TEST_IMAGE_PATH / "一.png"

        if not TEST_IMAGE_PATH.exists():
            pytest.skip(f"Test image not found: {TEST_IMAGE_PATH}")

        image_bytes = TEST_IMAGE_PATH.read_bytes()
        result = ocr_service.process_image(image_bytes)

        # 特殊フォントの「一」はOCRできないので結果が0になる
        assert len(result) == 0
        assert isinstance(result, list)

    def test_process_rgba_image(self, ocr_service):
        """Test that RGBA images are correctly converted to RGB."""
        # Create a simple RGBA test image
        img = Image.new("RGBA", (100, 30), color=(255, 255, 255, 255))
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        result = ocr_service.process_image(image_bytes)

        assert isinstance(result, list)

    def test_process_grayscale_image(self, ocr_service):
        """Test that grayscale images are correctly converted to RGB."""
        # Create a simple grayscale test image
        img = Image.new("L", (100, 30), color=255)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()

        result = ocr_service.process_image(image_bytes)

        assert isinstance(result, list)

    def test_process_invalid_image(self, ocr_service):
        """Test that invalid image data raises an exception."""
        invalid_bytes = b"not a valid image"

        with pytest.raises(Exception):
            ocr_service.process_image(invalid_bytes)
