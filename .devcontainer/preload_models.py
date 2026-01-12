#!/usr/bin/env python3
"""
PaddleOCR model pre-download script for Docker build.

This script initializes PaddleOCR to download all required models
during Docker image build, so they are available at runtime.
"""

from paddleocr import PaddleOCR

print("Starting PaddleOCR model download...")
print("This may take a few minutes on first run...")

# Initialize PaddleOCR with the same settings used in production
# This will download detection, recognition, and angle classification models
ocr = PaddleOCR(
    lang="japan",
    use_textline_orientation=True,
)

print("✓ PaddleOCR models successfully downloaded and cached")
print("Models are stored in: ~/.paddlex/official_models/")
