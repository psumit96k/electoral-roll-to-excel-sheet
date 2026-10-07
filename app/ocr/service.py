"""
OCR service using PaddleOCR.
"""
import cv2
import numpy as np
from paddleocr import PaddleOCR
from typing import List, Dict, Any, Optional, Tuple
import logging
from pathlib import Path

from app.config.settings import settings

logger = logging.getLogger(__name__)


class OCRService:
    """
    Service for performing OCR using PaddleOCR.
    """

    def __init__(self):
        """Initialize the OCR service with PaddleOCR."""
        self.ocr = None
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize PaddleOCR with configured parameters."""
        try:
            self.ocr = PaddleOCR(
                use_doc_orientation_classify=settings.OCR_USE_DOC_ORIENTATION_CLASSIFY,
                use_doc_unwarping=settings.OCR_USE_DOC_UNWARPING,
                use_textline_orientation=settings.OCR_USE_TEXTLINE_ORIENTATION,
                lang=settings.OCR_LANG,
                enable_mkldnn=settings.OCR_ENABLE_MKLDNN
            )
            logger.info("PaddleOCR initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better OCR results.
        Applies conservative deskewing, denoising and contrast adjustment.
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        # Apply conservative denoising
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

        # Apply contrast adjustment using CLAHE
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)

        # Convert back to BGR for PaddleOCR (expects color image)
        if len(image.shape) == 3:
            result = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
        else:
            result = enhanced

        return result

    def _normalize_ocr_result(self, detection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Normalize a single detection dict from PaddleOCR.predict into a list of OCR result dicts.

        Args:
            detection: A dictionary from the list returned by PaddleOCR.predict().
                       Expected to have keys: 'rec_texts', 'rec_scores', 'rec_polys'
                       (each being a list, or None/missing).

        Returns:
            A list of OCR result dictionaries, each with keys: 'text', 'confidence', 'bbox'.
            'bbox' is a list of 4 points [x, y] or None if unavailable.
        """
        # Extract arrays from the detection dict
        rec_texts = detection.get('rec_texts', [])
        rec_scores = detection.get('rec_scores')
        rec_polys = detection.get('rec_polys')

        text_len = len(rec_texts)

        # Handle scores: if None or missing, use None for all; else validate length
        if rec_scores is None:
            scores = [None] * text_len
        else:
            if len(rec_scores) != text_len:
                raise ValueError(
                    f"Score array length {len(rec_scores)} does not match text array length {text_len}"
                )
            scores = rec_scores

        # Handle polys: if None or missing, use None for all; else validate length
        if rec_polys is None:
            polys = [None] * text_len
        else:
            if len(rec_polys) != text_len:
                raise ValueError(
                    f"Polygon array length {len(rec_polys)} does not match text array length {text_len}"
                )
            polys = rec_polys

        ocr_results = []
        for i in range(text_len):
            text = rec_texts[i]
            score = scores[i]
            poly = polys[i]

            # Convert numpy types to Python native
            if text is not None and hasattr(text, 'item'):
                text = text.item()
            if score is not None and hasattr(score, 'item'):
                score = score.item()
            if poly is not None:
                if hasattr(poly, 'tolist'):
                    poly = poly.tolist()
                # Ensure it's a list of 4 points
                if isinstance(poly, list) and len(poly) == 4:
                    poly = [list(map(float, p)) for p in poly]
                else:
                    logger.warning(f"Unexpected polygon format in detection: {poly}")

            # Build OCR result for this text/score/poly triple
            ocr_results.append({
                "text": str(text).strip() if text is not None else None,
                "confidence": float(score) if score is not None else None,
                "bbox": poly  # polygon as list of 4 points, or None
            })

        return ocr_results

    def extract_text_from_region(
        self,
        image: np.ndarray,
        region: Tuple[int, int, int, int]
    ) -> List[Dict[str, Any]]:
        """
        Extract text from a specific region of an image.

        Args:
            image: Input image as numpy array
            region: Tuple of (left, top, right, bottom) coordinates

        Returns:
            List of OCR results with text, confidence, and bounding box.
            Each result is a dictionary with keys: text, confidence, bbox, region.
            If polygon is unavailable, bbox is None.
            If scores/polys are missing, they are None.
        """
        if self.ocr is None:
            raise RuntimeError("OCR service not initialized")

        # Validate image
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        # Validate region
        left, top, right, bottom = region
        h, w = image.shape[:2]
        if left < 0 or top < 0 or right > w or bottom > h or left >= right or top >= bottom:
            raise ValueError(f"Region {region} is out of bounds for image shape ({h}, {w})")

        # Extract region
        roi = image[top:bottom, left:right]

        # Preprocess the region
        processed_roi = self.preprocess_image(roi)

        # Perform OCR using predict
        try:
            predict_result = self.ocr.predict(processed_roi)
            logger.debug(f"PaddleOCR predict result (region): {predict_result}")

            # Normalize each detection
            normalized = []
            for detection in predict_result:
                normalized.extend(self._normalize_ocr_result(detection))

            # Offset bounding box coordinates by region left/top and set region
            for result in normalized:
                result["region"] = region
                bbox = result["bbox"]
                if bbox is not None:
                    offset_bbox = []
                    for point in bbox:
                        offset_bbox.append([point[0] + left, point[1] + top])
                    result["bbox"] = offset_bbox

            return normalized
        except Exception as e:
            logger.exception(f"OCR processing failed for region {region}")
            raise

    def extract_text_from_image(
        self,
        image: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Extract text from entire image.

        Args:
            image: Input image as numpy array

        Returns:
            List of OCR results with text, confidence, and bounding box.
            Each result is a dictionary with keys: text, confidence, bbox.
            If polygon is unavailable, bbox is None.
            If scores/polys are missing, they are None.
        """
        if self.ocr is None:
            raise RuntimeError("OCR service not initialized")

        # Validate image
        if image is None or image.size == 0:
            raise ValueError("Input image is empty")

        # Preprocess the image
        processed_image = self.preprocess_image(image)

        # Perform OCR using predict
        try:
            predict_result = self.ocr.predict(processed_image)
            logger.debug(f"PaddleOCR predict result (full image): {predict_result}")

            # Normalize each detection
            normalized = []
            for detection in predict_result:
                normalized.extend(self._normalize_ocr_result(detection))

            return normalized
        except Exception as e:
            logger.exception("OCR processing failed for full image")
            raise

    def close(self):
        """Clean up resources."""
        # PaddleOCR doesn't require explicit cleanup in most cases
        pass


# Global OCR service instance
ocr_service = OCRService()