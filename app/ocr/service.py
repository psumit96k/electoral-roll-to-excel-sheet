"""
OCR service using PaddleOCR with CPU/GPU selection.
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
    Service for performing OCR using PaddleOCR with configurable device selection.
    """

    def __init__(self, settings=None):
        """Initialize the OCR service with PaddleOCR.

        Args:
            settings: Optional settings object. If not provided, uses the global settings.
        """
        self.ocr = None
        if settings is None:
            from app.config.settings import settings
            self.settings = settings
        else:
            self.settings = settings
        self._initialize_ocr()

    def _initialize_ocr(self):
        """Initialize PaddleOCR with configured parameters and device selection."""
        try:
            # Resolve device based on OCR_DEVICE setting
            device = self._resolve_ocr_device()

            # Prepare PaddleOCR parameters
            paddleocr_params = {
                "use_doc_orientation_classify": self.settings.OCR_USE_DOC_ORIENTATION_CLASSIFY,
                "use_doc_unwarping": self.settings.OCR_USE_DOC_UNWARPING,
                "use_textline_orientation": self.settings.OCR_USE_TEXTLINE_ORIENTATION,
                "lang": self.settings.OCR_LANG,
                "enable_mkldnn": self.settings.OCR_ENABLE_MKLDNN
            }

            # Add device parameter if specified (not None)
            if device is not None:
                paddleocr_params["device"] = device

            self.ocr = PaddleOCR(**paddleocr_params)
            logger.info(f"PaddleOCR initialized successfully on device: {device or 'default'}")
        except Exception as e:
            logger.error(f"Failed to initialize PaddleOCR: {e}")
            raise

    def _resolve_ocr_device(self) -> Optional[str]:
        """
        Resolve the OCR device based on settings.

        Returns:
            Device string for PaddleOCR (e.g., 'cpu', 'gpu:0') or None for default

        Raises:
            ValueError: If OCR_DEVICE setting is invalid
            RuntimeError: If GPU is requested but not available
        """
        device_setting = self.settings.OCR_DEVICE.lower()

        # Validate device setting
        if device_setting not in ("cpu", "gpu", "auto"):
            raise ValueError(
                f"Invalid OCR_DEVICE setting: '{self.settings.OCR_DEVICE}'. "
                f"Must be one of: 'cpu', 'gpu', 'auto'"
            )

        # Log the requested mode
        logger.info(f"OCR device setting requested: {device_setting}")

        if device_setting == "cpu":
            logger.info("Selected CPU device explicitly")
            return "cpu"

        elif device_setting == "gpu":
            # Check if GPU is available
            if self._is_gpu_available():
                logger.info("Selected GPU device explicitly")
                return "gpu:0"
            else:
                raise RuntimeError(
                    "GPU requested for OCR but GPU is not available. "
                    "Please ensure: "
                    "1. CUDA-enabled PaddlePaddle is installed, "
                    "2. CUDA drivers are available, "
                    "3. At least one GPU is detected."
                )

        else:  # device_setting == "auto"
            # Auto-select: use GPU if available, otherwise CPU
            if self._is_gpu_available():
                logger.info("Auto-selected GPU device (CUDA and GPU available)")
                return "gpu:0"
            else:
                logger.info("Auto-selected CPU device (GPU not available)")
                return "cpu"

    def _is_gpu_available(self) -> bool:
        """
        Check if GPU is available for PaddleOCR.

        Returns:
            True if GPU is available, False otherwise
        """
        try:
            import paddle
            # Check if PaddlePaddle was compiled with CUDA
            if not paddle.is_compiled_with_cuda():
                logger.debug("PaddlePaddle not compiled with CUDA")
                return False

            # Check GPU count
            gpu_count = paddle.device.cuda.device_count()
            logger.debug(f"CUDA is available with {gpu_count} GPU(s)")
            return gpu_count > 0
        except Exception as e:
            logger.debug(f"Error checking GPU availability: {e}")
            return False

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