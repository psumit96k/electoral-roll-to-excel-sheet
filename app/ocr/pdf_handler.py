"""
PDF handling utilities for converting PDF pages to images.
"""
import fitz  # PyMuPDF
import numpy as np
from typing import List, Tuple, Optional
from pathlib import Path
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class PDFHandler:
    """
    Handle PDF processing - convert PDF pages to images for OCR.
    """

    def __init__(self, dpi: int = 150):
        """
        Initialize PDF handler.

        Args:
            dpi: Resolution for rendering PDF pages (default 150 DPI for OCR)
        """
        self.dpi = dpi

    def pdf_to_images(
        self,
        pdf_path: str,
        page_numbers: Optional[List[int]] = None
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Convert PDF pages to numpy arrays (images).

        Args:
            pdf_path: Path to the PDF file
            page_numbers: List of page numbers to convert (1-indexed, None for all)

        Returns:
            List of tuples (page_number, image_array)
        """
        try:
            doc = fitz.open(pdf_path)
            images = []

            # Determine which pages to process
            total_pages = len(doc)
            if page_numbers is None:
                page_range = range(total_pages)
            else:
                # Convert to 0-indexed and validate
                page_range = [p-1 for p in page_numbers if 1 <= p <= total_pages]

            for page_num in page_range:
                page = doc.load_page(page_num)

                # Calculate transformation matrix for desired DPI
                # Default DPI of PDF is 72, so scale factor = desired_dpi / 72
                scale_factor = self.dpi / 72.0
                mat = fitz.Matrix(scale_factor, scale_factor)

                # Render page to pixmap
                pix = page.get_pixmap(matrix=mat, alpha=False)

                # Convert to numpy array
                img_data = pix.samples
                img = np.frombuffer(img_data, dtype=np.uint8).reshape(
                    pix.height, pix.width, pix.n
                )

                # Convert RGBA to RGB if needed
                if pix.n == 4:  # RGBA
                    img = img[:, :, :3]  # Drop alpha channel

                images.append((page_num + 1, img))  # Store as 1-indexed page number

                # Clean up
                pix = None

            doc.close()
            return images

        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise

    def get_page_dimensions(self, pdf_path: str) -> List[Tuple[int, int, int]]:
        """
        Get dimensions of all pages in a PDF.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            List of tuples (page_number, width, height) in points
        """
        try:
            doc = fitz.open(pdf_path)
            dimensions = []

            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                rect = page.rect
                dimensions.append((page_num + 1, int(rect.width), int(rect.height)))

            doc.close()
            return dimensions

        except Exception as e:
            logger.error(f"Failed to get PDF page dimensions: {e}")
            raise

    def extract_page_as_image(
        self,
        pdf_path: str,
        page_number: int,
        dpi: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract a single page as an image.

        Args:
            pdf_path: Path to the PDF file
            page_number: Page number to extract (1-indexed)
            dpi: Resolution for rendering (uses instance dpi if None)

        Returns:
            Image as numpy array
        """
        if dpi is None:
            dpi = self.dpi

        images = self.pdf_to_images(pdf_path, [page_number])
        if images:
            return images[0][1]
        else:
            raise ValueError(f"Could not extract page {page_number} from PDF")


# Global PDF handler instance
pdf_handler = PDFHandler(dpi=150)