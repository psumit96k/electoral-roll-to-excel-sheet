"""
Geometry handling for electoral roll PDF parsing.
Implements the exact 30-slot geometry from the specification.
"""
from typing import List, Tuple, Dict, Optional, Any
import logging

from app.config import settings

logger = logging.getLogger(__name__)


class GeometryParser:
    """
    Handle the geometry parsing for electoral roll cards.
    Implements the fixed 30-slot layout from the specification.
    """

    # Reference dimensions from specification
    REFERENCE_WIDTH = 1447
    REFERENCE_HEIGHT = 2048

    # Column bounds (left, right) for each of the 3 columns
    COLUMN_BOUNDS = [
        (34, 485),   # Column 1
        (496, 949),  # Column 2
        (959, 1411)  # Column 3
    ]

    # Row bounds (top, bottom) for each of the 10 rows
    ROW_BOUNDS = [
        (68, 250),   # Row 1
        (261, 443),  # Row 2
        (454, 637),  # Row 3
        (647, 829),  # Row 4
        (840, 1022), # Row 5
        (1032, 1215), # Row 6
        (1226, 1408), # Row 7
        (1419, 1601), # Row 8
        (1611, 1794), # Row 9
        (1805, 1987)  # Row 10
    ]

    def __init__(self):
        """Initialize the geometry parser."""
        pass

    def get_card_slot_bounds(
        self,
        row: int,
        column: int,
        actual_width: int,
        actual_height: int
    ) -> Tuple[int, int, int, int]:
        """
        Get the bounding box for a specific card slot.

        Args:
            row: Row number (1-10)
            column: Column number (1-3)
            actual_width: Actual width of the source image
            actual_height: Actual height of the source image

        Returns:
            Tuple of (left, top, right, bottom) in actual image coordinates
        """
        if not (1 <= row <= 10):
            raise ValueError(f"Row must be between 1 and 10, got {row}")
        if not (1 <= column <= 3):
            raise ValueError(f"Column must be between 1 and 3, got {column}")

        # Get reference bounds
        ref_left, ref_right = self.COLUMN_BOUNDS[column - 1]
        ref_top, ref_bottom = self.ROW_BOUNDS[row - 1]

        # Calculate scaling factors
        scale_x = actual_width / self.REFERENCE_WIDTH
        scale_y = actual_height / self.REFERENCE_HEIGHT

        # Apply scaling
        left = int(ref_left * scale_x)
        top = int(ref_top * scale_y)
        right = int(ref_right * scale_x)
        bottom = int(ref_bottom * scale_y)

        # Ensure bounds are within image
        left = max(0, min(left, actual_width))
        top = max(0, min(top, actual_height))
        right = max(left + 1, min(right, actual_width))
        bottom = max(top + 1, min(bottom, actual_height))

        return (left, top, right, bottom)

    def get_all_card_slots(
        self,
        actual_width: int,
        actual_height: int
    ) -> List[Dict[str, Any]]:
        """
        Get bounding boxes for all 30 card slots.

        Args:
            actual_width: Actual width of the source image
            actual_height: Actual height of the source image

        Returns:
            List of dictionaries with slot information
        """
        slots = []
        slot_number = 1

        for row in range(1, 11):  # Rows 1-10
            for column in range(1, 4):  # Columns 1-3
                bounds = self.get_card_slot_bounds(row, column, actual_width, actual_height)
                left, top, right, bottom = bounds

                slot_info = {
                    "slot_number": slot_number,
                    "physical_slot": slot_number,  # Same as slot_number for now
                    "row": row,
                    "column": column,
                    "bounds": bounds,
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                    "width": right - left,
                    "height": bottom - top
                }
                slots.append(slot_info)
                slot_number += 1

        return slots

    def validate_geometry_match(
        self,
        actual_width: int,
        actual_height: int
    ) -> Dict[str, Any]:
        """
        Validate that the image matches the expected voter card geometry.

        Args:
            actual_width: Actual width of the source image
            actual_height: Actual height of the source image

        Returns:
            Dictionary with validation results
        """
        # Calculate expected dimensions based on aspect ratio
        expected_ratio = self.REFERENCE_WIDTH / self.REFERENCE_HEIGHT
        actual_ratio = actual_width / actual_height

        # Allow some tolerance for scaling differences
        ratio_tolerance = 0.05  # 5% tolerance
        ratio_match = abs(expected_ratio - actual_ratio) / expected_ratio <= ratio_tolerance

        # Check if dimensions are reasonable (not extremely small or large)
        reasonable_size = (
            1000 <= actual_width <= 4000 and
            1000 <= actual_height <= 4000
        )

        return {
            "width": actual_width,
            "height": actual_height,
            "expected_ratio": expected_ratio,
            "actual_ratio": actual_ratio,
            "ratio_match": ratio_match,
            "reasonable_size": reasonable_size,
            "geometry_valid": ratio_match and reasonable_size,
            "scale_x": actual_width / self.REFERENCE_WIDTH,
            "scale_y": actual_height / self.REFERENCE_HEIGHT
        }

    def adjust_for_alignment(
        self,
        bounds: Tuple[int, int, int, int],
        actual_width: int,
        actual_height: int,
        alignment_offset: Tuple[int, int] = (0, 0)
    ) -> Tuple[int, int, int, int]:
        """
        Adjust bounds for alignment correction.

        Args:
            bounds: Original bounds (left, top, right, bottom)
            actual_width: Actual image width
            actual_height: Actual image height
            alignment_offset: (x_offset, y_offset) to apply

        Returns:
            Adjusted bounds
        """
        left, top, right, bottom = bounds
        x_offset, y_offset = alignment_offset

        # Apply offset
        adjusted_left = max(0, left + x_offset)
        adjusted_top = max(0, top + y_offset)
        adjusted_right = min(actual_width, right + x_offset)
        adjusted_bottom = min(actual_height, bottom + y_offset)

        # Ensure valid bounds
        if adjusted_left >= adjusted_right or adjusted_top >= adjusted_bottom:
            logger.warning("Alignment adjustment resulted in invalid bounds")
            return bounds  # Return original if adjustment invalid

        return (adjusted_left, adjusted_top, adjusted_right, adjusted_bottom)


# Global geometry parser instance
geometry_parser = GeometryParser()