"""
Excel export functionality using openpyxl.
Generates exact Excel format as per specification.
"""
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.workbook.properties import WorkbookProperties
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

from app.config import settings
from app.models.database import (
    Document, ReviewItem, CardSlot, Page,
    ReviewStatusEnum
)
from app.database import SessionLocal

logger = logging.getLogger(__name__)


class ExcelExporter:
    """
    Generate Excel exports in the exact format specified.
    """

    def __init__(self):
        """Initialize the Excel exporter."""
        pass

    def create_workbook(self) -> Workbook:
        """
        Create a new Excel workbook with the exact format specified.

        Returns:
            Configured Workbook object
        """
        wb = Workbook()

        # Remove default sheet and create "Voters" sheet
        wb.remove(wb.active)
        ws = wb.create_sheet(title="Voters")

        # Set up headers exactly as specified
        headers = [
            "Row No.",      # A1
            "Full Name",    # B1
            "Father's Name", # C1
            "Husband's Name", # D1
            "House Number", # E1
            "Relation Type", # F1
            "Relative Name", # G1
            "Gender",       # H1
            "Age",          # I1
            "Caste",        # J1
            "Phone Number", # K1
            "EPIC"          # L1
        ]

        # Write headers
        for col_num, header in enumerate(headers, 1):
            cell = ws.cell(row=1, column=col_num, value=header)
            # Apply header formatting: bold white text on dark background
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="366092", end_color="366092", fill_type="solid")
            cell.alignment = Alignment(horizontal="center", vertical="center")

        # Set column widths as specified
        column_widths = {
            1: 10,   # A - Row No.
            2: 36,   # B - Full Name
            3: 36,   # C - Father's Name
            4: 36,   # D - Husband's Name
            5: 15,   # E - House Number
            6: 15,   # F - Relation Type
            7: 36,   # G - Relative Name
            8: 12,   # H - Gender
            9: 10,   # I - Age
            10: 22,  # J - Caste
            11: 22,  # K - Phone Number
            12: 18   # L - EPIC
        }

        for col_num, width in column_widths.items():
            ws.column_dimensions[get_column_letter(col_num)].width = width

        # Freeze panes at A2
        ws.freeze_panes = "A2"

        # Set default font to 11-point for data rows (formatting will be applied as data is added)
        pass

        # Add light borders
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Apply borders to data rows (will be applied as data is added)
        ws._current_border = thin_border

        # Set document properties on existing wb.properties
        wb.properties.title = "Election Roll Export"
        wb.properties.creator = "Election PDF to Excel Converter"
        wb.properties.created = datetime.now()

        return wb

    def add_data_rows(
        self,
        wb: Workbook,
        reviewed_data: List[Dict[str, Any]],
        start_row: int = 2
    ) -> int:
        """
        Add reviewed data rows to the worksheet.

        Args:
            wb: Workbook to add data to
            reviewed_data: List of dictionaries containing reviewed data
            start_row: Row number to start adding data (default 2)

        Returns:
            Next available row number after adding data
        """
        ws = wb["Voters"]
        current_row = start_row

        # Get the thin border style
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )

        # Process records in order (they should already be sorted by document, page, row, column)
        # Assign continuous output numbering
        output_idx = 1
        for record in reviewed_data:
            # Row No. (continuous output numbering after filtering)
            ws.cell(row=current_row, column=1, value=output_idx)
            output_idx += 1

            # Full Name
            full_name = record.get('reviewed_full_name')
            ws.cell(row=current_row, column=2, value=full_name if full_name else None)

            # Father's Name (only when relation is Father)
            father_name = None
            if record.get('reviewed_relation_type') == 'FATHER':
                father_name = record.get('reviewed_relative_name')
            ws.cell(row=current_row, column=3, value=father_name if father_name else None)

            # Husband's Name (only when relation is Husband)
            husband_name = None
            if record.get('reviewed_relation_type') == 'HUSBAND':
                husband_name = record.get('reviewed_relative_name')
            ws.cell(row=current_row, column=4, value=husband_name if husband_name else None)

            # House Number (explicitly as text)
            house_number = record.get('reviewed_house_number')
            ws.cell(row=current_row, column=5, value=str(house_number) if house_number is not None else None)

            # Relation Type
            relation_type = record.get('reviewed_relation_type')
            ws.cell(row=current_row, column=6, value=relation_type if relation_type else None)

            # Relative Name
            relative_name = record.get('reviewed_relative_name')
            ws.cell(row=current_row, column=7, value=relative_name if relative_name else None)

            # Gender
            gender = record.get('reviewed_gender')
            ws.cell(row=current_row, column=8, value=gender if gender else None)

            # Age (as integer)
            age = record.get('reviewed_age')
            ws.cell(row=current_row, column=9, value=int(age) if age is not None else None)

            # Caste
            caste = record.get('reviewed_caste')
            ws.cell(row=current_row, column=10, value=caste if caste else None)

            # Phone Number (as string to preserve formatting)
            phone = record.get('reviewed_phone_number')
            ws.cell(row=current_row, column=11, value=str(phone) if phone is not None else None)

            # EPIC (as string to preserve leading zeros)
            epic = record.get('reviewed_epic')
            ws.cell(row=current_row, column=12, value=str(epic) if epic is not None else None)

            # Apply data formatting: 11-point font, text alignment, borders
            for col_num in range(1, 13):  # Columns A-L
                cell = ws.cell(row=current_row, column=col_num)
                cell.font = Font(size=11)

                # Set alignment based on column type
                if col_num in [1, 8, 9]:  # Row No., Gender, Age - center align
                    cell.alignment = Alignment(horizontal="center", vertical="center")
                elif col_num == 11:  # Phone Number - left align to preserve leading zeros
                    cell.alignment = Alignment(horizontal="left", vertical="center")
                else:  # Name and text fields - left align
                    cell.alignment = Alignment(horizontal="left", vertical="center")

                # Apply borders
                cell.border = thin_border

                # Handle special data types
                if col_num in [5, 11, 12]:  # House Number, Phone and EPIC - ensure text format
                    if cell.value is not None:
                        # Already stored as string, but ensure it's treated as text
                        cell.value = str(cell.value)

            current_row += 1

        return current_row

    def apply_auto_filter(self, ws, last_row: int):
        """
        Apply auto-filter across the data range.

        Args:
            ws: Worksheet to apply filter to
            last_row: Last row containing data
        """
        if last_row >= 2:  # Only if we have data rows
            ws.auto_filter.ref = f"A1:L{last_row}"

    def generate_excel_bytes(self, reviewed_data: List[Dict[str, Any]]) -> bytes:
        """
        Generate Excel file as bytes.

        Args:
            reviewed_data: List of reviewed data records

        Returns:
            Excel file as bytes
        """
        wb = self.create_workbook()
        last_row = self.add_data_rows(wb, reviewed_data)

        # Apply auto-filter
        ws = wb["Voters"]
        self.apply_auto_filter(ws, last_row - 1)  # last_row is next available, so -1 for last data row

        # Save to bytes
        from io import BytesIO
        buffer = BytesIO()
        wb.save(buffer)
        buffer.seek(0)
        return buffer.getvalue()

    def generate_excel_file(
        self,
        reviewed_data: List[Dict[str, Any]],
        output_path: str
    ) -> str:
        """
        Generate Excel file and save to disk.

        Args:
            reviewed_data: List of reviewed data records
            output_path: Path to save the Excel file

        Returns:
            Path to the generated file
        """
        wb = self.create_workbook()
        last_row = self.add_data_rows(wb, reviewed_data)

        # Apply auto-filter
        ws = wb["Voters"]
        self.apply_auto_filter(ws, last_row - 1)

        # Save file
        wb.save(output_path)
        return output_path


def generate_excel_export(
    document_ids: List[int],
    output_path: str,
    include_unconfirmed: bool = False,
    db_session: Optional[SessionLocal] = None
) -> Dict[str, Any]:
    """
    Background task to generate Excel export from reviewed data.

    Args:
        document_ids: List of document IDs to include in export
        output_path: Path where Excel file should be saved
        include_unconfirmed: Whether to include unreviewed items
        db_session: Database session (will create one if not provided)

    Returns:
        Dictionary with export results
    """
    close_session = False
    if db_session is None:
        db_session = SessionLocal()
        close_session = True

    try:
        logger.info(f"Starting Excel export for documents: {document_ids}")

        # Query reviewed data
        from sqlalchemy import and_

        # Build query for review items
        query = db_session.query(
            ReviewItem,
            CardSlot,
            Page,
            Document
        ).join(
            CardSlot, ReviewItem.card_slot_id == CardSlot.id
        ).join(
            Page, CardSlot.page_id == Page.id
        ).join(
            Document, Page.document_id == Document.id
        ).filter(
            Document.id.in_(document_ids)
        ).order_by(
            Document.id,      # Document
            Page.page_number, # Page number
            CardSlot.row_number, # Row
            CardSlot.column_number  # Column
        )

        # Filter by review status unless including unconfirmed
        if not include_unconfirmed:
            query = query.filter(
                ReviewItem.review_status.in_([
                    ReviewStatusEnum.REVIEWED.value,
                    ReviewStatusEnum.CONFIRMED.value
                ])
            )
        else:
            # Include everything except explicitly excluded
            query = query.filter(
                ReviewItem.review_status != ReviewStatusEnum.EXCLUDED.value
            )

        results = query.all()

        # Convert to the format needed for Excel export
        reviewed_data = []
        for review_item, card_slot, page, document in results:
            record = {
                'reviewed_full_name': review_item.reviewed_full_name,
                'reviewed_house_number': review_item.reviewed_house_number,
                'reviewed_relation_type': review_item.reviewed_relation_type,
                'reviewed_relative_name': review_item.reviewed_relative_name,
                'reviewed_gender': review_item.reviewed_gender,
                'reviewed_age': review_item.reviewed_age,
                'reviewed_caste': review_item.reviewed_caste,
                'reviewed_phone_number': review_item.reviewed_phone_number,
                'reviewed_epic': review_item.reviewed_epic
            }

            reviewed_data.append(record)

        logger.info(f"Found {len(reviewed_data)} reviewed records for export")

        # Generate Excel file
        exporter = ExcelExporter()
        exporter.generate_excel_file(reviewed_data, output_path)

        logger.info(f"Excel export completed: {output_path}")

        return {
            "success": True,
            "record_count": len(reviewed_data),
            "output_path": output_path,
            "message": f"Export generated successfully with {len(reviewed_data)} records"
        }

    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        return {
            "success": False,
            "error": str(e),
            "record_count": 0,
            "output_path": output_path
        }
    finally:
        if close_session:
            db_session.close()


# Convenience function for direct use
def export_to_excel(
    reviewed_data: List[Dict[str, Any]],
    output_path: str
) -> str:
    """
    Export reviewed data to Excel file.

    Args:
        reviewed_data: List of reviewed data records
        output_path: Path to save the Excel file

    Returns:
        Path to the generated file
    """
    exporter = ExcelExporter()
    return exporter.generate_excel_file(reviewed_data, output_path)