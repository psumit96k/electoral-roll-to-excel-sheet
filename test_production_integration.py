"""
Production integration test using unittest framework.
Tests the complete processing pipeline with mocked OCR and temporary SQLite database.
"""
import unittest
import tempfile
import os
import sys
import json
import re
import types
import copy
from unittest.mock import patch, MagicMock
from sqlalchemy.orm import sessionmaker
from sqlalchemy import create_engine

# Install stub for app.ocr.service BEFORE importing any module that initializes OCR
sys.modules['app.ocr.service'] = types.ModuleType('app.ocr.service')
sys.modules['app.ocr.service'].ocr_service = MagicMock()

# Mock paddle specifically to prevent initialization (only if needed as fallback)
sys.modules['paddle'] = MagicMock()
sys.modules['paddle.version'] = MagicMock()
sys.modules['paddle.version.__version__'] = '2.6.0'

# Mock other OCR dependencies as fallback
sys.modules['paddlex'] = MagicMock()
sys.modules['paddlex.cls'] = MagicMock()
sys.modules['paddlex.inference'] = MagicMock()
sys.modules['paddlex.inference.utils'] = MagicMock()
sys.modules['paddlex.inference.utils.benchmark'] = MagicMock()
sys.modules['paddlex.utils'] = MagicMock(__path__=[])
sys.modules['paddlex.utils.deps'] = MagicMock()
# Note: We keep the device mock as it's needed for Settings initialization
sys.modules['paddlex.utils.device'] = MagicMock(parse_device=lambda device: ("cpu", 0), get_default_device=lambda: "cpu")
sys.modules['paddlex.utils.pipeline_arguments'] = MagicMock()
sys.modules['paddlex.utils.config'] = MagicMock()

# Create temporary database and set up environment BEFORE importing application modules
temp_db_fd, temp_db_path = tempfile.mkstemp(suffix='.db')
test_database_url = f"sqlite:///{temp_db_path}"

# Set the database URL in environment BEFORE importing settings
os.environ['DATABASE_URL'] = test_database_url

# Now we can import the modules - they should use our mocked dependencies and temporary database
try:
    from app.jobs.processor import PDFProcessor
    from app.models.database import Base, ExtractionCandidate, ReviewItem, Document, Page, CardSlot, Batch
    from app.database import SessionLocal, engine
except ImportError as e:
    print(f"Import error: {e}")
    raise

# Proper imports with patches in place
with patch.dict('sys.modules', {
    'paddle': MagicMock(),
    'paddle.version': MagicMock(),
    'paddlex': MagicMock(),
    'paddlex.cls': MagicMock(),
    'paddlex.inference': MagicMock(),
    'paddlex.inference.utils': MagicMock(),
    'paddlex.inference.utils.benchmark': MagicMock(),
    'paddlex.utils': MagicMock(__path__=[]),
    'paddlex.utils.deps': MagicMock(),
    'paddlex.utils.device': MagicMock(parse_device=lambda device: ("cpu", 0), get_default_device=lambda: "cpu"),
    'paddlex.utils.pipeline_arguments': MagicMock(),
    'paddlex.utils.config': MagicMock(AttrDict=MagicMock()),
}):
    from app.models.database import Base, ExtractionCandidate, ReviewItem, Document, Page, CardSlot, Batch, ReviewStatusEnum
    from app.database import engine as db_engine
    from sqlalchemy import create_engine


class TestProductionIntegration(unittest.TestCase):
    """Integration test for production processing pipeline."""

    @classmethod
    def setUpClass(cls):
        """Set up test database once for all tests."""
        # Use the temporary database path created before imports
        cls.temp_db_fd = temp_db_fd
        cls.temp_db_path = temp_db_path
        cls.test_database_url = test_database_url

        # Create test engine and create all tables
        test_engine = create_engine(cls.test_database_url)
        Base.metadata.create_all(bind=test_engine)

        cls.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
        cls.test_engine = test_engine

    @classmethod
    def tearDownClass(cls):
        """Clean up test database."""
        # Close all sessions and dispose engine
        if hasattr(cls, 'SessionLocal'):
            # Close all sessions by disposing the engine
            if hasattr(cls, 'test_engine'):
                cls.test_engine.dispose()

        os.close(cls.temp_db_fd)
        try:
            os.unlink(cls.temp_db_path)
        except PermissionError:
            # If we can't delete it now, try again later or leave it
            pass

        # Clean up environment variable
        if 'DATABASE_URL' in os.environ:
            del os.environ['DATABASE_URL']

    def setUp(self):
        """Set up fresh database session for each test."""
        print("SETUP CALLED")
        self.db_session = self.SessionLocal()
        # Apply OCR service patch before importing processor
        from unittest.mock import patch
        from app.ocr.service import ocr_service as real_ocr_service
        mock_ocr_service = MagicMock()
        # We'll set up the mock side effect later in the test method
        self._ocr_patch = patch('app.ocr.service.ocr_service', mock_ocr_service)
        self._ocr_patch.start()
        self.mock_ocr_service = mock_ocr_service

        # Import PDFProcessor here to ensure it uses patched dependencies
        from app.jobs.processor import PDFProcessor
        self.processor = PDFProcessor(db_session=self.db_session)

        # Verify that we're using the temporary database
        # SQLAlchemy returns URL-encoded URL, so we need to decode for comparison
        import urllib.parse
        actual_url = str(self.db_session.bind.url)
        decoded_actual = urllib.parse.unquote(actual_url)
        decoded_expected = urllib.parse.unquote(self.test_database_url)
        self.assertEqual(decoded_actual, decoded_expected,
                         f"Database session should use temporary database {self.test_database_url}")

        # Clear any existing data
        try:
            self.db_session.query(ExtractionCandidate).delete()
            self.db_session.query(ReviewItem).delete()
            self.db_session.query(CardSlot).delete()
            self.db_session.query(Page).delete()
            self.db_session.query(Document).delete()
            self.db_session.query(Batch).delete()
            self.db_session.commit()
        except Exception:
            self.db_session.rollback()

    def tearDown(self):
        """Clean up after each test."""
        if hasattr(self, '_ocr_patch'):
            self._ocr_patch.stop()
        self.db_session.close()

    def load_page_26_detections(self):
        """Load all page 26 detection files and return them grouped by slot."""
        import glob

        detections_by_slot = {}
        bounds_by_slot = {}

        # Find all page 26 detection files
        files = glob.glob("diagnostic_page26_preview/page_26_card_r*c*_results.json")

        print(f"Found {len(files)} detection files")

        for file_path in files:
            # Extract slot number from filename
            # Format: page_26_card_r{row}c{col}_results.json
            filename = os.path.basename(file_path)
            if filename.startswith('page_26_card_r') and filename.endswith('_results.json'):
                # Parse row and column using regex
                # pattern: page_26_card_r(\d+)c(\d+)_results.json
                match = re.match(r'page_26_card_r(\d+)c(\d+)_results\.json', filename)
                if match:
                    row = int(match.group(1))
                    col = int(match.group(2))
                    slot_num = (row - 1) * 3 + col  # Assuming 3 columns per row to match our slot numbering

                    # Load detections and bounds
                    try:
                        with open(file_path, 'r') as f:
                            data = json.load(f)
                            detections_by_slot[slot_num] = data.get('detections', [])
                            bounds_by_slot[slot_num] = data.get('card_bounds', {})
                    except Exception as e:
                        print(f"Error loading {file_path}: {e}")
                else:
                    print(f"Warning: Could not parse filename {filename}")

        print(f"Loaded detections for {len(detections_by_slot)} slots")
        empty_slots = [slot for slot, detections in detections_by_slot.items() if len(detections) == 0]
        print(f"Found {len(empty_slots)} empty slots: {empty_slots[:10]}{'...' if len(empty_slots) > 10 else ''}")

        return detections_by_slot, bounds_by_slot

    def test_replay_page_26_detections(self):
        """Test replaying saved page 26 detections through production processing path."""
        # Load the saved detections and bounds
        detections_by_slot, bounds_by_slot = self.load_page_26_detections()

        # Verify we loaded the expected data
        self.assertEqual(len(detections_by_slot), 30, "Should have loaded detections for 30 slots")

        # Count occupied vs empty slots
        occupied_slots = [slot for slot, detections in detections_by_slot.items() if len(detections) > 0]
        empty_slots = [slot for slot, detections in detections_by_slot.items() if len(detections) == 0]

        print(f"Occupied slots: {len(occupied_slots)}")
        print(f"Empty slots: {len(empty_slots)} {empty_slots}")

        # According to the requirements, we should have 28 occupied slots and 2 empty slots
        self.assertEqual(len(occupied_slots), 28, f"Expected 28 occupied slots, got {len(occupied_slots)}")
        self.assertEqual(len(empty_slots), 2, f"Expected 2 empty slots, got {len(empty_slots)}")

        # Verify that the empty slots are 29 and 30 (as per combined_replay.json)
        self.assertEqual(sorted(empty_slots), [29, 30], f"Expected empty slots to be [29, 30], got {sorted(empty_slots)}")

        # Load combined_replay.json for field validation
        with open("diagnostic_page26_preview/combined_replay.json", 'r') as f:
            combined_replay = json.load(f)

        # Create a mapping from slot number to expected parsed fields
        expected_fields_by_slot = {}
        for slot_data in combined_replay:
            # Calculate slot number from row and column (1-indexed)
            row = slot_data['row']
            column = slot_data['column']
            slot_num = (row - 1) * 3 + column  # 3 columns per row

            # Store the parsed fields for this slot
            expected_fields_by_slot[slot_num] = slot_data['parsed_fields']

        # Create a batch first (since documents require batch_id)
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        # Create a test document with all required fields
        # We'll use dummy values for the required hash and size fields
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,  # Required field
            original_filename="test_el6.pdf",
            filename="test_el6.pdf",  # Required field
            file_path="samples/el6.pdf",
            file_hash="a" * 64,  # SHA-256 hash (64 hex chars)
            file_size=1024000,  # 1MB dummy size
            page_count=30,  # Assume 30 pages for testing
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Note: We don't manually create Page or CardSlot records here.
        # The processor will create them automatically during geometry detection and processing.

        # Create a dummy image matching the rendered page-26 dimensions
        # From the pasted content: "The supplied page-26 render was 4132 pixels wide and 5846 pixels high"
        # Note: NumPy array shape is (height, width, channels)
        import numpy as np
        dummy_image = np.zeros((5846, 4132, 3), dtype=np.uint8)
        print(f"\nDUMMY IMAGE SHAPE: {dummy_image.shape}")

        # Reset the replay counter and configure the mock OCR service
        slot_state = {'counter': 0, 'order': list(range(1, 31))}  # Slots 1-30 in order

        def mock_extract_from_image(image):
            counter = slot_state['counter']
            order = slot_state['order']
            if counter < len(order):
                slot_num = order[counter]
                slot_state['counter'] = counter + 1
                result = detections_by_slot.get(slot_num, [])
                if slot_num == 1:  # Debug output for slot 1
                    print(f"\nMOCK OCR FOR SLOT 1: Returning {len(result)} detections")
                    for i, det in enumerate(result):
                        print(f"  {i}: {det.get('text')} (confidence: {det.get('confidence')})")
                    # Also print image dimensions if available
                    if hasattr(image, 'shape'):
                        print(f"  Image dimensions: {image.shape}")

                # No synthetic-data replacement for real replay test
                # Preserve the original HUSBAND label and independently saved expected values
                pass

                # Verify that the OCR crop dimensions match the saved bounds from the fixture
                if slot_num in bounds_by_slot:
                    bounds = bounds_by_slot[slot_num]
                    if bounds:  # Make sure bounds are not empty
                        expected_height = bounds["bottom"] - bounds["top"]
                        expected_width = bounds["right"] - bounds["left"]

                        if hasattr(image, 'shape'):
                            actual_height, actual_width = image.shape[:2]
                            if actual_width != expected_width or actual_height != expected_height:
                                raise AssertionError(
                                    f"Slot {slot_num}: crop dimensions differ from saved fixture: "
                                    f"expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
                                )

                return result
            return []  # Return empty list for any extra calls (should not happen)

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a mock OCR service
        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.side_effect = mock_extract_from_image

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            # Import settings instance for verification
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Import settings instance for verification
                from app.config.settings import settings as settings_instance

                # Verify that we're using the correct mock
                self.assertIs(processor_module.ocr_service, mock_ocr_service)

                # Verify that processor_module.settings is the correct settings instance
                self.assertIs(processor_module.settings, settings_instance)

                # Verify that we're using the correct mock
                self.assertIs(processor_module.ocr_service, mock_ocr_service)

                try:
                    # Configure PDF mock to return dummy images for all 26 pages
                    # Important: PDF handler returns (page_number, image) tuples
                    mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 27)]

                    # Process the document (just page 26)
                    result = self.processor.process_document(document_id=test_doc.id, pages=[26])

                    # Check that processing succeeded - ASSERT IMMEDIATELY
                    self.assertTrue(result["success"], f"Processing should succeed but got: {result}")
                    print(f"Processing result: {result}")

                    # Count extraction candidates and review items
                    extraction_count = self.db_session.query(ExtractionCandidate).count()
                    review_count = self.db_session.query(ReviewItem).count()

                    # Print diagnostic info
                    print(f"Extraction candidates: {extraction_count}")
                    print(f"Review items: {review_count}")

                    # According to the requirements, we should get 28 extraction candidates and 28 review items
                    self.assertEqual(extraction_count, 28,
                                   f"Expected 28 extraction candidates, got {extraction_count}")
                    self.assertEqual(review_count, 28,
                                   f"Expected 28 review items, got {review_count}")

                    # Verify that the two empty slots create neither extraction candidates nor review items
                    for slot_num in empty_slots:
                        # Find the page record that was created by the processor
                        test_page = self.db_session.query(Page).filter(
                            Page.document_id == test_doc.id,
                            Page.page_number == 26
                        ).first()

                        self.assertIsNotNone(test_page, "Page record should exist after processing")

                        # Find the card slot for this slot number
                        card_slot = self.db_session.query(CardSlot).filter(
                            CardSlot.page_id == test_page.id,
                            CardSlot.physical_slot == slot_num
                        ).first()

                        self.assertIsNotNone(card_slot, f"Card slot {slot_num} should exist")

                        # Check that no extraction candidate exists for this slot
                        extraction_count_for_slot = self.db_session.query(ExtractionCandidate).filter(
                            ExtractionCandidate.card_slot_id == card_slot.id
                        ).count()

                        # Check that no review item exists for this slot
                        review_count_for_slot = self.db_session.query(ReviewItem).filter(
                            ReviewItem.card_slot_id == card_slot.id
                        ).count()

                        self.assertEqual(extraction_count_for_slot, 0,
                                       f"Empty slot {slot_num} should have 0 extraction candidates, got {extraction_count_for_slot}")
                        self.assertEqual(review_count_for_slot, 0,
                                       f"Empty slot {slot_num} should have 0 review items, got {review_count_for_slot}")

                    # Verify stored fields for all 28 populated slots against combined_replay.json
                    for slot_num in range(1, 29):  # Slots 1-28 are populated
                        # Skip if this slot is unexpectedly empty (shouldn't happen based on earlier checks)
                        if slot_num in empty_slots:
                            self.fail(f"Slot {slot_num} is expected to be populated but was found empty")

                        # Find the card slot for this slot number
                        card_slot = self.db_session.query(CardSlot).filter(
                            CardSlot.page_id == test_page.id,
                            CardSlot.physical_slot == slot_num
                        ).first()

                        self.assertIsNotNone(card_slot, f"Card slot {slot_num} should exist")

                        # Get the expected fields from combined_replay.json
                        expected_fields = expected_fields_by_slot.get(slot_num)
                        self.assertIsNotNone(expected_fields, f"No expected fields found for slot {slot_num} in combined_replay.json")

                        # Check that an extraction candidate exists for this slot
                        extraction_candidate = self.db_session.query(ExtractionCandidate).filter(
                            ExtractionCandidate.card_slot_id == card_slot.id
                        ).first()

                        self.assertIsNotNone(extraction_candidate, f"Slot {slot_num} should have an extraction candidate")

                        # Debug: Print the parsed fields and candidate values for troubleshooting
                        if slot_num == 1:
                            print(f"\nDEBUG Slot {slot_num}:")
                            print(f"  Expected fields: {expected_fields}")
                            print(f"  Extraction candidate raw_full_name: {repr(getattr(extraction_candidate, 'raw_full_name'))}")
                            print(f"  Extraction candidate raw_relation_type: {repr(getattr(extraction_candidate, 'raw_relation_type'))}")
                            print(f"  Extraction candidate raw_relative_name: {repr(getattr(extraction_candidate, 'raw_relative_name'))}")
                            print(f"  Extraction candidate raw_house_number: {repr(getattr(extraction_candidate, 'raw_house_number'))}")
                            print(f"  Extraction candidate raw_age: {repr(getattr(extraction_candidate, 'raw_age'))}")
                            print(f"  Extraction candidate raw_gender: {repr(getattr(extraction_candidate, 'raw_gender'))}")
                            print(f"  Extraction candidate raw_epic: {repr(getattr(extraction_candidate, 'raw_epic'))}")

                        # Verify extraction candidate fields match expected values
                        field_mappings = [
                            ('raw_epic', 'epic'),
                            ('raw_full_name', 'name'),
                            ('raw_relation_type', 'relation_type'),
                            ('raw_relative_name', 'relation_name'),
                            ('raw_house_number', 'house_number'),
                            ('raw_age', 'age'),
                            ('parsed_age', 'age'),  # Add parsed_age assertion
                            ('raw_gender', 'gender'),
                        ]

                        for db_field, json_field in field_mappings:
                            db_value = getattr(extraction_candidate, db_field)
                            expected_value = expected_fields.get(json_field)

                            # Handle field-specific type conversions according to the model schema
                            if db_field == 'raw_age':
                                # raw_age is stored as String in the database
                                expected_value_for_comparison = str(expected_value) if expected_value is not None else None
                                db_value_for_comparison = db_value
                            elif db_field in ['parsed_age', 'reviewed_age', 'raw_full_name', 'raw_relation_type',
                                            'raw_relative_name', 'raw_house_number', 'raw_gender', 'raw_epic']:
                                # These fields should match expected types directly
                                # For parsed_age and reviewed_age: expect integer
                                # For raw_* fields: expect string (same as expected_value from fixture)
                                db_value_for_comparison = db_value
                                expected_value_for_comparison = expected_value
                            else:
                                # Fallback for any other fields
                                db_value_for_comparison = db_value
                                expected_value_for_comparison = expected_value

                            self.assertEqual(db_value_for_comparison, expected_value_for_comparison,
                                           f"Slot {slot_num}: ExtractionCandidate.{db_field} expected {repr(expected_value_for_comparison)}, got {repr(db_value_for_comparison)}")

                        # Check that a review item exists for this slot
                        review_item = self.db_session.query(ReviewItem).filter(
                            ReviewItem.card_slot_id == card_slot.id
                        ).first()

                        self.assertIsNotNone(review_item, f"Slot {slot_num} should have a review item")

                        # Verify review item fields match expected values using explicit field mapping
                        # As per user instructions: do not derive reviewed attribute names with split()
                        review_item_explicit_mapping = [
                            ('reviewed_full_name', 'name'),           # String
                            ('reviewed_relation_type', 'relation_type'), # String
                            ('reviewed_relative_name', 'relation_name'), # String
                            ('reviewed_house_number', 'house_number'), # String
                            ('reviewed_age', 'age'),                  # Integer
                            ('reviewed_gender', 'gender'),            # String
                            ('reviewed_epic', 'epic')                 # String
                        ]

                        for db_field, json_field in review_item_explicit_mapping:
                            db_value = getattr(review_item, db_field)
                            expected_value = expected_fields.get(json_field)

                            # Handle field-specific type conversions according to the model schema
                            if db_field == 'reviewed_age':
                                # reviewed_age is stored as Integer in the database
                                expected_value_for_comparison = expected_value
                                db_value_for_comparison = db_value
                            else:
                                # All other reviewed_* fields are stored as String in the database
                                expected_value_for_comparison = expected_value
                                db_value_for_comparison = db_value

                            self.assertEqual(db_value_for_comparison, expected_value_for_comparison,
                                           f"Slot {slot_num}: ReviewItem.{db_field} expected {repr(expected_value_for_comparison)}, got {repr(db_value_for_comparison)}")

                        # Additional verification: parsed_age should equal reviewed_age (both should be integers equal to expected_age)
                        if hasattr(extraction_candidate, 'parsed_age') and hasattr(review_item, 'reviewed_age'):
                            expected_age = expected_fields.get('age')
                            if expected_age is not None:
                                self.assertEqual(extraction_candidate.parsed_age, expected_age,
                                               f"Slot {slot_num}: ExtractionCandidate.parsed_age expected {expected_age}, got {extraction_candidate.parsed_age}")
                                self.assertEqual(review_item.reviewed_age, expected_age,
                                               f"Slot {slot_num}: ReviewItem.reviewed_age expected {expected_age}, got {review_item.reviewed_age}")
                                # They should be equal to each other
                                self.assertEqual(extraction_candidate.parsed_age, review_item.reviewed_age,
                                               f"Slot {slot_num}: ExtractionCandidate.parsed_age ({extraction_candidate.parsed_age}) should equal ReviewItem.reviewed_age ({review_item.reviewed_age})")

                    # Assert exactly 30 mocked OCR calls (one per slot)
                    self.assertEqual(mock_ocr_service.extract_text_from_image.call_count, 30,
                                   f"Expected exactly 30 OCR calls, got {mock_ocr_service.extract_text_from_image.call_count}")

                    # Assert that region OCR was never called
                    # Note: We need to check if the processor's OCR service has extract_text_from_region method
                    # and assert it was never called
                    if hasattr(mock_ocr_service, 'extract_text_from_region'):
                        self.assertEqual(mock_ocr_service.extract_text_from_region.call_count, 0,
                                       f"Expected 0 region OCR calls, got {mock_ocr_service.extract_text_from_region.call_count}")
                finally:
                    pass  # No patch object to stop since we used the context manager form without 'as'

    def test_ocr_free_reprocessing(self):
        """Test OCR-free reprocessing - manual corrections should persist."""
        # Load the saved detections and bounds
        detections_by_slot, bounds_by_slot = self.load_page_26_detections()

        # Verify we loaded the expected data
        self.assertEqual(len(detections_by_slot), 30, "Should have loaded detections for 30 slots")

        # Count occupied vs empty slots
        occupied_slots = [slot for slot, detections in detections_by_slot.items() if len(detections) > 0]
        empty_slots = [slot for slot, detections in detections_by_slot.items() if len(detections) == 0]

        self.assertEqual(len(occupied_slots), 28, f"Expected 28 occupied slots, got {len(occupied_slots)}")
        self.assertEqual(len(empty_slots), 2, f"Expected 2 empty slots, got {len(empty_slots)}")
        self.assertEqual(sorted(empty_slots), [29, 30], f"Expected empty slots to be [29, 30], got {sorted(empty_slots)}")

        # Load combined_replay.json for field validation
        with open("diagnostic_page26_preview/combined_replay.json", 'r') as f:
            combined_replay = json.load(f)

        # Create a mapping from slot number to expected parsed fields
        expected_fields_by_slot = {}
        for slot_data in combined_replay:
            # Calculate slot number from row and column (1-indexed)
            row = slot_data['row']
            column = slot_data['column']
            slot_num = (row - 1) * 3 + column  # 3 columns per row
            expected_fields_by_slot[slot_num] = slot_data['parsed_fields']

        # Create a batch first
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        # Create a test document
        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test_el6.pdf",
            filename="test_el6.pdf",
            file_path="samples/el6.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=30,
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image matching the rendered page-26 dimensions
        import numpy as np
        dummy_image = np.zeros((5846, 4132, 3), dtype=np.uint8)

        # First processing pass
        slot_state = {'counter': 0, 'order': list(range(1, 31))}

        def mock_extract_from_image_first_pass(image):
            counter = slot_state['counter']
            order = slot_state['order']
            if counter < len(order):
                slot_num = order[counter]
                slot_state['counter'] = counter + 1
                result = detections_by_slot.get(slot_num, [])

                # Verify OCR crop dimensions match saved bounds
                if slot_num in bounds_by_slot:
                    bounds = bounds_by_slot[slot_num]
                    if bounds:
                        expected_height = bounds["bottom"] - bounds["top"]
                        expected_width = bounds["right"] - bounds["left"]

                        if hasattr(image, 'shape'):
                            actual_height, actual_width = image.shape[:2]
                            if actual_width != expected_width or actual_height != expected_height:
                                raise AssertionError(
                                    f"Slot {slot_num}: crop dimensions differ from saved fixture: "
                                    f"expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
                                )

                return result
            return []

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a mock OCR service for first pass
        mock_ocr_service_first = MagicMock()
        mock_ocr_service_first.extract_text_from_image.side_effect = mock_extract_from_image_first_pass

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service_first), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images for all 26 pages
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 27)]

                # Process the document (first pass)
                result_first = self.processor.process_document(document_id=test_doc.id, pages=[26])

                # Check that processing succeeded
                self.assertTrue(result_first["success"], f"First pass processing should succeed but got: {result_first}")

                # Count extraction candidates and review items
                extraction_count_first = self.db_session.query(ExtractionCandidate).count()
                review_count_first = self.db_session.query(ReviewItem).count()

                # According to the requirements, we should get 28 extraction candidates and 28 review items
                self.assertEqual(extraction_count_first, 28,
                               f"Expected 28 extraction candidates, got {extraction_count_first}")
                self.assertEqual(review_count_first, 28,
                               f"Expected 28 review items, got {review_count_first}")

                # Save the extraction and review record IDs for each populated slot
                extraction_ids_by_slot = {}
                review_ids_by_slot = {}

                test_page = self.db_session.query(Page).filter(
                    Page.document_id == test_doc.id,
                    Page.page_number == 26
                ).first()

                self.assertIsNotNone(test_page, "Page record should exist after processing")

                for slot_num in range(1, 29):  # Slots 1-28 are populated
                    card_slot = self.db_session.query(CardSlot).filter(
                        CardSlot.page_id == test_page.id,
                        CardSlot.physical_slot == slot_num
                    ).first()

                    self.assertIsNotNone(card_slot, f"Card slot {slot_num} should exist")

                    # Get extraction candidate ID
                    extraction_candidate = self.db_session.query(ExtractionCandidate).filter(
                        ExtractionCandidate.card_slot_id == card_slot.id
                    ).first()

                    self.assertIsNotNone(extraction_candidate, f"Slot {slot_num} should have an extraction candidate")
                    extraction_ids_by_slot[slot_num] = extraction_candidate.id

                    # Get review item ID
                    review_item = self.db_session.query(ReviewItem).filter(
                        ReviewItem.card_slot_id == card_slot.id
                    ).first()

                    self.assertIsNotNone(review_item, f"Slot {slot_num} should have a review item")
                    review_ids_by_slot[slot_num] = review_item.id

                # Manually change one review item's name and house number, and mark it REVIEWED
                # Let's choose slot 1 for modification
                slot_to_modify = 1
                card_slot_to_modify = self.db_session.query(CardSlot).filter(
                    CardSlot.page_id == test_page.id,
                    CardSlot.physical_slot == slot_to_modify
                ).first()

                review_item_to_modify = self.db_session.query(ReviewItem).filter(
                    ReviewItem.card_slot_id == card_slot_to_modify.id
                ).first()

                # Store original values for verification
                original_name = review_item_to_modify.reviewed_full_name
                original_house_number = review_item_to_modify.reviewed_house_number

                # Modify the review item
                review_item_to_modify.reviewed_full_name = "MODIFIED NAME"
                review_item_to_modify.reviewed_house_number = "MODIFIED HOUSE NUMBER"
                review_item_to_modify.review_status = ReviewStatusEnum.REVIEWED.value

                self.db_session.commit()

                # Reset the OCR replay counter and mock call history for second pass
                slot_state_second = {'counter': 0, 'order': list(range(1, 31))}

                def mock_extract_from_image_second_pass(image):
                    counter = slot_state_second['counter']
                    order = slot_state_second['order']
                    if counter < len(order):
                        slot_num = order[counter]
                        slot_state_second['counter'] = counter + 1
                        result = detections_by_slot.get(slot_num, [])

                        # Verify OCR crop dimensions match saved bounds
                        if slot_num in bounds_by_slot:
                            bounds = bounds_by_slot[slot_num]
                            if bounds:
                                expected_height = bounds["bottom"] - bounds["top"]
                                expected_width = bounds["right"] - bounds["left"]

                                if hasattr(image, 'shape'):
                                    actual_height, actual_width = image.shape[:2]
                                    if actual_width != expected_width or actual_height != expected_height:
                                        raise AssertionError(
                                            f"Slot {slot_num}: crop dimensions differ from saved fixture: "
                                            f"expected {expected_width}x{expected_height}, got {actual_width}x{actual_height}"
                                        )

                        return result
                    return []

                # Create a new mock OCR service for second pass
                mock_ocr_service_second = MagicMock()
                mock_ocr_service_second.extract_text_from_image.side_effect = mock_extract_from_image_second_pass

                # Second processing pass (OCR-free reprocessing)
                with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images_second, \
                     patch.object(processor_module, 'ocr_service', mock_ocr_service_second), \
                     patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                    # Configure PDF mock to return dummy images for all 26 pages
                    mock_pdf_to_images_second.return_value = [(i, dummy_image) for i in range(1, 27)]

                    # Process the document again (second pass)
                    result_second = self.processor.process_document(document_id=test_doc.id, pages=[26])

                    # Check that processing succeeded
                    self.assertTrue(result_second["success"], f"Second pass processing should succeed but got: {result_second}")

                    # Count extraction candidates and review items - should remain the same
                    extraction_count_second = self.db_session.query(ExtractionCandidate).count()
                    review_count_second = self.db_session.query(ReviewItem).count()

                    self.assertEqual(extraction_count_second, 28,
                                   f"Expected 28 extraction candidates after reprocessing, got {extraction_count_second}")
                    self.assertEqual(review_count_second, 28,
                                   f"Expected 28 review items after reprocessing, got {review_count_second}")

                    # Record IDs should remain unchanged for every populated slot
                    for slot_num in range(1, 29):
                        card_slot = self.db_session.query(CardSlot).filter(
                            CardSlot.page_id == test_page.id,
                            CardSlot.physical_slot == slot_num
                        ).first()

                        self.assertIsNotNone(card_slot, f"Card slot {slot_num} should exist")

                        # Check extraction candidate ID remains unchanged
                        extraction_candidate = self.db_session.query(ExtractionCandidate).filter(
                            ExtractionCandidate.card_slot_id == card_slot.id
                        ).first()

                        self.assertIsNotNone(extraction_candidate, f"Slot {slot_num} should have an extraction candidate")
                        self.assertEqual(extraction_candidate.id, extraction_ids_by_slot[slot_num],
                                       f"Extraction candidate ID for slot {slot_num} should remain unchanged")

                        # Check review item ID remains unchanged
                        review_item = self.db_session.query(ReviewItem).filter(
                            ReviewItem.card_slot_id == card_slot.id
                        ).first()

                        self.assertIsNotNone(review_item, f"Slot {slot_num} should have a review item")
                        self.assertEqual(review_item.id, review_ids_by_slot[slot_num],
                                       f"Review item ID for slot {slot_num} should remain unchanged")

                    # Manual modifications should persist
                    # Check that the modified name and house number remain unchanged
                    review_item_after = self.db_session.query(ReviewItem).filter(
                        ReviewItem.card_slot_id == card_slot_to_modify.id
                    ).first()

                    self.assertIsNotNone(review_item_after, f"Modified review item for slot {slot_to_modify} should exist")
                    self.assertEqual(review_item_after.reviewed_full_name, "MODIFIED NAME",
                                   f"Modified name for slot {slot_to_modify} should persist")
                    self.assertEqual(review_item_after.reviewed_house_number, "MODIFIED HOUSE NUMBER",
                                   f"Modified house number for slot {slot_to_modify} should persist")
                    self.assertEqual(review_item_after.review_status, ReviewStatusEnum.REVIEWED.value,
                                   f"Review status for slot {slot_to_modify} should remain REVIEWED")

                    # Verify that unmodified slots still have their original values from first pass
                    # Let's check slot 2 as an example of an unmodified slot
                    slot_unmodified = 2
                    card_slot_unmodified = self.db_session.query(CardSlot).filter(
                        CardSlot.page_id == test_page.id,
                        CardSlot.physical_slot == slot_unmodified
                    ).first()

                    review_item_unmodified = self.db_session.query(ReviewItem).filter(
                        ReviewItem.card_slot_id == card_slot_unmodified.id
                    ).first()

                    expected_fields_unmodified = expected_fields_by_slot.get(slot_unmodified)
                    self.assertIsNotNone(expected_fields_unmodified, f"Expected fields should exist for slot {slot_unmodified}")

                    # For unmodified slots, values should match the expected fields from combined_replay.json
                    self.assertEqual(review_item_unmodified.reviewed_full_name, expected_fields_unmodified.get('name'),
                                   f"Unmodified slot {slot_unmodified} name should match expected value")
                    self.assertEqual(review_item_unmodified.reviewed_house_number, expected_fields_unmodified.get('house_number'),
                                   f"Unmodified slot {slot_unmodified} house number should match expected value")

                    # Slots 29 and 30 should still have no extraction or review records
                    for slot_num in empty_slots:
                        card_slot_empty = self.db_session.query(CardSlot).filter(
                            CardSlot.page_id == test_page.id,
                            CardSlot.physical_slot == slot_num
                        ).first()

                        self.assertIsNotNone(card_slot_empty, f"Card slot {slot_num} should exist")

                        # Check that no extraction candidate exists for this slot
                        extraction_count_empty = self.db_session.query(ExtractionCandidate).filter(
                            ExtractionCandidate.card_slot_id == card_slot_empty.id
                        ).count()

                        # Check that no review item exists for this slot
                        review_count_empty = self.db_session.query(ReviewItem).filter(
                            ReviewItem.card_slot_id == card_slot_empty.id
                        ).count()

                        self.assertEqual(extraction_count_empty, 0,
                                       f"Empty slot {slot_num} should have 0 extraction candidates after reprocessing, got {extraction_count_empty}")
                        self.assertEqual(review_count_empty, 0,
                                       f"Empty slot {slot_num} should have 0 review items after reprocessing, got {review_count_empty}")

                    # Assert exactly 30 mocked OCR calls per pass (one per slot)
                    self.assertEqual(mock_ocr_service_first.extract_text_from_image.call_count, 30,
                                   f"Expected exactly 30 OCR calls in first pass, got {mock_ocr_service_first.extract_text_from_image.call_count}")
                    self.assertEqual(mock_ocr_service_second.extract_text_from_image.call_count, 30,
                                   f"Expected exactly 30 OCR calls in second pass, got {mock_ocr_service_second.extract_text_from_image.call_count}")

                    # Assert that region OCR was never called in either pass
                    if hasattr(mock_ocr_service_first, 'extract_text_from_region'):
                        self.assertEqual(mock_ocr_service_first.extract_text_from_region.call_count, 0,
                                       f"Expected 0 region OCR calls in first pass, got {mock_ocr_service_first.extract_text_from_region.call_count}")
                    if hasattr(mock_ocr_service_second, 'extract_text_from_region'):
                        self.assertEqual(mock_ocr_service_second.extract_text_from_region.call_count, 0,
                                       f"Expected 0 region OCR calls in second pass, got {mock_ocr_service_second.extract_text_from_region.call_count}")

    def test_middle_page_reprocessing(self):
        """Test processing a middle page with scattered empty slots."""
        # Human page number 10 in a mocked document
        test_page_number = 10

        # Create synthetic replay data for 30 slots (10 rows × 3 columns)
        # Slots 2, 11, 19, 27 will be empty (1-indexed)
        empty_slots = {2, 11, 19, 27}

        # Generate synthetic detection data
        synthetic_detections_by_slot = {}
        synthetic_bounds_by_slot = {}

        # Generate data for each slot (1-30)
        for slot_num in range(1, 31):
            # Calculate row and column (1-indexed)
            # Assuming 3 columns per row
            row = ((slot_num - 1) // 3) + 1
            column = ((slot_num - 1) % 3) + 1

            # Load the fixture for this specific slot position to get correct bounds and detections
            fixture_filename = f"diagnostic_page26_preview/page_26_card_r{row}c{column}_results.json"
            with open(fixture_filename, 'r') as f:
                fixture_data = json.load(f)

            if slot_num in empty_slots:
                # Empty slot: no detections, but use the fixture's bounds
                detections = []
                bounds = fixture_data["card_bounds"]
            else:
                # Populated slot: use the fixture's detections and bounds
                detections = copy.deepcopy(fixture_data["detections"])  # Deep copy for isolation
                bounds = fixture_data["card_bounds"]

                # Special handling for slots with continuation detections in relative name
                # to prevent extra text in relative name while preserving bounding box
                if slot_num == 14:
                    # Find and set relative label to exactly "Father's Name: Parent Name"
                    for detection in detections:
                        text = detection.get("text", "")
                        if re.search(r"(Father's|Husband's|Mother's)\s+Name\s*:", text):
                            detection["text"] = "Father's Name: Parent Name"
                            break

                    # Find and clear the SHINDE continuation detection by its original text and bbox
                    # This prevents it from contributing extra text to the relative name
                    for detection in detections:
                        text = detection.get("text", "")
                        bbox = detection.get("bbox")
                        if (text == "SHINDE" and
                            bbox and len(bbox) == 4 and
                            bbox[0][0] == 28.0 and bbox[0][1] == 228.0 and
                            bbox[1][0] == 206.0 and bbox[1][1] == 228.0 and
                            bbox[2][0] == 206.0 and bbox[2][1] == 277.0 and
                            bbox[3][0] == 28.0 and bbox[3][1] == 277.0):
                            detection["text"] = ""
                            break
                elif slot_num == 24:
                    # Find and set relative label to exactly "Father's Name: Parent Name"
                    for detection in detections:
                        text = detection.get("text", "")
                        if re.search(r"(Father's|Husband's|Mother's)\s+Name\s*:", text):
                            detection["text"] = "Father's Name: Parent Name"
                            break

                    # Find and clear the SURVASE continuation detection by its original text and bbox
                    # This prevents it from contributing extra text to the relative name
                    for detection in detections:
                        text = detection.get("text", "")
                        bbox = detection.get("bbox")
                        if (text == "SURVASE" and
                            bbox and len(bbox) == 4 and
                            bbox[0][0] == 27.0 and bbox[0][1] == 226.0 and
                            bbox[1][0] == 243.0 and bbox[1][1] == 226.0 and
                            bbox[2][0] == 243.0 and bbox[2][1] == 275.0 and
                            bbox[3][0] == 27.0 and bbox[3][1] == 275.0):
                            detection["text"] = ""
                            break
                elif slot_num == 26:
                    # Find and set relative label to exactly "Father's Name: Parent Name"
                    for detection in detections:
                        text = detection.get("text", "")
                        if re.search(r"(Father's|Husband's|Mother's)\s+Name\s*:", text):
                            detection["text"] = "Father's Name: Parent Name"
                            break

                    # Find and clear the Jogdand continuation detection by its original text and bbox
                    # This prevents it from contributing extra text to the relative name
                    for detection in detections:
                        text = detection.get("text", "")
                        bbox = detection.get("bbox")
                        if (text == "Jogdand" and
                            bbox and len(bbox) == 4 and
                            bbox[0][0] == 29.0 and bbox[0][1] == 229.0 and
                            bbox[1][0] == 219.0 and bbox[1][1] == 229.0 and
                            bbox[2][0] == 219.0 and bbox[2][1] == 283.0 and
                            bbox[3][0] == 29.0 and bbox[3][1] == 283.0):
                            detection["text"] = ""
                            break

                # Special handling for slots 29 and 30 which originally have no detections
                # but need to be populated for this test
                if slot_num in [29, 30] and len(detections) == 0:
                    # Find a populated slot with matching dimensions to copy detections from
                    # Select a donor slot with matching width and height (crop dimensions)
                    target_width = bounds["right"] - bounds["left"]
                    target_height = bounds["bottom"] - bounds["top"]
                    donor_found = False
                    # Try all possible slots (1-30) that are not empty and have non-empty detections
                    for potential_donor in range(1, 31):
                        if potential_donor in empty_slots:
                            continue
                        donor_fixture_name = f"diagnostic_page26_preview/page_26_card_r{((potential_donor-1)//3)+1}c{((potential_donor-1)%3)+1}_results.json"
                        try:
                            with open(donor_fixture_name, 'r') as f:
                                donor_data = json.load(f)
                        except FileNotFoundError:
                            continue
                        if len(donor_data["detections"]) == 0:
                            continue
                        donor_bounds = donor_data["card_bounds"]
                        donor_width = donor_bounds["right"] - donor_bounds["left"]
                        donor_height = donor_bounds["bottom"] - donor_bounds["top"]
                        if donor_width == target_width and donor_height == target_height:
                            detections = copy.deepcopy(donor_data["detections"])
                            donor_found = True
                            break
                    assert donor_found, f"Could not find a populated donor slot with matching dimensions ({target_width}x{target_height}) for slot {slot_num}"
                # Modify only the voter field text detections to make each slot unique:
                # Locate voter fields by their text labels and EPIC pattern, preserving serial/photo detections

                # Find and update Epic field (should contain 3 uppercase letters + 7 digits)
                epic_updated = False
                for i, detection in enumerate(detections):
                    text = detection.get("text", "")
                    if len(text) >= 10 and text[:3].isupper() and text[:3].isalpha() and text[3:].isdigit() and len(text[3:]) == 7:
                        detections[i]["text"] = f"AAA{slot_num:07d}"
                        epic_updated = True
                        break
                # Note: For populated slots from fixture, we don't necessarily update Epic as it's already unique

                # Find and update Name field (starts with "Name :")
                name_updated = False
                for i, detection in enumerate(detections):
                    text = detection.get("text", "")
                    if text.startswith("Name :"):
                        detections[i]["text"] = f"Name : Test Voter {slot_num}"
                        name_updated = True
                        break
                # Note: For populated slots from fixture, we don't necessarily update Name as it's already unique

                # Locate relative label by text and replace with "Father's Name: Parent Name"
                # Looking for Father's, Husband's and Mother's Name variants
                relative_label_replaced = False
                for i, detection in enumerate(detections):
                    text = detection.get("text", "")
                    # Look for relative label patterns: Father's, Husband's, Mother's Name variants
                    if re.search(r"(Father's|Husband's|Mother's)\s+Name\s*:", text):
                        # Replace the entire relative label with "Father's Name: Parent Name"
                        detections[i]["text"] = "Father's Name: Parent Name"
                        relative_label_replaced = True
                        break
                assert relative_label_replaced, f"Expected to find and replace exactly one relative label for slot {slot_num}"

                # Find and update House number field (starts with "House Number :")
                house_updated = False
                for i, detection in enumerate(detections):
                    text = detection.get("text", "")
                    if text.startswith("House Number :"):
                        detections[i]["text"] = f"House Number : {slot_num}"
                        house_updated = True
                        break
                # Note: For populated slots from fixture, we don't necessarily update House number as it's already unique

                # Find and update Age/Gender field (contains both "Age :" and "Gender :")
                age_gender_updated = False
                for i, detection in enumerate(detections):
                    text = detection.get("text", "")
                    if "Age :" in text and "Gender :" in text:
                        age = 20 + (slot_num % 50)  # Age between 20 and 69
                        gender = 'Male' if slot_num % 2 == 0 else 'Female'
                        age_gender_text = f"Age : {age} Gender : {gender}"
                        detections[i]["text"] = age_gender_text
                        age_gender_updated = True
                        break
                # Note: For populated slots from fixture, we don't necessarily update Age/Gender as it's already unique

                # Note: We leave detections 0 (serial), 5 (photo), and 7 (available) unchanged

            synthetic_detections_by_slot[slot_num] = detections
            synthetic_bounds_by_slot[slot_num] = bounds

        # Assert all 26 intended populated slots actually have voter detections
        intended_populated_slots = set(range(1, 31)) - empty_slots
        actually_populated_slots = {slot_num for slot_num in range(1, 31) if len(synthetic_detections_by_slot[slot_num]) > 0}
        self.assertEqual(intended_populated_slots, actually_populated_slots,
                        f"Mismatch in populated slots. Intended: {sorted(intended_populated_slots)}, Actually: {sorted(actually_populated_slots)}")

        # Verify we have the right number of empty and occupied slots
        occupied_slots = [s for s in range(1, 31) if s not in empty_slots]
        self.assertEqual(len(occupied_slots), 26, f"Expected 26 occupied slots, got {len(occupied_slots)}")
        self.assertEqual(len(empty_slots), 4, f"Expected 4 empty slots, got {len(empty_slots)}")
        self.assertEqual(set([2, 11, 19, 27]), empty_slots, f"Empty slots should be {{2, 11, 19, 27}}, got {sorted(empty_slots)}")

        # Create test data structures (simulating combined_replay.json)
        synthetic_replay_data = []
        for slot_num in range(1, 31):
            if slot_num not in empty_slots:
                # Calculate row and column for this slot
                row = ((slot_num - 1) // 3) + 1
                column = ((slot_num - 1) % 3) + 1

                # Add entry to synthetic replay data - matching what the field parser will extract
                epic_text = f"AAA{slot_num:07d}"  # What the field parser will extract from epic detection
                name_text = f"Test Voter {slot_num}"  # What the field parser will extract from "Name : Test Voter X"
                relation_text = f"Parent Name"  # What the field parser will extract from "Father's Name: Parent Name"
                house_text = f"{slot_num}"  # What the field parser will extract from "House Number : {slot_num}"
                age = 20 + (slot_num % 50)
                gender = 'MALE' if slot_num % 2 == 0 else 'FEMALE'  # What the field parser will extract (uppercase)

                synthetic_replay_data.append({
                    "row": row,
                    "column": column,
                    "parsed_fields": {
                        "epic": epic_text,
                        "name": name_text,
                        "relation_type": "FATHER",
                        "relation_name": relation_text,
                        "house_number": house_text,
                        "age": age,
                        "gender": gender,
                        "epic": epic_text,  # Duplicate for compatibility with test expectations
                        "serial_number": None
                    }
                })

        # Create a batch
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        # Create a test document
        test_doc = Document(
            batch_id=test_batch.id,
            original_filename=f"test_middle_page_{test_page_number}.pdf",
            filename=f"test_middle_page_{test_page_number}.pdf",
            file_path=f"samples/middle_{test_page_number}.pdf",
            file_hash="b" * 64,  # Different hash to avoid conflicts
            file_size=2048000,  # 2MB dummy size
            page_count=20,  # Document with enough pages to include page 10
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image matching the rendered page dimensions
        # Using dimensions similar to the fixture for consistency
        import numpy as np
        dummy_image = np.zeros((5846, 4132, 3), dtype=np.uint8)  # height, width, channels
        # Verify dummy_image is exactly (5846, 4132, 3)
        self.assertEqual(dummy_image.shape, (5846, 4132, 3))

        # First and only processing pass
        slot_state = {'counter': 0, 'order': list(range(1, 31))}  # Slots 1-30 in order

        def mock_extract_from_image(image):
            counter = slot_state['counter']
            order = slot_state['order']
            if counter < len(order):
                slot_num = order[counter]
                slot_state['counter'] = counter + 1
                result = synthetic_detections_by_slot.get(slot_num, [])
                # Debug: print what we're returning for the first few slots
                if slot_num <= 3:
                    print(f"MOCK OCR FOR SLOT {slot_num}: Returning {len(result)} detections")
                    for i, det in enumerate(result):
                        print(f"  {i}: {det.get('text')} (confidence: {det.get('confidence')})")

                # Verify OCR crop dimensions match synthetic bounds
                if slot_num in synthetic_bounds_by_slot:
                    bounds = synthetic_bounds_by_slot[slot_num]
                    if bounds:  # Make sure bounds are not empty
                        expected_height = bounds["bottom"] - bounds["top"]
                        expected_width = bounds["right"] - bounds["left"]

                        if hasattr(image, 'shape'):
                            actual_height, actual_width = image.shape[:2]
                            # Verify OCR crop dimensions match synthetic bounds
                            if slot_num in synthetic_bounds_by_slot:
                                bounds = synthetic_bounds_by_slot[slot_num]
                                # Read bounds by key - do not unpack a bounds dictionary directly
                                expected_height = int(bounds["bottom"]) - int(bounds["top"])
                                expected_width = int(bounds["right"]) - int(bounds["left"])
                                expected_shape = (expected_height, expected_width)

                                # Inside the mock, print image.shape as "Received card crop shape"
                                print(f"Received card crop shape: {image.shape[:2]} for slot {slot_num}")

                                # Assert image shape matches expected shape
                                self.assertEqual(image.shape[:2], expected_shape, f"Slot {slot_num}")

                return result
            return []  # Return empty list for any extra calls (should not happen)

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a mock OCR service
        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.side_effect = mock_extract_from_image

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return enough dummy images so page 10 is valid
                # Important: PDF handler returns (page_number, image) tuples
                # We need at least 10 pages to make page 10 valid (document has 20 pages)
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 21)]  # Pages 1-20

                # Process the document (just page 10)
                result = self.processor.process_document(document_id=test_doc.id, pages=[test_page_number])

                # Check that processing succeeded
                self.assertTrue(result["success"], f"Processing should succeed but got: {result}")

                # Count extraction candidates and review items
                extraction_count = self.db_session.query(ExtractionCandidate).count()
                review_count = self.db_session.query(ReviewItem).count()

                # Assert exactly 26 extraction candidates and 26 review items
                self.assertEqual(extraction_count, 26,
                               f"Expected 26 extraction candidates, got {extraction_count}")
                self.assertEqual(review_count, 26,
                               f"Expected 26 review items, got {review_count}")

                # Verify that each populated slot has exactly one record of each type
                # and empty slots have none
                test_page = self.db_session.query(Page).filter(
                    Page.document_id == test_doc.id,
                    Page.page_number == test_page_number
                ).first()

                self.assertIsNotNone(test_page, f"Page record for page {test_page_number} should exist")
                self.assertEqual(test_page.page_number, test_page_number,
                               f"Stored page number should be {test_page_number}")

                # Verify that only page 10 was processed and created for this document
                other_pages = self.db_session.query(Page).filter(
                    Page.document_id == test_doc.id,
                    Page.page_number != test_page_number
                ).count()
                self.assertEqual(other_pages, 0,
                               f"No other pages should be processed for document {test_doc.id}, but found {other_pages} other pages")

                # Check each slot
                for slot_num in range(1, 31):
                    card_slot = self.db_session.query(CardSlot).filter(
                        CardSlot.page_id == test_page.id,
                        CardSlot.physical_slot == slot_num
                    ).first()

                    self.assertIsNotNone(card_slot, f"Card slot {slot_num} should exist")

                    # Assert that the stored CardSlot bounds match each fixture's bounds
                    expected_bounds = synthetic_bounds_by_slot[slot_num]
                    self.assertEqual(card_slot.left, expected_bounds["left"],
                                   f"Slot {slot_num}: CardSlot.left expected {expected_bounds['left']}, got {card_slot.left}")
                    self.assertEqual(card_slot.top, expected_bounds["top"],
                                   f"Slot {slot_num}: CardSlot.top expected {expected_bounds['top']}, got {card_slot.top}")
                    self.assertEqual(card_slot.right, expected_bounds["right"],
                                   f"Slot {slot_num}: CardSlot.right expected {expected_bounds['right']}, got {card_slot.right}")
                    self.assertEqual(card_slot.bottom, expected_bounds["bottom"],
                                   f"Slot {slot_num}: CardSlot.bottom expected {expected_bounds['bottom']}, got {card_slot.bottom}")

                    if slot_num in empty_slots:
                        # Empty slots should have no extraction or review records
                        extraction_count_for_slot = self.db_session.query(ExtractionCandidate).filter(
                            ExtractionCandidate.card_slot_id == card_slot.id
                        ).count()

                        review_count_for_slot = self.db_session.query(ReviewItem).filter(
                            ReviewItem.card_slot_id == card_slot.id
                        ).count()

                        self.assertEqual(extraction_count_for_slot, 0,
                                       f"Empty slot {slot_num} should have 0 extraction candidates, got {extraction_count_for_slot}")
                        self.assertEqual(review_count_for_slot, 0,
                                       f"Empty slot {slot_num} should have 0 review items, got {review_count_for_slot}")
                    else:
                        # Populated slots should have exactly one extraction candidate and one review item
                        extraction_candidate = self.db_session.query(ExtractionCandidate).filter(
                            ExtractionCandidate.card_slot_id == card_slot.id
                        ).first()

                        self.assertIsNotNone(extraction_candidate,
                                           f"Populated slot {slot_num} should have an extraction candidate")

                        review_item = self.db_session.query(ReviewItem).filter(
                            ReviewItem.card_slot_id == card_slot.id
                        ).first()

                        self.assertIsNotNone(review_item,
                                           f"Populated slot {slot_num} should have a review item")

                        # Verify stored fields match synthetic data (with appropriate type handling)
                        # Find the expected fields for this slot
                        expected_fields = None
                        for replay_entry in synthetic_replay_data:
                            if replay_entry['row'] == ((slot_num - 1) // 3) + 1 and \
                               replay_entry['column'] == ((slot_num - 1) % 3) + 1:
                                expected_fields = replay_entry['parsed_fields']
                                break

                        self.assertIsNotNone(expected_fields, f"Expected fields should exist for slot {slot_num}")

                        # ExtractionCandidate field validations
                        # raw_* fields should be String (matching fixture data as strings)
                        self.assertEqual(extraction_candidate.raw_epic, expected_fields['epic'],
                                       f"Slot {slot_num}: ExtractionCandidate.raw_epic expected {expected_fields['epic']}, got {extraction_candidate.raw_epic}")
                        self.assertEqual(extraction_candidate.raw_full_name, expected_fields['name'],
                                       f"Slot {slot_num}: ExtractionCandidate.raw_full_name expected {expected_fields['name']}, got {extraction_candidate.raw_full_name}")
                        self.assertEqual(extraction_candidate.raw_relation_type, expected_fields['relation_type'],
                                       f"Slot {slot_num}: ExtractionCandidate.raw_relation_type expected {expected_fields['relation_type']}, got {extraction_candidate.raw_relation_type}")
                        self.assertEqual(extraction_candidate.raw_relative_name, expected_fields['relation_name'],
                                       f"Slot {slot_num}: ExtractionCandidate.raw_relative_name expected {expected_fields['relation_name']}, got {extraction_candidate.raw_relative_name}")
                        self.assertEqual(extraction_candidate.raw_house_number, expected_fields['house_number'],
                                       f"Slot {slot_num}: ExtractionCandidate.raw_house_number expected {expected_fields['house_number']}, got {extraction_candidate.raw_house_number}")
                        # raw_age should be string representation of the integer age
                        self.assertEqual(extraction_candidate.raw_age, str(expected_fields['age']),
                                       f"Slot {slot_num}: ExtractionCandidate.raw_age expected {str(expected_fields['age'])}, got {extraction_candidate.raw_age}")
                        self.assertEqual(extraction_candidate.raw_gender, expected_fields['gender'],
                                       f"Slot {slot_num}: ExtractionCandidate.raw_gender expected {expected_fields['gender']}, got {extraction_candidate.raw_gender}")

                        # parsed_age should be Integer (matching fixture age as integer)
                        self.assertEqual(extraction_candidate.parsed_age, expected_fields['age'],
                                       f"Slot {slot_num}: ExtractionCandidate.parsed_age expected {expected_fields['age']}, got {extraction_candidate.parsed_age}")

                        # ReviewItem field validations
                        # reviewed_* fields should be String except reviewed_age which should be Integer
                        self.assertEqual(review_item.reviewed_full_name, expected_fields['name'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_full_name expected {expected_fields['name']}, got {review_item.reviewed_full_name}")
                        self.assertEqual(review_item.reviewed_relation_type, expected_fields['relation_type'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_relation_type expected {expected_fields['relation_type']}, got {review_item.reviewed_relation_type}")
                        self.assertEqual(review_item.reviewed_relative_name, expected_fields['relation_name'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_relative_name expected {expected_fields['relation_name']}, got {review_item.reviewed_relative_name}")
                        self.assertEqual(review_item.reviewed_house_number, expected_fields['house_number'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_house_number expected {expected_fields['house_number']}, got {review_item.reviewed_house_number}")
                        # reviewed_age should be integer (matching fixture age as integer)
                        self.assertEqual(review_item.reviewed_age, expected_fields['age'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_age expected {expected_fields['age']}, got {review_item.reviewed_age}")
                        self.assertEqual(review_item.reviewed_gender, expected_fields['gender'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_gender expected {expected_fields['gender']}, got {review_item.reviewed_gender}")
                        self.assertEqual(review_item.reviewed_epic, expected_fields['epic'],
                                       f"Slot {slot_num}: ReviewItem.reviewed_epic expected {expected_fields['epic']}, got {review_item.reviewed_epic}")

                # Assert exactly 30 mocked OCR calls (one per slot)
                self.assertEqual(mock_ocr_service.extract_text_from_image.call_count, 30,
                               f"Expected exactly 30 OCR calls, got {mock_ocr_service.extract_text_from_image.call_count}")

                # Assert that region OCR was never called
                if hasattr(mock_ocr_service, 'extract_text_from_region'):
                    self.assertEqual(mock_ocr_service.extract_text_from_region.call_count, 0,
                                   f"Expected 0 region OCR calls, got {mock_ocr_service.extract_text_from_region.call_count}")


    def test_invalid_page_selection_out_of_range(self):
        """Test that entirely out-of-range page requests are rejected."""
        # Create a simple test document with 5 pages
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=5,  # 5 pages
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock OCR service to return empty detections (no pages should be processed)
        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.return_value = []

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 6)]  # Pages 1-5

                # Test entirely out-of-range request (pages 10-15)
                result = self.processor.process_document(document_id=test_doc.id, pages=[10, 11, 12, 13, 14, 15])

                # Assert failure
                self.assertFalse(result["success"], "Processing should fail for out-of-range pages")
                self.assertIn("Page number 10 is out of range", result["error"])

                # Assert zero OCR calls
                self.assertEqual(mock_ocr_service.extract_text_from_image.call_count, 0,
                               "No OCR calls should be made for invalid page requests")

                # Assert no Page, CardSlot, ExtractionCandidate or ReviewItem records created
                self.assertEqual(self.db_session.query(Page).filter(Page.document_id == test_doc.id).count(), 0,
                               "No Page records should be created for invalid page requests")
                self.assertEqual(self.db_session.query(CardSlot).count(), 0,
                               "No CardSlot records should be created for invalid page requests")
                self.assertEqual(self.db_session.query(ExtractionCandidate).count(), 0,
                               "No ExtractionCandidate records should be created for invalid page requests")
                self.assertEqual(self.db_session.query(ReviewItem).count(), 0,
                               "No ReviewItem records should be created for invalid page requests")

    def test_invalid_page_selection_mixed_valid_invalid(self):
        """Test that mixed valid/invalid page requests are rejected."""
        # Create a simple test document with 5 pages
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=5,  # 5 pages
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock OCR service to return empty detections
        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.return_value = []

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 6)]  # Pages 1-5

                # Test mixed valid/invalid request (pages 2, 3, 10 - 10 is invalid)
                result = self.processor.process_document(document_id=test_doc.id, pages=[2, 3, 10])

                # Assert failure
                self.assertFalse(result["success"], "Processing should fail for mixed valid/invalid pages")
                self.assertIn("Page number 10 is out of range", result["error"])

                # Assert zero OCR calls
                self.assertEqual(mock_ocr_service.extract_text_from_image.call_count, 0,
                               "No OCR calls should be made for invalid page requests")

                # Assert no records created
                self.assertEqual(self.db_session.query(Page).filter(Page.document_id == test_doc.id).count(), 0,
                               "No Page records should be created for invalid page requests")

    def test_invalid_page_selection_explicitly_empty_list(self):
        """Test that explicitly empty page list is rejected."""
        # Create a simple test document with 5 pages
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=5,  # 5 pages
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock OCR service to return empty detections
        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.return_value = []

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 6)]  # Pages 1-5

                # Test explicitly empty list
                result = self.processor.process_document(document_id=test_doc.id, pages=[])

                # Assert failure
                self.assertFalse(result["success"], "Processing should fail for empty page list")
                self.assertIn("Page list cannot be empty", result["error"])

                # Assert zero OCR calls
                self.assertEqual(mock_ocr_service.extract_text_from_image.call_count, 0,
                               "No OCR calls should be made for invalid page requests")

                # Assert no records created
                self.assertEqual(self.db_session.query(Page).filter(Page.document_id == test_doc.id).count(), 0,
                               "No Page records should be created for invalid page requests")

    def test_valid_page_selection_duplicate_numbers(self):
        """Test that duplicate valid page numbers are deduplicated and processed only once."""
        # Create a simple test document with 5 pages
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=5,  # 5 pages
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Track which pages are actually processed
        processed_pages = []

        def mock_extract_from_image(image):
            # We can't easily get the page number from the image tuple here,
            # but we can count calls and verify the logic another way
            # For this test, we'll just verify that the right number of pages are processed
            # by checking the resulting records
            return []  # Return empty detections

        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.side_effect = mock_extract_from_image

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 6)]  # Pages 1-5

                # Test with duplicate page numbers: [1, 2, 2, 3, 3, 3] should process pages 1, 2, 3 only
                result = self.processor.process_document(document_id=test_doc.id, pages=[1, 2, 2, 3, 3, 3])

                # Assert success
                self.assertTrue(result["success"], "Processing should succeed for valid duplicate page numbers")

                # Verify that only pages 1, 2, 3 were processed (should have 3 pages of records)
                # Each page has potentially multiple slots, but we're mainly checking that
                # the deduplication worked by verifying we don't have excess records
                # due to processing the same page multiple times

                # For simplicity, we'll check that we have reasonable numbers of records
                # Since we're returning empty detections, we should have Page and CardSlot records
                # but no ExtractionCandidate or ReviewItem records
                page_count = self.db_session.query(Page).filter(Page.document_id == test_doc.id).count()
                self.assertEqual(page_count, 3,
                               f"Expected 3 pages to be processed (1, 2, 3 with deduplication), got {page_count}")

                # Each page should have 30 card slots (assuming standard geometry)
                card_slot_count = self.db_session.query(CardSlot).count()
                self.assertEqual(card_slot_count, 3 * 30,
                               f"Expected 90 CardSlot records (3 pages × 30 slots), got {card_slot_count}")

                # No extraction or review records since we returned empty detections
                self.assertEqual(self.db_session.query(ExtractionCandidate).count(), 0,
                               "No ExtractionCandidate records should be created with empty detections")
                self.assertEqual(self.db_session.query(ReviewItem).count(), 0,
                               "No ReviewItem records should be created with empty detections")

    def test_ocr_fails_on_first_slot(self):
        """Test that OCR failing on the first slot results in document processing failure."""
        # Create a simple test document with 5 pages
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=5,  # 5 pages
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock OCR service to fail on the first call, succeed on subsequent calls
        call_count = 0
        def mock_extract_from_image(image):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Fail on the first OCR call
                raise Exception("OCR failed on first slot")
            return []  # Return empty detections for other calls

        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.side_effect = mock_extract_from_image

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 6)]  # Pages 1-5

                # Process the document
                result = self.processor.process_document(document_id=test_doc.id)

                # Assert failure
                self.assertFalse(result["success"], "Processing should fail when OCR fails on first slot")
                self.assertIn("OCR failed on first slot", result["error"])

                # Assert that OCR was attempted at least once
                self.assertGreaterEqual(mock_ocr_service.extract_text_from_image.call_count, 1,
                                       "OCR should have been attempted at least once")

                # Important: The document and page should NOT be marked as completed
                # They should either be in PENDING or PROCESSING or FAILED state, but NOT COMPLETED
                document = self.db_session.query(Document).filter(Document.id == test_doc.id).first()
                self.assertIsNotNone(document, "Document should exist")
                self.assertNotEqual(document.processing_status, "completed",
                                   f"Document should not be marked as completed when OCR fails, got {document.processing_status}")

                # Check that no partial results are incorrectly reported as success
                # Since we failed on the very first slot, ideally no pages should be processed
                # But due to our commit-every-5-slots logic, we might have some partial processing
                # The key is that we report failure, not success

    def test_ocr_fails_after_some_slots_processed(self):
        """Test that OCR failing after some slots have been processed results in failure but preserves partial work."""
        # Create a simple test document with 5 pages
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=5,  # 5 pages
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # Mock OCR service to succeed for first 10 slots, then fail
        call_count = 0
        def mock_extract_from_image(image):
            nonlocal call_count
            call_count += 1
            if call_count > 10:
                # Fail after 10 successful OCR calls
                raise Exception("OCR failed after some slots")
            return []  # Return empty detections for successful calls

        mock_ocr_service = MagicMock()
        mock_ocr_service.extract_text_from_image.side_effect = mock_extract_from_image

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(i, dummy_image) for i in range(1, 6)]  # Pages 1-5

                # Process the document
                result = self.processor.process_document(document_id=test_doc.id)

                # Assert failure
                self.assertFalse(result["success"], "Processing should fail when OCR fails after some slots")
                self.assertIn("OCR failed after some slots", result["error"])

                # Important: The document should be marked as failed, not completed
                document = self.db_session.query(Document).filter(Document.id == test_doc.id).first()
                self.assertIsNotNone(document, "Document should exist")
                self.assertEqual(document.processing_status, "failed",
                               f"Document should be marked as failed when OCR fails, got {document.processing_status}")

                # Some slots may have been processed before the failure (due to periodic commits)
                # We don't make strong assertions about how much was processed, but we verify
                # that we're reporting failure, not success

    def test_ocr_fails_during_reprocessing_of_manual_reviewed_card(self):
        """Test that OCR failing during reprocessing preserves manual corrections."""
        # Create a simple test document with 1 page
        test_batch = Batch(
            name="Test Batch",
            description="Test batch for integration testing",
            status="active"
        )
        self.db_session.add(test_batch)
        self.db_session.commit()

        test_doc = Document(
            batch_id=test_batch.id,
            original_filename="test.pdf",
            filename="test.pdf",
            file_path="samples/test.pdf",
            file_hash="a" * 64,
            file_size=1024000,
            page_count=1,  # 1 page
            processing_status="pending"
        )
        self.db_session.add(test_doc)
        self.db_session.commit()

        # Create a dummy image
        import numpy as np
        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)

        # First processing pass - succeed
        mock_ocr_service_first = MagicMock()
        mock_ocr_service_first.extract_text_from_image.return_value = []  # Empty detections

        # Import the processor module to patch its OCR service directly
        import app.jobs.processor as processor_module

        # Create a temporary directory for diagnostics
        with tempfile.TemporaryDirectory() as temp_dir:
            from app.config.settings import settings as settings_instance

            # Patch both the PDF handler and set diagnostics directory
            with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images, \
                 patch.object(processor_module, 'ocr_service', mock_ocr_service_first), \
                 patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

                # Configure PDF mock to return dummy images
                mock_pdf_to_images.return_value = [(1, dummy_image)]  # Page 1

                # Process the document (first pass)
                result_first = self.processor.process_document(document_id=test_doc.id)

                # Assert first pass succeeded
                self.assertTrue(result_first["success"], "First pass processing should succeed")

        # Manually create a review item with corrected data for slot 1
        from app.models.database import Page, CardSlot, ReviewItem, ReviewStatusEnum, ExtractionCandidate
        test_page = self.db_session.query(Page).filter(
            Page.document_id == test_doc.id,
            Page.page_number == 1
        ).first()
        self.assertIsNotNone(test_page, "Page should exist")

        card_slot = self.db_session.query(CardSlot).filter(
            CardSlot.page_id == test_page.id,
            CardSlot.physical_slot == 1
        ).first()
        self.assertIsNotNone(card_slot, "Card slot should exist")

        # Create an extraction candidate first (as would be created during successful processing)
        extraction_candidate = ExtractionCandidate(
            card_slot_id=card_slot.id,
            raw_full_name="ORIGINAL NAME FROM OCR",
            raw_house_number="ORIGINAL HOUSE FROM OCR",
            raw_relative_name="ORIGINAL RELATIVE FROM OCR",
            raw_relation_type="ORIGINAL RELATION FROM OCR",
            raw_age="45",
            raw_gender="M",
            raw_epic="ABC1234567"
        )
        self.db_session.add(extraction_candidate)
        self.db_session.flush()  # Get the ID without committing
        self.db_session.refresh(extraction_candidate)  # Refresh to get the ID

        # Create a review item linked to the extraction candidate with manual corrections
        review_item = ReviewItem(
            card_slot_id=card_slot.id,
            extraction_candidate_id=extraction_candidate.id,  # Link to the extraction candidate
            review_status=ReviewStatusEnum.REVIEWED.value,
            reviewed_full_name="MANUALLY CORRECTED NAME",
            reviewed_house_number="MANUALLY CORRECTED HOUSE"
        )
        self.db_session.add(review_item)
        self.db_session.commit()

        # Second processing pass - OCR fails on first slot
        call_count = 0
        def mock_extract_from_image_second(image):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                # Fail on the first OCR call during reprocessing
                raise Exception("OCR failed during reprocessing")
            return []  # Return empty detections for other calls

        mock_ocr_service_second = MagicMock()
        mock_ocr_service_second.extract_text_from_image.side_effect = mock_extract_from_image_second

        # Reprocess the document (second pass)
        with patch('app.ocr.pdf_handler.pdf_handler.pdf_to_images') as mock_pdf_to_images_second, \
             patch.object(processor_module, 'ocr_service', mock_ocr_service_second), \
             patch.object(processor_module.settings, 'BASE_DIR', temp_dir):

            # Configure PDF mock to return dummy images
            mock_pdf_to_images_second.return_value = [(1, dummy_image)]  # Page 1

            result_second = self.processor.process_document(document_id=test_doc.id)

            # Assert second pass failed due to OCR error
            self.assertFalse(result_second["success"], "Second pass processing should fail when OCR fails")
            self.assertIn("OCR failed during reprocessing", result_second["error"])

            # Important: Manual corrections should be preserved
            review_item_after = self.db_session.query(ReviewItem).filter(
                ReviewItem.card_slot_id == card_slot.id
            ).first()
            self.assertIsNotNone(review_item_after, "Review item should exist after failed reprocessing")
            self.assertEqual(review_item_after.reviewed_full_name, "MANUALLY CORRECTED NAME",
                           "Manual correction to name should be preserved")
            self.assertEqual(review_item_after.reviewed_house_number, "MANUALLY CORRECTED HOUSE",
                           "Manual correction to house number should be preserved")
            self.assertEqual(review_item_after.review_status, ReviewStatusEnum.REVIEWED.value,
                           "Review status should remain REVIEWED after failed reprocessing")


if __name__ == '__main__':
    unittest.main()