"""
Field parsing utilities for extracting structured data from OCR text.
"""
import re
from typing import Dict, Any, Optional, Tuple, List
import logging
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class FieldParser:
    """
    Parse and normalize fields extracted from OCR.
    """

    def __init__(self):
        """Initialize the field parser."""
        # EPIC pattern: 3 letters followed by 7 digits
        self.epic_pattern = re.compile(r'^[A-Z]{3}[0-9]{7}$')

        # Common relation types
        self.relation_types = {
            'FATHER': ['FATHER', 'FTR', 'FT'],
            'HUSBAND': ['HUSBAND', 'HSBN', 'HUB', 'HUSB'],
            'MOTHER': ['MOTHER', 'MTHR', 'MT']
        }

        # Gender mappings
        self.gender_mappings = {
            'M': ['MALE', 'M', 'MAN', 'MALE'],
            'F': ['FEMALE', 'F', 'WOMAN', 'FEMALE'],
            'O': ['OTHER', 'O', 'OTHERS', 'THIRD GENDER']
        }

        # Pre-compiled regex patterns for labeled fields (case-insensitive, whitespace tolerant)
        self._name_re = re.compile(r'^\s*Name\s*:\s*(.*)$', re.IGNORECASE)
        self._house_number_re = re.compile(r'^\s*House\s+Number\s*:\s*(.*)$', re.IGNORECASE)
        self._father_name_re = re.compile(r'^\s*Father\'s\s+Name\s*:\s*(.*)$', re.IGNORECASE)
        self._husband_name_re = re.compile(r'^\s*Husband\'s\s+Name\s*:\s*(.*)$', re.IGNORECASE)
        self._mother_name_re = re.compile(r'^\s*Mother\'s\s+Name\s*:\s*(.*)$', re.IGNORECASE)
        # Age and gender patterns as per the user's request
        self._age_re = re.compile(r'\bAge\s*:\s*(\d+)\b', re.IGNORECASE)
        self._gender_re = re.compile(r'\bGender\s*:\s*(Male|Female|Other|Third\s+Gender|M|F|O)\b', re.IGNORECASE)

    def normalize_epic(self, epic: str) -> Optional[str]:
        """
        Normalize EPIC number according to specification.
        Outer-whitespace trimming and uppercase letters only.

        Args:
            epic: Raw EPIC string from OCR

        Returns:
            Normalized EPIC or None if invalid
        """
        if not epic:
            return None

        # Trim outer whitespace
        epic = epic.strip()

        # Convert to uppercase
        epic = epic.upper()

        # Remove any non-alphanumeric characters that might be OCR artifacts
        # But keep the core pattern
        epic_cleaned = re.sub(r'[^A-Z0-9]', '', epic)

        # Validate against pattern
        if self.epic_pattern.match(epic_cleaned):
            return epic_cleaned
        else:
            # Log invalid format but return None (don't guess corrections)
            logger.debug(f"Invalid EPIC format: {epic}")
            return None

    def parse_age(self, age_str: str) -> Optional[int]:
        """
        Parse age string to integer.

        Args:
            age_str: Raw age string from OCR

        Returns:
            Age as integer or None if invalid/missing
        """
        if not age_str:
            return None

        # Extract digits
        age_match = re.search(r'\d+', age_str.strip())
        if age_match:
            try:
                age = int(age_match.group())
                # Reasonable age range for electoral rolls
                if 18 <= age <= 120:
                    return age
                else:
                    logger.debug(f"Age outside reasonable range: {age}")
                    return None
            except ValueError:
                return None
        return None

    def normalize_gender(self, gender_str: str) -> Optional[str]:
        """
        Normalize gender to MALE, FEMALE, or OTHER.

        Args:
            gender_str: Raw gender string from OCR

        Returns:
            Normalized gender or None if unreadable
        """
        if not gender_str:
            return None

        gender_upper = gender_str.strip().upper()

        # Check against known mappings
        for normalized, variants in self.gender_mappings.items():
            if gender_upper in variants:
                if normalized == 'M':
                    return 'MALE'
                elif normalized == 'F':
                    return 'FEMALE'
                elif normalized == 'O':
                    return 'OTHER'

        # If not found, return None (unreadable/unknown)
        logger.debug(f"Unrecognized gender: {gender_str}")
        return None

    def parse_relation(self, relation_str: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Parse relation string to determine type and extract relative name.

        Args:
            relation_str: Raw relation string from OCR (e.g., "FTR: John Doe")
                or "Father's Name: John Doe"

        Returns:
            Tuple of (relation_type, relative_name)
            relation_type: 'FATHER', 'HUSBAND', or None
            relative_name: Extracted name or None
        """
        if not relation_str:
            return None, None

        relation_str = relation_str.strip()

        # Common patterns: "FTR: Name", "HSBN: Name", "MT: Name"
        # Or sometimes just the name with implied relation

        # Try to split by colon first
        if ':' in relation_str:
            parts = relation_str.split(':', 1)
            relation_part = parts[0].strip().upper()
            name_part = parts[1].strip() if len(parts) > 1 else ""

            # Check relation part
            if any(variant in relation_part for variant in self.relation_types['FATHER']):
                return 'FATHER', name_part if name_part else None
            elif any(variant in relation_part for variant in self.relation_types['HUSBAND']):
                return 'HUSBAND', name_part if name_part else None
            elif any(variant in relation_part for variant in self.relation_types['MOTHER']):
                # Mother relation - treat as other for now (spec only mentions Father/Husband for Excel)
                return None, name_part if name_part else None
            else:
                # Unrecognized relation type
                logger.debug(f"Unrecognized relation type: {relation_part}")
                return None, None
        else:
            # No colon - assume it's just a name
            # Without explicit relation indicator, we can't determine type
            return None, relation_str if relation_str else None

    def clean_name(self, name: str) -> Optional[str]:
        """
        Clean and normalize a name string.
        Preserve spelling, spaces and Unicode; do not force to uppercase.

        Args:
            name: Raw name string from OCR

        Returns:
            Cleaned name or None if empty
        """
        if not name:
            return None

        # Strip whitespace
        name = name.strip()

        # Remove obvious OCR artifacts (but be conservative)
        # Remove multiple spaces
        name = re.sub(r'\s+', ' ', name)

        # Remove leading/trailing punctuation that's likely OCR artifacts
        name = re.sub(r'^[^\w\s]+|[^\w\s]+$', '', name)

        # Return None if empty after cleaning
        if not name:
            return None

        return name

    def clean_address_field(self, field: str) -> Optional[str]:
        """
        Clean address-related fields (house number, etc.).

        Args:
            field: Raw field string from OCR

        Returns:
            Cleaned field or None if empty
        """
        return self.clean_name(field)  # Same cleaning logic for now

    def clean_phone_number(self, phone: str) -> Optional[str]:
        """
        Clean phone number while preserving leading zeros and + prefix.
        Do not convert to numeric value.

        Args:
            phone: Raw phone string from OCR

        Returns:
            Cleaned phone number or None if empty
        """
        if not phone:
            return None

        # Strip whitespace
        phone = phone.strip()

        # Preserve leading + and spaces/dashes within the number
        # But remove obvious OCR artifacts at edges
        phone = re.sub(r'^[^\d\+]+|[^\d\+]+$', '', phone)

        # Internal cleanup - normalize separators
        # Replace multiple spaces/dashes with single space
        phone = re.sub(r'[\s\-]+', ' ', phone)

        # Remove leading/trailing spaces
        phone = phone.strip()

        # Return None if empty
        if not phone:
            return None

        return phone

    def _resolve_candidates(self, field_name: str, candidates: List[Any]) -> Tuple[Any, Optional[List[Any]]]:
        """
        Resolve candidates for a field.
        Returns (value, conflict_list) where:
          - value is the resolved value (None if no candidates or conflict)
          - conflict_list is None if no conflict, otherwise a list of the conflicting values (unique, differing only)
        """
        if not candidates:
            return None, None
        # Deduplicate: only differing values matter
        unique_candidates = list(dict.fromkeys(candidates))  # preserves order of first occurrence
        if len(unique_candidates) == 1:
            return unique_candidates[0], None
        else:
            # More than one distinct value -> conflict
            return None, unique_candidates

    def parse_voter_card_lines(self, lines: List[str]) -> Dict[str, Any]:
        """
        Parse a list of OCR text lines (strings) to extract voter card fields.

        Implements the assignment rules:
          1. EPIC: accept a standalone validated EPIC reading.
          2. Name: match a line beginning with "Name", followed by a colon.
             Clean only the value after the label.
             Do not match "Father's Name" as the person's name.
          3. Relation: match explicit Father's/Husband's/Mother's Name labels.
             Parse only that labeled line.
          4. House number: match "House Number:" and extract only its same-line value.
             An empty value stays None. Never consume the next line or photo text.
          5. Age and gender: extract their labeled values separately, including
             when both occur on one line, such as "Age : 41 Gender : Male".
          6. Serial number: use the standalone integer in the card's top-left
             serial region, supported by bbox position—not an arbitrary number.
             (Note: This method does not receive bbox information; the caller
              must provide the serial number separately if needed via bbox.)
          7. Ignore "Photo" and "Available".
          8. Conflicting candidates must produce a diagnostic rather than silently
             keeping the first or overwriting a value.

        Args:
            lines: List of strings, each line of OCR text (already stripped? We'll strip inside).

        Returns:
            A dictionary with keys: 'epic', 'name', 'relation_type', 'relation_name',
            'house_number', 'age', 'gender', 'serial_number'.
            Values are either the parsed value or None if not found.
            If there are conflicting candidates for a field, the field will be None
            and a '_conflicts' key will be present with details.
        """
        # We'll collect candidates for each field
        candidates = {
            'epic': [],
            'name': [],
            'relation_type': [],
            'relation_name': [],
            'house_number': [],
            'age': [],
            'gender': [],
            # serial_number is handled by caller via bbox, so we leave it empty here
        }

        for line in lines:
            if line is None:
                continue
            line_stripped = line.strip()
            if line_stripped == "":
                continue

            # Ignore "Photo" and "Available"
            if line_stripped.lower() in ("photo", "available"):
                continue

            # 1. EPIC: standalone validated EPIC reading
            epic_candidate = self.normalize_epic(line_stripped)
            if epic_candidate is not None:
                candidates['epic'].append(epic_candidate)

            # 2. Name: line beginning with "Name", followed by a colon
            m = self._name_re.match(line_stripped)
            if m:
                name_candidate = self.clean_name(m.group(1))
                if name_candidate is not None:
                    candidates['name'].append(name_candidate)

            # 3. Relation: match explicit Father's/Husband's/Mother's Name labels
            m = self._father_name_re.match(line_stripped)
            if m:
                rel_name = self.clean_name(m.group(1))
                if rel_name is not None:
                    candidates['relation_type'].append('FATHER')
                    candidates['relation_name'].append(rel_name)
            else:
                m = self._husband_name_re.match(line_stripped)
                if m:
                    rel_name = self.clean_name(m.group(1))
                    if rel_name is not None:
                        candidates['relation_type'].append('HUSBAND')
                        candidates['relation_name'].append(rel_name)
                else:
                    m = self._mother_name_re.match(line_stripped)
                    if m:
                        rel_name = self.clean_name(m.group(1))
                        if rel_name is not None:
                            candidates['relation_type'].append('MOTHER')
                            candidates['relation_name'].append(rel_name)

            # 4. House number: match "House Number:" and extract only its same-line value
            m = self._house_number_re.match(line_stripped)
            if m:
                hn_candidate = self.clean_address_field(m.group(1))
                candidates['house_number'].append(hn_candidate)

            # 5. Age and gender: extract their labeled values separately using search
            m = self._age_re.search(line_stripped)
            if m:
                age_candidate = self.parse_age(m.group(1))
                if age_candidate is not None:
                    candidates['age'].append(age_candidate)
            m = self._gender_re.search(line_stripped)
            if m:
                raw_gender = m.group(1)
                gender_candidate = self.normalize_gender(raw_gender)
                if gender_candidate is not None:
                    candidates['gender'].append(gender_candidate)

            # Note: We do not parse serial number here because it requires bbox position.
            # The caller should handle serial number separately via bbox.

        # Now, resolve candidates: if exactly one candidate, use it; if zero, None; if more than one distinct value, conflict -> None and record conflict.
        result = {}
        conflicts = {}

        for field, cand_list in candidates.items():
            value, conflict_list = self._resolve_candidates(field, cand_list)
            result[field] = value
            if conflict_list is not None:
                if '_conflicts' not in result:
                    result['_conflicts'] = {}
                result['_conflicts'][field] = conflict_list

        # Serial number is not extracted from lines; caller must provide it via bbox.
        result['serial_number'] = None

        return result

    def parse_voter_card_detections(
        self,
        detections: List[Dict[str, Any]],
        card_width: Optional[int] = None,
        card_height: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Parse OCR detection dictionaries to extract voter card fields.

        This method handles labeled fields by extracting text lines in reading order
        and reusing the line parser. It also attempts to attach unlabeled continuations
        to the preceding person-name or relative-name field under specific conditions.

        Args:
            detections: List of OCR detection dictionaries, each expected to have
                        at least a 'text' key and optionally a 'bbox' key.
            card_width: Width of the card in pixels (optional, but unused now).
            card_height: Height of the card in pixels (optional, but unused now).

        Returns:
            A dictionary with keys: 'epic', 'name', 'relation_type', 'relation_name',
            'house_number', 'age', 'gender', 'serial_number'.
            Values are either the parsed value or None if not found.
            If there are conflicting candidates for a field, the field will be None
            and a '_conflicts' key will be present with details.
            Unassigned text lines are stored in '_unassigned' if any.
        """
        if not detections:
            return {
                'epic': None,
                'name': None,
                'relation_type': None,
                'relation_name': None,
                'house_number': None,
                'age': None,
                'gender': None,
                'serial_number': None,
                '_unassigned': []
            }
        # Collect EPIC readings before spatial filtering.
        epic_candidates = []
        for detection in detections:
            text = detection.get("text")
            if isinstance(text, str):
                epic = self.normalize_epic(text.strip())
                if epic is not None:
                    epic_candidates.append(epic)

        # Step 1: Exclude photo-region detections before grouping lines
        # We'll use layout geometry scaled to card coordinates to identify photo region
        # Assuming photo region is on the right side of the card (typically right 30-40%)
        filtered_detections = []
        photo_region_threshold = 0.6  # Left 60% is details column, right 40% is photo region

        for det in detections:
            bbox = det.get('bbox')
            if not bbox or len(bbox) == 0:
                # If no bbox, we can't determine if it's in photo region, so keep it
                filtered_detections.append(det)
                continue

            # Calculate center x of the bbox
            xs = [point[0] for point in bbox]
            center_x = sum(xs) / len(xs)

            # If card_width is provided, use it to determine if in photo region
            if card_width is not None:
                # Normalize center_x to [0, 1] range based on card width
                normalized_x = center_x / card_width
                # Keep detections in the details column (left side)
                if normalized_x < photo_region_threshold:
                    filtered_detections.append(det)
                # Else it's in photo region, exclude it
            else:
                # If card_width not provided, we can't determine photo region, so keep it
                filtered_detections.append(det)

        # Step 1.5: Exclude standalone integer detections in the serial box before grouping lines
        # Source serial extraction is cancelled; ignore standalone integers in top-left serial box
        serial_filtered_detections = []
        if card_width is not None and card_height is not None:
            serial_box_width_ratio = 420.0 / 1287.0
            serial_box_height_ratio = 420.0 / 1287.0
            serial_box_x_max = card_width * serial_box_width_ratio
            serial_box_y_max = card_height * serial_box_height_ratio

            for det in filtered_detections:
                # Check if this is a standalone integer detection
                text = det.get('text', '')
                is_standalone_integer = isinstance(text, str) and re.fullmatch(r'\s*\d+\s*', text) is not None

                if is_standalone_integer:
                    # For standalone integers, check if they're in the serial box
                    bbox = det.get('bbox')
                    if bbox and len(bbox) > 0:
                        # Calculate center of the bbox
                        xs = [point[0] for point in bbox]
                        ys = [point[1] for point in bbox]
                        center_x = sum(xs) / len(xs)
                        center_y = sum(ys) / len(ys)

                        # Exclude if center falls within the serial box
                        if center_x < serial_box_x_max and center_y < serial_box_y_max:
                            # This standalone integer is in the serial box - exclude it intentionally
                            continue

                # Keep all other detections (non-integers, or integers outside serial box)
                serial_filtered_detections.append(det)
        else:
            # If card dimensions not provided, we can't determine serial box, so keep all
            serial_filtered_detections = filtered_detections

        # Step 2: Sort detections by reading order (top to bottom, left to right)
        def get_sort_key(det):
            bbox = det.get('bbox')
            if bbox and len(bbox) > 0 and len(bbox[0]) >= 2:
                # bbox is a list of 4 points, each [x, y]
                # Use the first point (top-left of the bbox) for sorting.
                return (bbox[0][1], bbox[0][0])
            else:
                return (0, 0)

        sorted_detections = sorted(serial_filtered_detections, key=get_sort_key)

        # Step 3: Group detections into lines based on vertical proximity
        lines = []  # each element is a list of detections belonging to the same line
        line_heights = []  # height of each line in pixels
        current_line = []

        # Helper to compute vertical center of a bbox
        def vertical_center(bbox):
            ys = [point[1] for point in bbox]
            return sum(ys) / len(ys)

        # Helper to compute line height (range of y in the line)
        def line_height(line_dets):
            all_points = [point for det in line_dets for point in det['bbox']]
            ys = [point[1] for point in all_points]
            return max(ys) - min(ys) if ys else 0

        for det in sorted_detections:
            if not current_line:
                current_line.append(det)
            else:
                last_det = current_line[-1]
                vc_det = vertical_center(det['bbox'])
                vc_last = vertical_center(last_det['bbox'])
                # Use half the median line height of the lines we have so far as threshold
                if line_heights:
                    sorted_heights = sorted(line_heights)
                    median_line_height = sorted_heights[len(sorted_heights)//2]
                    threshold = median_line_height * 0.5
                else:
                    threshold = 5.0  # default threshold in pixels
                if abs(vc_det - vc_last) < threshold:
                    current_line.append(det)
                else:
                    # Finalize the current line
                    if current_line:
                        lines.append(current_line)
                        line_heights.append(line_height(current_line))
                    current_line = [det]
        if current_line:
            lines.append(current_line)
            line_heights.append(line_height(current_line))

        # Step 4: For each line, compute the line text and line bbox
        line_texts = []
        line_bboxes = []  # each as (min_x, min_y, max_x, max_y)
        for line_dets in lines:
            # Sort detections in the line by left-to-right (x of top-left point)
            sorted_line_dets = sorted(line_dets, key=lambda det: det['bbox'][0][0])
            # Concatenate the texts
            line_text = ' '.join(det.get('text', '').strip() for det in sorted_line_dets if det.get('text'))
            line_texts.append(line_text)
            # Compute the bbox of the line
            all_points = [point for det in line_dets for point in det['bbox']]
            xs = [point[0] for point in all_points]
            ys = [point[1] for point in all_points]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            line_bboxes.append((min_x, min_y, max_x, max_y))

        # Step 5: Initialize fields and tracking for single-pass processing
        # We'll collect candidates for each field
        candidates = {
            'epic': epic_candidates,
            'name': [],
            'relation_type': [],
            'relation_name': [],
            'house_number': [],
            'age': [],
            'gender': [],
        }

        # Track unassigned lines for warning
        unassigned_line_indices = []

        # Track active name candidate (the current field we're building)
        active_name_field = None  # 'name', 'relation_name'
        active_name_value = ""    # Current accumulated value for the active field
        active_name_line_idx = -1 # Index of line where active field started

        # Track the last seen label to know when to end name continuation
        last_label_type = None    # 'name', 'father', 'husband', 'mother', 'house', 'age', 'gender', 'other'

        # Track if we've already warned about unassigned text to avoid duplicate warnings
        warned_about_unassigned = False

        # Step 6: Single pass over lines in reading order
        for idx, line_text in enumerate(line_texts):
            line_stripped = line_text.strip()
            if line_stripped == "":
                # Empty line - treat as used but don't process
                continue

            # Ignore "Photo" and "Available" (should already be filtered, but double-check)
            if line_stripped.lower() in ("photo", "available"):
                continue

            matched = False

            # Check for EPIC (standalone validated EPIC reading)
            epic_candidate = self.normalize_epic(line_stripped)
            if epic_candidate is not None:
                candidates['epic'].append(epic_candidate)
                matched = True
                # EPIC ends name continuation
                if active_name_field is not None:
                    # Finalize the active name candidate
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""
                last_label_type = 'epic'
                continue

            # Check for Name label
            m = self._name_re.match(line_stripped)
            if m:
                # Finalize previous active name candidate if any
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))

                # Start new name candidate
                active_name_field = 'name'
                active_name_value = self.clean_name(m.group(1)) or ""
                active_name_line_idx = idx
                matched = True
                last_label_type = 'name'
                continue

            # Check for House Number label (ends name continuation)
            m = self._house_number_re.match(line_stripped)
            if m:
                # Finalize active name candidate if any
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""

                hn_candidate = self.clean_address_field(m.group(1))
                candidates['house_number'].append(hn_candidate)
                matched = True
                last_label_type = 'house'
                continue

            # Check for Father's Name label
            m = self._father_name_re.match(line_stripped)
            if m:
                # Finalize active name candidate if any
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""

                # Start new relative name candidate (father)
                active_name_field = 'relation_name'
                active_name_value = self.clean_name(m.group(1)) or ""
                active_name_line_idx = idx
                # Also track the relation type separately
                rel_name = self.clean_name(m.group(1))
                if rel_name is not None:
                    candidates['relation_type'].append('FATHER')
                matched = True
                last_label_type = 'father'
                continue

            # Check for Husband's Name label
            m = self._husband_name_re.match(line_stripped)
            if m:
                # Finalize active name candidate if any
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""

                # Start new relative name candidate (husband)
                active_name_field = 'relation_name'
                active_name_value = self.clean_name(m.group(1)) or ""
                active_name_line_idx = idx
                # Also track the relation type separately
                rel_name = self.clean_name(m.group(1))
                if rel_name is not None:
                    candidates['relation_type'].append('HUSBAND')
                matched = True
                last_label_type = 'husband'
                continue

            # Check for Mother's Name label
            m = self._mother_name_re.match(line_stripped)
            if m:
                # Finalize active name candidate if any
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""

                # Start new relative name candidate (mother)
                active_name_field = 'relation_name'
                active_name_value = self.clean_name(m.group(1)) or ""
                active_name_line_idx = idx
                # Also track the relation type separately
                rel_name = self.clean_name(m.group(1))
                if rel_name is not None:
                    candidates['relation_type'].append('MOTHER')
                matched = True
                last_label_type = 'mother'
                continue

            # Check for Age anywhere in the line (independent search)
            m = self._age_re.search(line_stripped)
            if m:
                # Finalize active name candidate if any (Age ends name continuation)
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""

                age_candidate = self.parse_age(m.group(1))
                if age_candidate is not None:
                    candidates['age'].append(age_candidate)
                matched = True
                last_label_type = 'age'
                # Note: We don't break here because Gender might also be on the same line

            # Check for Gender anywhere in the line (independent search)
            m = self._gender_re.search(line_stripped)
            if m:
                # Finalize active name candidate if any (Gender ends name continuation)
                if active_name_field is not None:
                    if active_name_field == 'name' and active_name_value:
                        candidates['name'].append(self.clean_name(active_name_value))
                    elif active_name_field == 'relation_name' and active_name_value:
                        candidates['relation_name'].append(self.clean_name(active_name_value))
                    active_name_field = None
                    active_name_value = ""

                raw_gender = m.group(1)
                gender_candidate = self.normalize_gender(raw_gender)
                if gender_candidate is not None:
                    candidates['gender'].append(gender_candidate)
                matched = True
                last_label_type = 'gender'

            # If we matched any label or field, we've already handled continuation ending above
            if matched:
                continue

            # If we get here, the line did not match any label or field
            # Check if it should be attached to the active name candidate

            # First, check if it's obviously not a continuation (standalone integer, etc.)
            if re.fullmatch(r'\s*\d+\s*', line_stripped):
                # Standalone integer, not a continuation
                unassigned_line_indices.append(idx)
                last_label_type = 'other'
                continue

            # Check if we have an active name candidate to attach to
            if active_name_field is not None and active_name_line_idx != -1:
                # Check alignment and vertical proximity
                prev_bbox = line_bboxes[active_name_line_idx]
                curr_bbox = line_bboxes[idx]

                # Condition 1: Lies below that field and before the next recognized field label
                # Since we're processing in order and haven't seen a label yet for this line,
                # and we know there are no labels between active_name_line_idx and idx
                # (because we would have processed them and set active_name_field to None),
                # this condition is satisfied if active_name_field is not None

                # Condition 2: Aligns with the left text column
                if line_heights:
                    sorted_heights = sorted(line_heights)
                    median_line_height = sorted_heights[len(sorted_heights)//2]
                    alignment_threshold = median_line_height * 0.5
                else:
                    alignment_threshold = 5.0

                if abs(curr_bbox[0] - prev_bbox[0]) > alignment_threshold:
                    # Not aligned, treat as unassigned
                    unassigned_line_indices.append(idx)
                    last_label_type = 'other'
                    continue

                # Condition 3: Has a vertical gap consistent with nearby text-line heights
                gap = curr_bbox[1] - prev_bbox[3]  # current line's min_y - preceding line's max_y
                if line_heights:
                    sorted_heights = sorted(line_heights)
                    median_line_height = sorted_heights[len(line_heights)//2]
                    gap_threshold = median_line_height * 1.5
                else:
                    gap_threshold = 20.0

                if gap < 0 or gap > gap_threshold:
                    # Gap too large or negative (overlapping), treat as unassigned
                    unassigned_line_indices.append(idx)
                    last_label_type = 'other'
                    continue

                # Condition 4: Is outside the photo region
                # We already filtered detections, so this should be satisfied
                # But double-check if card_width is available
                if card_width is not None:
                    line_center_x = (curr_bbox[0] + curr_bbox[2]) / 2.0
                    if line_center_x > card_width * 0.6:  # assuming photo region is on the right 40%
                        unassigned_line_indices.append(idx)
                        last_label_type = 'other'
                        continue

                # If we get here, the line meets all conditions for being a continuation
                # Attach it to the active name candidate by updating the string
                # Preserve repeated words and capitalization by adding a space and the line text
                if active_name_value:
                    active_name_value += " " + line_stripped
                else:
                    active_name_value = line_stripped

                # Mark this line as used (by not adding to unassigned)
                last_label_type = 'continuation'
                continue

            # If we get here, the line is not a continuation and not matched to any label/field
            unassigned_line_indices.append(idx)
            last_label_type = 'other'

        # Step 7: Finalize any active name candidate at the end
        if active_name_field is not None:
            if active_name_field == 'name' and active_name_value:
                candidates['name'].append(self.clean_name(active_name_value))
            elif active_name_field == 'relation_name' and active_name_value:
                candidates['relation_name'].append(self.clean_name(active_name_value))

        # Step 8: Now, resolve the candidates for each field (as in the line parser)
        result = {}
        conflicts = {}

        for field, cand_list in candidates.items():
            value, conflict_list = self._resolve_candidates(field, cand_list)
            result[field] = value
            if conflict_list is not None:
                if '_conflicts' not in result:
                    result['_conflicts'] = {}
                result['_conflicts'][field] = conflict_list

        # Step 9: Add unassigned lines if any (for warning, but exclude ignored serial and photo readings)
        if unassigned_line_indices and not warned_about_unassigned:
            # Filter out any that might be serial numbers or photo readings that were incorrectly included
            # For now, we'll just store them but the requirement says to exclude ignored serial and photo readings
            # from those warnings - since we already filtered photo regions, and serial numbers are handled separately,
            # we'll assume the remaining are genuinely unassigned details text
            result['_unassigned'] = [line_texts[i] for i in unassigned_line_indices]
            # In a real implementation, we would log a warning here once
            # logger.warning(f"Unassigned details text found: {result['_unassigned']}")

        # Step 10: Serial number is not extracted from OCR; leave as None for caller to fill.
        result['serial_number'] = None

        return result