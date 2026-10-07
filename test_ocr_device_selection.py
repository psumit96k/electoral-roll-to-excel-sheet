"""
Tests for OCR CPU/GPU device selection functionality.
"""
import os
import sys
import tempfile
from unittest.mock import patch, MagicMock
import pytest

# Store original environment for restoration
original_env = dict(os.environ)

# Module-level temporary directory for imports
_import_temp_dir = tempfile.TemporaryDirectory()

def setup_test_environment(temp_dir=None):
    """Set up isolated test environment with temporary directories."""
    # Use provided temp_dir or create a new one
    if temp_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        # We need to clean up this temp_dir later
        cleanup_needed = True
    else:
        # Using pre-created temp_dir (like module-level one)
        cleanup_needed = False

    # Store original values for the environment variables we'll modify
    original_values = {}
    for key in ['BASE_DIR', 'DATA_DIR', 'MODELS_DIR', 'UPLOAD_DIR', 'EXPORT_DIR', 'DATABASE_URL', 'OCR_DEVICE']:
        original_values[key] = os.environ.get(key)

    # Remove OCR_DEVICE from environment to isolate from inherited values
    os.environ.pop("OCR_DEVICE", None)

    # Set up isolated environment variables
    os.environ.update({
        # Use temporary directories for all paths
        'BASE_DIR': temp_dir.name,
        'DATA_DIR': os.path.join(temp_dir.name, 'data'),
        'MODELS_DIR': os.path.join(temp_dir.name, 'models'),
        'UPLOAD_DIR': os.path.join(temp_dir.name, 'uploads'),
        'EXPORT_DIR': os.path.join(temp_dir.name, 'exports'),
        # Use temporary database
        'DATABASE_URL': f'sqlite:///{os.path.join(temp_dir.name, "test.db").replace(os.sep, "/")}',
    })

    return temp_dir, original_values, cleanup_needed

def teardown_test_environment(original_values, cleanup_needed=True, temp_dir=None):
    """Restore original environment for modified variables."""
    # Restore original values for the environment variables we modified
    for key, original_value in original_values.items():
        if original_value is None:
            # If the variable wasn't originally set, remove it
            os.environ.pop(key, None)
        else:
            # Restore the original value
            os.environ[key] = original_value

    # Clean up temporary directory if needed
    if cleanup_needed and temp_dir is not None:
        temp_dir.cleanup()

@pytest.fixture(autouse=True)
def isolated_environment():
    """Fixture to provide isolated environment for each test."""
    temp_dir, original_values, cleanup_needed = setup_test_environment()
    try:
        yield temp_dir.name  # Yield the actual path, not the TemporaryDirectory object
    finally:
        teardown_test_environment(original_values, cleanup_needed, temp_dir)

# We need to set up the environment before importing app.ocr.service
# to prevent the global OCR service initialization from trying to load real models

# Set up the test environment for imports using module-level temporary directory
_import_temp_dir, _original_values, _import_cleanup_needed = setup_test_environment(_import_temp_dir)
try:
    # Patch DotEnvSettingsSource to prevent loading .env file during imports
    dotenv_patch = patch('pydantic_settings.sources.DotEnvSettingsSource.__call__', return_value={})
    dotenv_patch.start()

    # Then, patch paddleocr.PaddleOCR to prevent real initialization
    paddleocr_patch = patch('paddleocr.PaddleOCR')
    mock_paddleocr = paddleocr_patch.start()
    mock_paddleocr.return_value = MagicMock()

    # Also patch the paddle CUDA functions for hardware detection tests
    is_compiled_patch = patch('paddle.is_compiled_with_cuda')
    mock_is_compiled = is_compiled_patch.start()
    mock_is_compiled.return_value = False  # Default to CPU-only build

    device_count_patch = patch('paddle.device.cuda.device_count')
    mock_device_count = device_count_patch.start()
    mock_device_count.return_value = 0  # Default to zero GPUs

    # Now we can safely import app.ocr.service - the global ocr_service will get a mocked PaddleOCR
    from app.config.settings import Settings
    from app.ocr.service import OCRService
finally:
    # Stop the patches that were only needed for the initial import
    paddleocr_patch.stop()
    is_compiled_patch.stop()
    device_count_patch.stop()
    dotenv_patch.stop()
    # Restore the environment after imports
    teardown_test_environment(_original_values, _import_cleanup_needed, _import_temp_dir)


def test_ocr_device_setting_default():
    """Test that OCR_DEVICE defaults to 'cpu'."""
    settings = Settings(_env_file=None)
    assert settings.OCR_DEVICE == "cpu"


def test_ocr_device_setting_validation():
    """Test that OCR_DEVICE setting accepts values (validation at runtime)."""
    # Test that we can set the expected values
    for valid_value in ["cpu", "gpu", "auto"]:
        settings = Settings(OCR_DEVICE=valid_value, _env_file=None)
        assert settings.OCR_DEVICE == valid_value

    # Invalid values can be set but will cause runtime errors in _resolve_ocr_device
    # This behavior is tested in test_resolve_ocr_device_invalid


def test_resolve_ocr_device_invalid():
    """Test device resolution with invalid setting fails service initialization."""
    # For this test, we need to patch PaddleOCR to make sure it's not called
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        settings = Settings(OCR_DEVICE="invalid", _env_file=None)

        try:
            service = OCRService(settings=settings)
            assert False, "Should have raised ValueError during initialization"
        except ValueError as e:
            assert "Invalid OCR_DEVICE setting" in str(e)
            assert "invalid" in str(e)
            # Verify constructor was not called
            mock_paddleocr.assert_not_called()


def test_resolve_ocr_device_cpu():
    """Test device resolution for CPU mode."""
    # Use explicit constructor mocking
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        mock_instance = MagicMock()
        mock_paddleocr.return_value = mock_instance

        settings = Settings(OCR_DEVICE="cpu", _env_file=None)
        service = OCRService(settings=settings)

        device = service._resolve_ocr_device()
        assert device == "cpu"
        # Verify PaddleOCR was called with device="cpu"
        mock_paddleocr.assert_called_once()
        call_args = mock_paddleocr.call_args
        assert 'device' in call_args.kwargs
        assert call_args.kwargs['device'] == "cpu"
        # Verify the service has the mocked instance using 'is'
        assert service.ocr is mock_instance


def test_resolve_ocr_device_gpu_available():
    """Test device resolution for GPU mode when GPU is available."""
    # Need to patch the paddle functions and PaddleOCR constructor
    with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
        with patch('paddle.device.cuda.device_count') as mock_device_count:
            with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1
                mock_instance = MagicMock()
                mock_paddleocr.return_value = mock_instance

                settings = Settings(OCR_DEVICE="gpu", _env_file=None)
                service = OCRService(settings=settings)

                device = service._resolve_ocr_device()
                assert device == "gpu:0"
                # Verify PaddleOCR was called with device="gpu:0"
                mock_paddleocr.assert_called_once()
                call_args = mock_paddleocr.call_args
                assert 'device' in call_args.kwargs
                assert call_args.kwargs['device'] == "gpu:0"
                # Verify the service has the mocked instance using 'is'
                assert service.ocr is mock_instance


def test_resolve_ocr_device_gpu_unavailable():
    """Test device resolution for GPU mode when GPU is unavailable fails service initialization."""
    # Need to patch the paddle functions and verify PaddleOCR is not called
    with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
        with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
            mock_is_compiled.return_value = False

            settings = Settings(OCR_DEVICE="gpu", _env_file=None)

            try:
                service = OCRService(settings=settings)
                assert False, "Should have raised RuntimeError during initialization"
            except RuntimeError as e:
                assert "GPU requested for OCR but GPU is not available" in str(e)
                # Verify constructor was not called
                mock_paddleocr.assert_not_called()


def test_resolve_ocr_device_auto_gpu_available():
    """Test device resolution for AUTO mode when GPU is available."""
    # Need to patch the paddle functions and PaddleOCR constructor
    with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
        with patch('paddle.device.cuda.device_count') as mock_device_count:
            with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1
                mock_instance = MagicMock()
                mock_paddleocr.return_value = mock_instance

                settings = Settings(OCR_DEVICE="auto", _env_file=None)
                service = OCRService(settings=settings)

                device = service._resolve_ocr_device()
                assert device == "gpu:0"
                # Verify PaddleOCR was called with device="gpu:0"
                mock_paddleocr.assert_called_once()
                call_args = mock_paddleocr.call_args
                assert 'device' in call_args.kwargs
                assert call_args.kwargs['device'] == "gpu:0"
                # Verify the service has the mocked instance using 'is'
                assert service.ocr is mock_instance


def test_resolve_ocr_device_auto_cpu_fallback():
    """Test device resolution for AUTO mode when GPU is unavailable (falls back to CPU)."""
    # Need to patch the paddle functions and PaddleOCR constructor
    with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
        with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
            mock_is_compiled.return_value = False
            mock_instance = MagicMock()
            mock_paddleocr.return_value = mock_instance

            settings = Settings(OCR_DEVICE="auto", _env_file=None)
            service = OCRService(settings=settings)

            device = service._resolve_ocr_device()
            assert device == "cpu"
            # Verify PaddleOCR was called with device="cpu"
            mock_paddleocr.assert_called_once()
            call_args = mock_paddleocr.call_args
            assert 'device' in call_args.kwargs
            assert call_args.kwargs['device'] == "cpu"
            # Verify the service has the mocked instance using 'is'
            assert service.ocr is mock_instance


def test_is_gpu_available_cpu_only_build():
    """Test _is_gpu_available returns False when PaddlePaddle not compiled with CUDA."""
    # Explicitly mock PaddleOCR to prevent initialization
    with patch('app.ocr.service.PaddleOCR'):
        # Patch the paddle functions for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            mock_is_compiled.return_value = False

            settings = Settings(_env_file=None)
            service = OCRService(settings=settings)

            result = service._is_gpu_available()
            assert result is False  # Use 'is' for boolean check


def test_is_gpu_available_zero_gpus():
    """Test _is_gpu_available returns False when no GPUs detected."""
    # Explicitly mock PaddleOCR to prevent initialization
    with patch('app.ocr.service.PaddleOCR'):
        # Patch the paddle functions for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            with patch('paddle.device.cuda.device_count') as mock_device_count:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 0

                settings = Settings(_env_file=None)
                service = OCRService(settings=settings)

                result = service._is_gpu_available()
                assert result is False  # Use 'is' for boolean check


def test_is_gpu_available_gpu_detected():
    """Test _is_gpu_available returns True when GPU is detected."""
    # Explicitly mock PaddleOCR to prevent initialization
    with patch('app.ocr.service.PaddleOCR'):
        # Patch the paddle functions for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            with patch('paddle.device.cuda.device_count') as mock_device_count:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1

                settings = Settings(_env_file=None)
                service = OCRService(settings=settings)

                result = service._is_gpu_available()
                assert result is True  # Use 'is' for boolean check


def test_is_gpu_available_exception():
    """Test _is_gpu_available returns False when probe raises exception."""
    # Explicitly mock PaddleOCR to prevent initialization
    with patch('app.ocr.service.PaddleOCR'):
        # Patch the paddle functions for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            mock_is_compiled.side_effect = Exception("Probe failed")

            settings = Settings(_env_file=None)
            service = OCRService(settings=settings)

            result = service._is_gpu_available()
            assert result is False  # Use 'is' for boolean check


def test_ocr_initialization_with_device_cpu():
    """Test that OCRService initializes correctly with CPU device."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Mock the PaddleOCR constructor
        mock_instance = MagicMock()
        mock_paddleocr.return_value = mock_instance

        # Create settings with CPU device
        settings = Settings(OCR_DEVICE="cpu", _env_file=None)

        service = OCRService(settings=settings)

        # Verify PaddleOCR was called with device="cpu"
        mock_paddleocr.assert_called_once()
        call_args = mock_paddleocr.call_args
        assert 'device' in call_args.kwargs
        assert call_args.kwargs['device'] == "cpu"
        # Verify the service has the mocked instance using 'is'
        assert service.ocr is mock_instance


def test_ocr_initialization_with_device_gpu():
    """Test that OCRService initializes correctly with GPU device when available."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Need to also patch the hardware detection for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            with patch('paddle.device.cuda.device_count') as mock_device_count:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1

                # Mock the PaddleOCR constructor
                mock_instance = MagicMock()
                mock_paddleocr.return_value = mock_instance

                # Create settings with GPU device
                settings = Settings(OCR_DEVICE="gpu", _env_file=None)

                service = OCRService(settings=settings)

                # Verify PaddleOCR was called with device="gpu:0"
                mock_paddleocr.assert_called_once()
                call_args = mock_paddleocr.call_args
                assert 'device' in call_args.kwargs
                assert call_args.kwargs['device'] == "gpu:0"
                # Verify the service has the mocked instance using 'is'
                assert service.ocr is mock_instance


def test_ocr_initialization_with_device_auto_gpu():
    """Test that OCRService initializes with GPU in AUTO mode when available."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Need to also patch the hardware detection for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            with patch('paddle.device.cuda.device_count') as mock_device_count:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1

                # Mock the PaddleOCR constructor
                mock_instance = MagicMock()
                mock_paddleocr.return_value = mock_instance

                # Create settings with AUTO device
                settings = Settings(OCR_DEVICE="auto", _env_file=None)

                service = OCRService(settings=settings)

                # Verify PaddleOCR was called with device="gpu:0"
                mock_paddleocr.assert_called_once()
                call_args = mock_paddleocr.call_args
                assert 'device' in call_args.kwargs
                assert call_args.kwargs['device'] == "gpu:0"
                # Verify the service has the mocked instance using 'is'
                assert service.ocr is mock_instance


def test_ocr_initialization_with_device_auto_cpu():
    """Test that OCRService initializes with CPU in AUTO mode when GPU unavailable."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Need to also patch the hardware detection for this test
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            mock_is_compiled.return_value = False

            # Mock the PaddleOCR constructor
            mock_instance = MagicMock()
            mock_paddleocr.return_value = mock_instance

            # Create settings with AUTO device
            settings = Settings(OCR_DEVICE="auto", _env_file=None)

            service = OCRService(settings=settings)

            # Verify PaddleOCR was called with device="cpu"
            mock_paddleocr.assert_called_once()
            call_args = mock_paddleocr.call_args
            assert 'device' in call_args.kwargs
            assert call_args.kwargs['device'] == "cpu"
            # Verify the service has the mocked instance using 'is'
            assert service.ocr is mock_instance


def test_ocr_initialization_failure_gpu_mode():
    """Test that OCRService initialization failures in GPU mode propagate without CPU retry."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Need to also patch the hardware detection to simulate GPU available
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            with patch('paddle.device.cuda.device_count') as mock_device_count:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1

                # Make PaddleOCR constructor raise an exception
                mock_paddleocr.side_effect = RuntimeError("PaddleOCR failed to initialize")

                # Create settings with GPU device
                settings = Settings(OCR_DEVICE="gpu", _env_file=None)

                # Verify the exception propagates and constructor was called once
                try:
                    service = OCRService(settings=settings)
                    assert False, "Should have raised RuntimeError during initialization"
                except RuntimeError as e:
                    assert "PaddleOCR failed to initialize" in str(e)
                    # Verify constructor was called exactly once with gpu:0
                    mock_paddleocr.assert_called_once()
                    call_args = mock_paddleocr.call_args
                    assert 'device' in call_args.kwargs
                    assert call_args.kwargs['device'] == "gpu:0"


def test_ocr_initialization_failure_auto_with_gpu():
    """Test that OCRService initialization failures in Auto-with-GPU mode propagate without CPU retry."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Need to also patch the hardware detection to simulate GPU available
        with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
            with patch('paddle.device.cuda.device_count') as mock_device_count:
                mock_is_compiled.return_value = True
                mock_device_count.return_value = 1

                # Make PaddleOCR constructor raise an exception
                mock_paddleocr.side_effect = RuntimeError("PaddleOCR failed to initialize")

                # Create settings with AUTO device (should try GPU first)
                settings = Settings(OCR_DEVICE="auto", _env_file=None)

                # Verify the exception propagates and constructor was called once
                try:
                    service = OCRService(settings=settings)
                    assert False, "Should have raised RuntimeError during initialization"
                except RuntimeError as e:
                    assert "PaddleOCR failed to initialize" in str(e)
                    # Verify constructor was called exactly once with gpu:0
                    mock_paddleocr.assert_called_once()
                    call_args = mock_paddleocr.call_args
                    assert 'device' in call_args.kwargs
                    assert call_args.kwargs['device'] == "gpu:0"


def test_ocr_initialization_failure_propagates():
    """Test that OCRService initialization failures propagate without CPU fallback."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Make PaddleOCR constructor raise an exception
        mock_paddleocr.side_effect = RuntimeError("PaddleOCR failed to initialize")

        # Create settings with any device
        settings = Settings(OCR_DEVICE="cpu", _env_file=None)

        # Verify the exception propagates
        try:
            service = OCRService(settings=settings)
            assert False, "Should have raised RuntimeError during initialization"
        except RuntimeError as e:
            assert "PaddleOCR failed to initialize" in str(e)
            # Verify constructor was called exactly once
            mock_paddleocr.assert_called_once()


@pytest.mark.parametrize("device_setting,expected_device,mock_gpu", [
    ("cpu", "cpu", False),
    ("gpu", "gpu:0", True),
    ("auto", "gpu:0", True),
])
def test_ocr_predict_failure_propagates(device_setting, expected_device, mock_gpu):
    """Test that OCR inference exceptions propagate without reconstruction or fallback."""
    # Set up mocks based on device setting
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Mock the PaddleOCR constructor
        mock_instance = MagicMock()
        mock_paddleocr.return_value = mock_instance

        # Make the predict method raise an exception
        mock_instance.predict.side_effect = RuntimeError("Prediction failed")

        # Set up GPU mocking if needed
        if mock_gpu:
            with patch('paddle.is_compiled_with_cuda') as mock_is_compiled:
                with patch('paddle.device.cuda.device_count') as mock_device_count:
                    mock_is_compiled.return_value = True
                    mock_device_count.return_value = 1

                    # Create settings with the device setting
                    settings = Settings(OCR_DEVICE=device_setting, _env_file=None)

                    service = OCRService(settings=settings)

                    # Verify the service was initialized correctly with expected device
                    assert service.ocr is mock_instance
                    # Verify PaddleOCR was called with correct device
                    mock_paddleocr.assert_called_once()
                    call_args = mock_paddleocr.call_args
                    assert 'device' in call_args.kwargs
                    assert call_args.kwargs['device'] == expected_device

                    # Verify predict exception propagates
                    try:
                        # Call extract_text_from_image with a dummy image to trigger prediction
                        import numpy as np
                        dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
                        service.extract_text_from_image(dummy_image)
                        assert False, "Should have raised RuntimeError during prediction"
                    except RuntimeError as e:
                        assert "Prediction failed" in str(e)
                        # Verify predict was called
                        mock_instance.predict.assert_called_once()
                        # Verify the same OCR instance remains assigned
                        assert service.ocr is mock_instance
                        # Verify no reconstruction occurred (PaddleOCR not called again)
                        mock_paddleocr.assert_called_once()
        else:
            # CPU device test (no GPU mocking needed)
            # Create settings with the device setting
            settings = Settings(OCR_DEVICE=device_setting, _env_file=None)

            service = OCRService(settings=settings)

            # Verify the service was initialized correctly
            assert service.ocr is mock_instance
            # Verify PaddleOCR was called with correct device
            mock_paddleocr.assert_called_once()
            call_args = mock_paddleocr.call_args
            assert 'device' in call_args.kwargs
            assert call_args.kwargs['device'] == expected_device

            # Verify predict exception propagates
            try:
                # Call extract_text_from_image with a dummy image to trigger prediction
                import numpy as np
                dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
                service.extract_text_from_image(dummy_image)
                assert False, "Should have raised RuntimeError during prediction"
            except RuntimeError as e:
                assert "Prediction failed" in str(e)
                # Verify predict was called
                mock_instance.predict.assert_called_once()
                # Verify the same OCR instance remains assigned
                assert service.ocr is mock_instance
                # Verify no reconstruction occurred (PaddleOCR not called again)
                mock_paddleocr.assert_called_once()


def test_ocr_settings_isolation():
    """Test that OCR service respects custom settings values."""
    # Explicit constructor mocking for every service construction
    with patch('app.ocr.service.PaddleOCR') as mock_paddleocr:
        # Mock the PaddleOCR constructor
        mock_instance = MagicMock()
        mock_paddleocr.return_value = mock_instance

        # Create a mock settings object with specific values
        settings = MagicMock()
        settings.OCR_DEVICE = "cpu"
        settings.OCR_USE_DOC_ORIENTATION_CLASSIFY = False
        settings.OCR_USE_DOC_UNWARPING = False
        settings.OCR_USE_TEXTLINE_ORIENTATION = False
        settings.OCR_ENABLE_MKLDNN = False
        settings.OCR_LANG = "en"

        service = OCRService(settings=settings)

        # Verify PaddleOCR was called with correct parameters
        mock_paddleocr.assert_called_once()
        call_args = mock_paddleocr.call_args
        kwargs = call_args.kwargs
        assert kwargs['device'] == 'cpu'
        assert kwargs['lang'] == 'en'
        assert kwargs['use_doc_orientation_classify'] == False
        assert kwargs['use_doc_unwarping'] == False
        assert kwargs['use_textline_orientation'] == False
        assert kwargs['enable_mkldnn'] == False
        # Verify the service has the mocked instance using 'is'
        assert service.ocr is mock_instance


if __name__ == "__main__":
    # Run the tests
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"],
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    sys.exit(result.returncode)

def teardown_module(module):
    _import_temp_dir.cleanup()