import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.process_flusight_data import main as flusight_main


@pytest.fixture
def mock_preprocessor(monkeypatch):
    """Mocks the FluSightPreprocessor class."""
    
    call_args = {}

    class MockFluSightPreprocessor:
        def __init__(self, base_path: str, output_path: str, demo_mode: bool = False):
            call_args['base_path'] = base_path
            call_args['output_path'] = output_path
            call_args['demo_mode'] = demo_mode
            self.create_payloads_called = False

        def create_visualization_payloads(self):
            self.create_payloads_called = True
    
    monkeypatch.setattr("scripts.process_flusight_data.FluSightPreprocessor", MockFluSightPreprocessor)
    
    return call_args, MockFluSightPreprocessor


def test_main_default_args(monkeypatch, mock_preprocessor):
    """Tests main() with default command-line arguments."""
    call_args, _ = mock_preprocessor
    
    test_args = ["process_flusight_data.py"]
    monkeypatch.setattr(sys, "argv", test_args)

    flusight_main()

    assert call_args['base_path'] == './FluSight-forecast-hub'
    assert call_args['output_path'] == './processed_data'
    assert call_args['demo_mode'] is False

def test_main_custom_args(monkeypatch, mock_preprocessor):
    """Tests main() with custom command-line arguments."""
    call_args, _ = mock_preprocessor

    test_args = [
        "process_flusight_data.py",
        "--hub-path", "/custom/hub",
        "--output-path", "/custom/output",
        "--demo"
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    flusight_main()

    assert call_args['base_path'] == '/custom/hub'
    assert call_args['output_path'] == '/custom/output'
    assert call_args['demo_mode'] is True

def test_main_calls_create_payloads(monkeypatch, mock_preprocessor):
    """Verifies that main() calls the create_visualization_payloads method."""
    _, MockClass = mock_preprocessor
    
    original_init = MockClass.__init__
    def spy_init(self, *args, **kwargs):
        MockClass.instance = self
        original_init(self, *args, **kwargs)
    
    monkeypatch.setattr(MockClass, "__init__", spy_init)
    
    monkeypatch.setattr(sys, "argv", ["process_flusight_data.py"])
    
    flusight_main()
    
    assert hasattr(MockClass, 'instance')
    assert MockClass.instance.create_payloads_called is True

def test_main_exception_handling(monkeypatch, mock_preprocessor, caplog):
    """Tests that main() logs errors and re-raises them."""
    _, MockClass = mock_preprocessor

    def mock_create_payloads_fails(self):
        raise ValueError("Something went wrong")

    monkeypatch.setattr(MockClass, "create_visualization_payloads", mock_create_payloads_fails)
    monkeypatch.setattr(sys, "argv", ["process_flusight_data.py"])
    
    with pytest.raises(ValueError, match="Something went wrong"):
        flusight_main()

    assert "Failed to run preprocessing: Something went wrong" in caplog.text