import pytest
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(1, str(project_root / 'scripts'))

from scripts.process_rsv_data import main as rsv_main


@pytest.fixture
def mock_preprocessor(monkeypatch):
    """Mocks the RSVPreprocessor class."""
    
    call_args = {}
    class MockRSVPreprocessor:
        def __init__(self, base_path, output_path, demo_mode=False):
            call_args['base_path'] = base_path
            call_args['output_path'] = output_path
            call_args['demo_mode'] = demo_mode
            self.create_payloads_called = False

        def create_visualization_payloads(self):
            self.create_payloads_called = True
    
    monkeypatch.setattr("scripts.process_rsv_data.RSVPreprocessor", MockRSVPreprocessor)
    yield call_args, MockRSVPreprocessor


def test_main_default_args(monkeypatch, mock_preprocessor):
    """Tests main() with default command-line arguments."""
    call_args, _ = mock_preprocessor
    test_args = ["process_rsv_data.py"]
    monkeypatch.setattr(sys, "argv", test_args)

    monkeypatch.setattr("os.listdir", lambda: [])

    rsv_main()

    assert call_args['base_path'] == './rsv-forecast-hub'
    assert call_args['output_path'] == './processed_data'
    assert call_args['demo_mode'] is False

def test_main_custom_args(monkeypatch, mock_preprocessor):
    """Tests main() with custom command-line arguments."""
    call_args, _ = mock_preprocessor
    test_args = [
        "process_rsv_data.py",
        "--hub-path", "/test/rsv/hub",
        "--output-path", "/test/rsv/output",
        "--demo"
    ]
    monkeypatch.setattr(sys, "argv", test_args)
    
    monkeypatch.setattr("os.listdir", lambda: [])

    rsv_main()

    assert call_args['base_path'] == '/test/rsv/hub'
    assert call_args['output_path'] == '/test/rsv/output'
    assert call_args['demo_mode'] is True

def test_main_calls_create_payloads(monkeypatch, mock_preprocessor):
    """Verifies that main() calls the create_visualization_payloads method."""
    _, MockClass = mock_preprocessor
    
    original_init = MockClass.__init__
    def spy_init(self, *args, **kwargs):
        MockClass.instance = self
        original_init(self, *args, **kwargs)
    
    monkeypatch.setattr(MockClass, "__init__", spy_init)
    monkeypatch.setattr(sys, "argv", ["process_rsv_data.py"])
    
    monkeypatch.setattr("os.listdir", lambda: [])
    
    rsv_main()
    
    assert hasattr(MockClass, 'instance')
    assert MockClass.instance.create_payloads_called is True