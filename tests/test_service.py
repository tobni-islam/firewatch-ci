from pathlib import Path


def test_service_file_exists():
    assert Path("service/service.py").exists(), "service/service.py not found"


def test_bentofile_exists():
    assert Path("service/bentofile.yaml").exists(), "service/bentofile.yaml not found"
