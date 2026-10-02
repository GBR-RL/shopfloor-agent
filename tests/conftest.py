import shutil
from pathlib import Path

import pytest

DATA = Path(__file__).resolve().parents[1] / "data"
PLANT_DB = DATA / "plant.db"

needs_data = pytest.mark.skipif(not PLANT_DB.exists(), reason="run `shopfloor data` first")


@pytest.fixture
def plant_db(tmp_path: Path) -> Path:
    """A private copy of the plant database, so write tools cannot affect other tests."""
    if not PLANT_DB.exists():
        pytest.skip("run `shopfloor data` first")
    copy = tmp_path / "plant.db"
    shutil.copyfile(PLANT_DB, copy)
    return copy


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"
