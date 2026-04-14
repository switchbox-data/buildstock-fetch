from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def vcr_cassette_dir(request: pytest.FixtureRequest) -> str:
    module_name = Path(str(request.node.path)).stem
    cassette_dir = Path(__file__).resolve().parent.parent / "cassettes" / module_name
    return str(cassette_dir)
