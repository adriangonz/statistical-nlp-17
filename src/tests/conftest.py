import os
import pytest

FIXTURES_PATH = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def wiki_file_path():
    return os.path.join(FIXTURES_PATH, "wiki.fixture.tokens")
