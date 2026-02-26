"""conftest.py — pytest configuration and shared fixtures."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "db: marks tests requiring a live PostgreSQL connection")



def pytest_addoption(parser):
    parser.addoption(
        "--run-db",
        action="store_true",
        default=False,
        help="Run integration tests that require a live PostgreSQL connection",
    )


@pytest.fixture
def db_conn(request):
    if not request.config.getoption("--run-db"):
        pytest.skip("Pass --run-db to run DB integration tests")
    from scripts.ingest_utils import get_conn
    conn = get_conn()
    yield conn
    conn.close()
