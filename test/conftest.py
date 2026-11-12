import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure the project root is importable regardless of how pytest is invoked.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Keep any pre-existing environment from leaking into tests.
for key in (
    "FHIR_BASE_URL",
    "MLFLOW_ENABLED",
    "IAM_JWKS_URL",
    "IAM_ISSUER",
):
    os.environ.pop(key, None)

# Unit tests must not depend on an MLflow server. This is *set*, not left unset,
# because `load_dotenv` would otherwise turn tracking back on from `.env` — and
# every agent turn now creates a run, so the suite would spend minutes on
# network round-trips. Tests that need MLflow point it at their own backend
# explicitly (see test_mlflow_correlation.py).
os.environ["MLFLOW_ENABLED"] = "false"

# Agent loops create spans directly, independently of the config above, so give
# MLflow a local file backend: no server, no export retries.
os.environ["MLFLOW_TRACKING_URI"] = (
    f"sqlite:///{Path(tempfile.gettempdir()) / 'pytest-mlflow.db'}"
)


@pytest.fixture
def config():
    from config.config import Config

    return Config(cwd=Path.cwd())
