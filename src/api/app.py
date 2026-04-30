# Re-export shim — exposes the v1 FastAPI `app` instance at src.api.app:app
# so that `uvicorn src.api.app:app` and run_server.py continue to work.
from src.v1.api.app import *  # noqa: F401, F403
from src.v1.api.app import app  # noqa: F401
