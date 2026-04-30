# Re-export shim — delegates to the frozen v1 implementation.
# Do not add logic here; new Pydantic models live in src/v2/.
from src.v1.models import *  # noqa: F401, F403
