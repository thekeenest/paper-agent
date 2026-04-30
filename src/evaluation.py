# Re-export shim — delegates to the frozen v1 implementation.
# Do not add logic here; v2 evaluation lives in src/v2/eval/.
from src.v1.evaluation import *  # noqa: F401, F403
