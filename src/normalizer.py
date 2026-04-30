# Re-export shim — delegates to the frozen v1 implementation.
# Do not add logic here; new linker/normalizer code lives in src/v2/linkers/.
from src.v1.normalizer import *  # noqa: F401, F403
