# Re-export shim — delegates to the frozen v1 implementation.
# Do not add logic here; v2 uses ROR API + KuzuDB (src/v2/kg/, src/v2/linkers/).
from src.v1.knowledge_base import *  # noqa: F401, F403
