"""
Structural sanity tests — verify the repo scaffold is correct.

These tests do NOT call any LLM, make any network request, or touch the
file-system beyond reading already-present files.  They are the minimal
acceptance gate for the chore(scaffold) commit.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent


def _import(module: str) -> object:
    """Import a module, raising AssertionError with a helpful message on failure."""
    try:
        return importlib.import_module(module)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(f"Failed to import {module!r}: {exc}") from exc


# ---------------------------------------------------------------------------
# 1. v1 package integrity
# ---------------------------------------------------------------------------


class TestV1Imports:
    """v1 sub-modules must import cleanly from src.v1.*."""

    def test_v1_models(self) -> None:
        mod = _import("src.v1.models")
        assert hasattr(mod, "PaperMetadata")
        assert hasattr(mod, "AuthorAffiliation")

    def test_v1_state(self) -> None:
        mod = _import("src.v1.state")
        assert hasattr(mod, "AgentState")
        assert hasattr(mod, "create_initial_state")

    def test_v1_knowledge_base(self) -> None:
        _import("src.v1.knowledge_base")

    def test_v1_analytics(self) -> None:
        mod = _import("src.v1.analytics")
        assert hasattr(mod, "AnalyticsEngine")

    def test_v1_evaluation(self) -> None:
        _import("src.v1.evaluation")

    def test_v1_data_sources_package(self) -> None:
        mod = _import("src.v1.data_sources")
        assert hasattr(mod, "DataSourceType")
        assert hasattr(mod, "ArxivClient")
        assert hasattr(mod, "DataSourceRouter")


# ---------------------------------------------------------------------------
# 2. Re-export shims at src.* level
# ---------------------------------------------------------------------------


class TestShimReExports:
    """Legacy src.* imports must resolve through the shims without error."""

    def test_src_models_shim(self) -> None:
        mod = _import("src.models")
        assert hasattr(mod, "PaperMetadata")

    def test_src_state_shim(self) -> None:
        mod = _import("src.state")
        assert hasattr(mod, "create_initial_state")

    def test_src_analytics_shim(self) -> None:
        mod = _import("src.analytics")
        assert hasattr(mod, "AnalyticsEngine")

    def test_src_knowledge_base_shim(self) -> None:
        _import("src.knowledge_base")

    def test_src_evaluation_shim(self) -> None:
        _import("src.evaluation")

    def test_src_data_sources_shim(self) -> None:
        mod = _import("src.data_sources")
        assert hasattr(mod, "DataSourceRouter")
        assert hasattr(mod, "ArxivClient")


# ---------------------------------------------------------------------------
# 3. v2 skeleton packages importable
# ---------------------------------------------------------------------------


class TestV2Skeleton:
    """Every v2 stub package must be importable (no syntax errors, no missing deps)."""

    V2_PACKAGES = [
        "src.v2",
        "src.v2.orchestration",
        "src.v2.agents",
        "src.v2.agents.extractors",
        "src.v2.agents.critic",
        "src.v2.parsers",
        "src.v2.linkers",
        "src.v2.kg",
        "src.v2.analytics",
        "src.v2.eval",
    ]

    def test_all_v2_packages_importable(self) -> None:
        for pkg in self.V2_PACKAGES:
            _import(pkg)

    def test_v2_version(self) -> None:
        mod = _import("src.v2")
        assert hasattr(mod, "__version__")
        assert mod.__version__.startswith("2.")


# ---------------------------------------------------------------------------
# 4. Directory structure on disk
# ---------------------------------------------------------------------------


class TestDirectoryLayout:
    """Key directories and files must exist on disk."""

    REQUIRED_DIRS = [
        "src/v1",
        "src/v1/api",
        "src/v1/data_sources",
        "src/v2",
        "src/v2/orchestration",
        "src/v2/agents/extractors",
        "src/v2/agents/critic",
        "src/v2/parsers",
        "src/v2/linkers",
        "src/v2/kg",
        "src/v2/analytics",
        "src/v2/eval",
        "benchmark/PaperAffilBench/papers",
        "benchmark/PaperAffilBench/gold",
        "benchmark/PaperAffilBench/splits",
        "experiments",
        "docs",
        "tests",
    ]

    REQUIRED_FILES = [
        "src/v1/__init__.py",
        "src/v1/main.py",
        "src/v1/state.py",
        "src/v1/models.py",
        "src/v1/graph.py",
        "src/v1/nodes.py",
        "src/v1/normalizer.py",
        "src/v1/knowledge_base.py",
        "src/v1/analytics.py",
        "src/v1/evaluation.py",
        "src/v2/__init__.py",
        "src/analytics/__init__.py",
        "pyproject.toml",
        "docs/architecture.md",
        ".github/workflows/ci.yml",
    ]

    def test_required_dirs_exist(self) -> None:
        missing = [d for d in self.REQUIRED_DIRS if not (ROOT / d).is_dir()]
        assert not missing, f"Missing directories: {missing}"

    def test_required_files_exist(self) -> None:
        missing = [f for f in self.REQUIRED_FILES if not (ROOT / f).is_file()]
        assert not missing, f"Missing files: {missing}"

    def test_dev_plan_referenced_from_architecture_md(self) -> None:
        arch = (ROOT / "docs" / "architecture.md").read_text()
        assert "DEV_PLAN.md" in arch, "docs/architecture.md must reference DEV_PLAN.md"

    def test_v1_source_files_frozen(self) -> None:
        """v1 source files must NOT contain 'from src.v2' — they are frozen."""
        v1_dir = ROOT / "src" / "v1"
        violations = []
        for py_file in v1_dir.rglob("*.py"):
            text = py_file.read_text()
            if "from src.v2" in text or "import src.v2" in text:
                violations.append(str(py_file))
        assert not violations, f"v1 files must not import src.v2: {violations}"
