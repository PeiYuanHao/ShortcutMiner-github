"""Regression guard against benchmark dispatch in production runtime."""

import ast
from pathlib import Path


RUNTIME_ROOTS = [
    Path("shortcutminer/workflow"),
    Path("shortcutminer/runtime"),
    Path("shortcutminer/experiment"),
    Path("shortcutminer/tooling"),
]
BENCHMARK_NAMES = {"waterbirds", "colored_mnist", "synthetic_patch"}


def _strings(node: ast.AST) -> set[str]:
    return {
        item.value.casefold()
        for item in ast.walk(node)
        if isinstance(item, ast.Constant) and isinstance(item.value, str)
    }


def test_production_has_no_benchmark_specific_dispatch_conditionals():
    violations = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, (ast.If, ast.Match, ast.IfExp)):
                    hits = _strings(node) & BENCHMARK_NAMES
                    if hits:
                        violations.append((str(path), node.lineno, sorted(hits)))
    assert violations == []


def test_production_has_no_benchmark_adapter_registry():
    violations = []
    for root in RUNTIME_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Dict):
                    hits = _strings(node) & BENCHMARK_NAMES
                    if hits:
                        violations.append((str(path), node.lineno, sorted(hits)))
    assert violations == []


def test_agent_prompts_do_not_name_benchmark_datasets():
    violations = []
    for path in Path("agents").glob("*/workspace/*.md"):
        words = path.read_text(encoding="utf-8").casefold()
        hits = sorted(name for name in BENCHMARK_NAMES if name in words)
        if hits:
            violations.append((str(path), hits))
    assert violations == []
