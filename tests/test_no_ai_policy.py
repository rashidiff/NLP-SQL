from __future__ import annotations

import ast
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_PACKAGES = {
    "anthropic",
    "chromadb",
    "cohere",
    "faiss",
    "google-generativeai",
    "langchain",
    "llama-index",
    "openai",
    "pinecone",
    "sentence-transformers",
    "sklearn",
    "tensorflow",
    "torch",
    "transformers",
    "weaviate-client",
}

FORBIDDEN_IMPORT_ROOTS = {package.replace("-", "_") for package in FORBIDDEN_PACKAGES} | {
    "anthropic",
    "chromadb",
    "cohere",
    "faiss",
    "langchain",
    "llama_index",
    "openai",
    "pinecone",
    "sentence_transformers",
    "sklearn",
    "tensorflow",
    "torch",
    "transformers",
    "weaviate",
}


def test_project_dependencies_do_not_include_ai_or_ml_packages() -> None:
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))

    dependencies = set(pyproject["project"].get("dependencies", ()))
    optional_groups = pyproject["project"].get("optional-dependencies", {})
    for group_dependencies in optional_groups.values():
        dependencies.update(group_dependencies)

    normalized = {_package_name(dependency) for dependency in dependencies}

    assert normalized.isdisjoint(FORBIDDEN_PACKAGES)


def test_source_does_not_import_ai_or_ml_packages() -> None:
    offenders: list[str] = []
    for path in (ROOT / "src").rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            imported = _import_root(node)
            if imported in FORBIDDEN_IMPORT_ROOTS:
                offenders.append(f"{path.relative_to(ROOT)} imports {imported}")

    assert offenders == []


def _package_name(dependency: str) -> str:
    return dependency.split("[", 1)[0].split(">", 1)[0].split("<", 1)[0].split("=", 1)[0].strip()


def _import_root(node: ast.AST) -> str | None:
    if isinstance(node, ast.Import):
        return node.names[0].name.split(".", 1)[0]
    if isinstance(node, ast.ImportFrom) and node.module:
        return node.module.split(".", 1)[0]
    return None
