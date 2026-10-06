from __future__ import annotations

import glob as _glob
import os
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult

MAX_CONTENT_CHARS = 8000


class WikiSearchParams(BaseModel):
    query: str = Field(
        ...,
        description="Search term — patient name, disease, medication, doctor, or procedure",
    )
    category: str = Field(
        "Patients",
        description="Wiki category to scope the search: Patients, Diseases, Medications, Doctors, Procedures, or Reports",
    )


class WikiTool(Tool):
    name = "wiki_search"
    description = (
        "Search the medical wiki for patient records, diseases, medications, "
        "doctors, procedures, and clinical reports. Matches directory/file names "
        "case-insensitively against the query and returns relevant markdown content."
    )
    kind = ToolKind.READ
    schema = WikiSearchParams

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._root = Path(config.wiki_root_path)

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = WikiSearchParams(**invocation.params)
        category = params.category.strip()
        query = params.query.strip().lower()

        search_dir = self._root / category
        if not search_dir.is_dir():
            available = [d.name for d in self._root.iterdir() if d.is_dir()]
            return ToolResult.error_result(
                f"Category '{category}' not found. Available: {', '.join(available)}"
            )

        matches: list[Path] = []

        # Phase 1: broad glob, then filter
        candidate_dirs: list[Path] = []
        _glob_pattern = os.path.join(str(search_dir), "*")
        for path_str in _glob.iglob(_glob_pattern, recursive=False):
            p = Path(path_str)
            if not _matches_query(p, query):
                continue
            if p.is_dir():
                candidate_dirs.append(p)
            elif p.suffix == ".md":
                matches.append(p)

        # Phase 2: for directory matches, recurse into .md files
        if candidate_dirs and not matches:
            for d in candidate_dirs:
                for md in sorted(d.rglob("*.md")):
                    matches.append(md)

        if not matches:
            return ToolResult.success_result(
                f"No wiki results found for '{query}' in category '{category}'.",
                metadata={"query": query, "category": category, "found": False},
            )

        parts: list[str] = []
        sources: list[str] = []
        total_chars = 0

        for match in matches:
            header = f"--- {match.relative_to(self._root)} ---"
            try:
                content = match.read_text(encoding="utf-8")
            except Exception:
                continue

            remaining = MAX_CONTENT_CHARS - total_chars
            if remaining <= 0:
                break

            chunk = content[:remaining]
            parts.append(f"{header}\n{chunk}")
            sources.append(str(match.relative_to(self._root)))
            total_chars += len(chunk)

        return ToolResult.success_result(
            "\n\n".join(parts),
            metadata={
                "query": query,
                "category": category,
                "found": True,
                "match_count": len(matches),
                "sources": sources[:20],
            },
        )


def _matches_query(path: Path, query: str) -> bool:
    name_lower = path.name.lower()
    if query in name_lower:
        return True
    name_no_ext = path.stem.lower().replace("-", " ").replace("_", " ")
    if query in name_no_ext:
        return True
    return False
