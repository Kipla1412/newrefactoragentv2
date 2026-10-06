from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from config.config import Config
from tools.base import Tool, ToolInvocation, ToolKind, ToolResult

MAX_RESULT_ROWS = 50


class GraphQueryParams(BaseModel):
    entity: str = Field(
        ...,
        description="Entity name to query — patient, disease, medication, doctor, or hospital name",
    )
    relation: str = Field(
        "",
        description="Optional. Filter by relationship type: HAS_DISEASE, TAKES_MEDICATION, TREATED_BY, ADMITTED_AT, WORKS_AT, HAS_REPORT. Leave empty to get all relationships.",
    )


class Neo4jTool(Tool):
    name = "neo4j_query"
    description = (
        "Query the Neo4j medical knowledge graph for relationships between "
        "patients, diseases, medications, doctors, and hospitals. "
        "Returns entity→relation→entity triples. Use this to discover "
        "who treats whom, what conditions a patient has, which medications "
        "are prescribed, and where patients are admitted."
    )
    kind = ToolKind.READ
    schema = GraphQueryParams

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self._uri = config.neo4j_uri
        self._user = config.neo4j_user
        self._password = config.neo4j_password
        self._driver = None

    def _get_driver(self):
        if self._driver is None:
            from neo4j import GraphDatabase
            self._driver = GraphDatabase.driver(
                self._uri,
                auth=(self._user, self._password),
            )
        return self._driver

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = GraphQueryParams(**invocation.params)
        entity = params.entity.strip()
        relation = params.relation.strip()

        if not entity:
            return ToolResult.error_result("Empty entity name")

        try:
            driver = self._get_driver()
            with driver.session() as session:
                query = _build_cypher(entity, relation)
                pattern = entity.lower()
                result = session.run(query, pattern=pattern)
                records = list(result)

            if not records:
                return ToolResult.success_result(
                    f"No graph entities or relationships found for '{entity}'.",
                    metadata={"entity": entity, "found": False},
                )

            lines = [f"Graph results for '{entity}':", ""]
            for rec in records[:MAX_RESULT_ROWS]:
                src = rec.get("source", "?")
                rel = rec.get("relation", "?")
                tgt = rec.get("target", "?")
                lines.append(f"{src} —[{rel}]→ {tgt}")

            count = len(records)
            if count > MAX_RESULT_ROWS:
                lines.append(f"\n... (showing {MAX_RESULT_ROWS} of {count}, truncated)")

            lines.append(f"\n({count} relationship(s))")

            return ToolResult.success_result(
                "\n".join(lines),
                metadata={
                    "entity": entity,
                    "found": True,
                    "relationships": count,
                },
            )

        except Exception as exc:
            return ToolResult.error_result(
                f"Neo4j query failed: {exc}",
                metadata={"entity": entity},
            )


def _build_cypher(entity: str, relation: str) -> str:
    base = (
        "MATCH (a)-[r{rel}]->(b) "
        "WHERE toLower(a.label) CONTAINS $pattern "
        "OR toLower(b.label) CONTAINS $pattern "
        "RETURN a.label AS source, type(r) AS relation, b.label AS target"
    )
    if relation:
        return base.replace("{rel}", f":{relation.upper()}")
    return base.replace("{rel}", "")
