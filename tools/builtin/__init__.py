
from tools.builtin.memory import MemoryTool
from tools.builtin.wiki import WikiTool
from tools.builtin.neo4j import Neo4jTool

__all__ = [
    "MemoryTool",
    "WikiTool",
    "Neo4jTool",
]


def get_all_builtin_tools() -> list[type]:
    return [
        MemoryTool,
        WikiTool,
        Neo4jTool,
    ]