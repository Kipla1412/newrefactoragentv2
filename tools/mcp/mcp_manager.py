import asyncio
from typing import Any
from config.config import Config
from tools.mcp.client import MCPClient, MCPServerStatus
from tools.mcp.mcp_tool import MCPTool
from tools.registry import ToolRegistry
from utils.logging_config import get_logger

logger = get_logger(__name__)


class MCPManager:
    def __init__(self, config: Config):
        self.config = config
        self._clients: dict[str, MCPClient] = {}
        self._initialized = False
        self._auth_token: str | None = None
        self._lock = asyncio.Lock()

    async def initialize(self, auth_token: str | None = None) -> None:
        async with self._lock:
            # Skip initialization if already active with the identical auth token
            if self._initialized and self._auth_token == auth_token:
                return

            mcp_configs = self.config.mcp_servers_config

            if self._initialized:
                await self._shutdown_unlocked()

            if not mcp_configs:
                self._auth_token = auth_token
                self._initialized = True
                return

            for name, server_config in mcp_configs.items():
                if not server_config.enabled:
                    continue

                headers = dict(server_config.headers or {})

                # Inject dynamic Bearer token
                if auth_token:
                    headers["Authorization"] = f"Bearer {auth_token}"

                runtime_config = server_config.model_copy(
                    update={"headers": headers}
                )

                self._clients[name] = MCPClient(
                    name=name,
                    config=runtime_config,
                    cwd=self.config.cwd,
                )

            connection_tasks = [
                asyncio.wait_for(
                    client.connect(),
                    timeout=client.config.startup_timeout_sec,
                )
                for client in self._clients.values()
            ]

            results = await asyncio.gather(
                *connection_tasks,
                return_exceptions=True
            )

            for result in results:
                if isinstance(result, Exception):
                    logger.warning("MCP connection error: %s", result)

            self._auth_token = auth_token
            self._initialized = True

    def register_tools(self, registry: ToolRegistry) -> int:
        count = 0

        # Drop any previously-registered MCP tools so re-initializing with a
        # fresh auth token never leaves stale tools behind in the registry.
        registry.clear_mcp_tools()

        for client in self._clients.values():
            if client.status != MCPServerStatus.CONNECTED:
                continue

            for tool_info in client.tools:
                mcp_tool = MCPTool(
                    tool_info=tool_info,
                    client=client,
                    config=self.config,
                    name=f"{client.name}__{tool_info.name}",
                )
                registry.register_mcp(mcp_tool)
                count += 1

        return count

    async def shutdown(self) -> None:
        async with self._lock:
            await self._shutdown_unlocked()

    async def _shutdown_unlocked(self) -> None:
        disconnection_tasks = [
            client.disconnect() for client in self._clients.values()
        ]

        await asyncio.gather(*disconnection_tasks, return_exceptions=True)

        self._clients.clear()
        self._auth_token = None
        self._initialized = False

    def get_all_servers(self) -> list[dict[str, Any]]:
        return [
            {
                "name": name,
                "status": client.status.value,
                "tools": len(client.tools),
            }
            for name, client in self._clients.items()
        ]