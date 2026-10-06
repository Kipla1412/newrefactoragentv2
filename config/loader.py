from pathlib import Path
from typing import Any

from platformdirs import user_config_dir, user_data_dir
import tomli
import yaml
from dotenv import load_dotenv

from config.config import Config
from utils.errors import ConfigError
import logging

logger = logging.getLogger(__name__)
CONFIG_FILE_NAME = "config.toml"
AGENTS_YAML_FILE_NAME = "agents.yaml"

AGENT_MD_FILE = "AGENT.MD"


def get_config_dir() -> Path:
    return Path(user_config_dir("ai-agent"))


def get_data_dir() -> Path:
    return Path(user_data_dir("ai-agent"))


def get_system_config_path() -> Path:
    return get_config_dir() / CONFIG_FILE_NAME


def get_agents_yaml_path(cwd: Path | None = None) -> Path:
    """Return the path to the agents.yaml configuration file.

    Args:
        cwd: Project root to resolve ``config/agents.yaml`` from. Defaults
            to the project root derived from this module's location.

    Returns:
        The absolute path to ``config/agents.yaml``.
    """
    project_root = cwd or Path(__file__).resolve().parent.parent
    return project_root / "config" / AGENTS_YAML_FILE_NAME


def _parse_toml(path: Path):
    try:
        with open(path, "rb") as f:
            return tomli.load(f)
    except tomli.TOMLDecodeError as e:
        raise ConfigError("Invalid TOML in {path}: {e}", config_file=str(path)) from e
    except (OSError, IOError) as e:
        raise ConfigError(
            "Failed to read config file {path}: {e}", config_file=str(path)
        ) from e


def _parse_yaml(path: Path) -> dict[str, Any]:
    """Parse a YAML configuration file into a dict.

    Args:
        path: Path to the YAML file.

    Returns:
        The parsed YAML content as a mapping.

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = yaml.safe_load(f)
        return content if isinstance(content, dict) else {}
    except yaml.YAMLError as e:
        raise ConfigError(
            f"Invalid YAML in {path}: {e}", config_file=str(path)
        ) from e
    except (OSError, IOError) as e:
        raise ConfigError(
            f"Failed to read config file {path}: {e}", config_file=str(path)
        ) from e


def _get_project_config(cwd: Path) -> Path | None:
    current = cwd.resolve()
    agent_dir = current / ".ai-agent"

    if agent_dir.is_dir():
        config_file = agent_dir / CONFIG_FILE_NAME
        if config_file.is_file():
            return config_file

    return None


def _get_agent_md_files(cwd: Path) -> Path | None:
    current = cwd.resolve()

    if current.is_dir():
        agent_md_file = current / AGENT_MD_FILE
        if agent_md_file.is_file():
            content = agent_md_file.read_text(encoding="utf-8")
            return content

    return None


def _merge_dicts(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _merge_dicts(result[key], value)
        else:
            result[key] = value

    return result


def load_config(cwd: Path | None) -> Config:
    cwd = cwd or Path.cwd()

    # Load environment variables from .env file in the project root
    load_dotenv(dotenv_path=cwd.parent / ".env")

    system_path = get_system_config_path()

    config_dict: dict[str, Any] = {}

    if system_path.is_file():
        try:
            config_dict = _parse_toml(system_path)
        except ConfigError:
            logger.warning(f"Skipping invalid system config: {system_path}")

    project_path = _get_project_config(cwd)
    if project_path:
        try:
            project_config_dict = _parse_toml(project_path)
            config_dict = _merge_dicts(config_dict, project_config_dict)
        except ConfigError:
            logger.warning(f"Skipping invalid system config: {system_path}")

    if "cwd" not in config_dict:
        config_dict["cwd"] = cwd

    if "developer_instructions" not in config_dict:
        agent_md_content = _get_agent_md_files(cwd)
        if agent_md_content:
            config_dict["developer_instructions"] = agent_md_content

    # Load the YAML agent configuration (models/agents) once.
    agents_yaml_path = get_agents_yaml_path(cwd=cwd)
    if agents_yaml_path.is_file():
        try:
            agents_config = _parse_yaml(agents_yaml_path)
            models_section = {"definitions": agents_config.get("models", {})}
            agents_section = {"definitions": agents_config.get("agents", {})}
            config_dict = _merge_dicts(config_dict, {
                "models": models_section,
                "agents": agents_section,
            })
        except ConfigError as e:
            logger.warning("Skipping invalid agents YAML: %s", e)

    try:
        config = Config(**config_dict)
    except Exception as e:
        raise ConfigError(f"Invalid configuration: {e}") from e

    return config
