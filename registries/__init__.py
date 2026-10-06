"""Registries package for the agent framework.

Contains the ModelRegistry and PromptRegistry that resolve
logical model and prompt names to their concrete configurations
and content, respectively.
"""
from registries.model_registry import ModelRegistry
from registries.prompt_registry import PromptRegistry

__all__ = ["ModelRegistry", "PromptRegistry"]