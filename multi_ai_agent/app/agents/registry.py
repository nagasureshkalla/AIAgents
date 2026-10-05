"""Agent discovery and capability index.

Any concrete BaseAgent subclass with a `name` defined in a module of `app.agents` is discovered
automatically. Agents living elsewhere can opt in with the `@register_agent` decorator.
"""

from __future__ import annotations

import importlib
import inspect
import logging
import pkgutil

from app.agents.base_agent import AgentContext, BaseAgent
from app.core.exceptions import AgentNotFoundError
from app.schemas.agent import AgentInfo

logger = logging.getLogger(__name__)

_EXTRA_AGENTS: list[type[BaseAgent]] = []
_SKIP_MODULES = {"base_agent", "registry"}


def register_agent(cls: type[BaseAgent]) -> type[BaseAgent]:
    if cls not in _EXTRA_AGENTS:
        _EXTRA_AGENTS.append(cls)
    return cls


def discover_agent_classes(package: str = "app.agents") -> list[type[BaseAgent]]:
    pkg = importlib.import_module(package)
    classes: list[type[BaseAgent]] = []
    for mod_info in pkgutil.iter_modules(pkg.__path__):
        if mod_info.name in _SKIP_MODULES or mod_info.name.startswith("_"):
            continue
        module = importlib.import_module(f"{package}.{mod_info.name}")
        for obj in vars(module).values():
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseAgent)
                and obj is not BaseAgent
                and not inspect.isabstract(obj)
                and obj.__module__ == module.__name__
                and obj.name
            ):
                classes.append(obj)
    for cls in _EXTRA_AGENTS:
        if cls not in classes:
            classes.append(cls)
    return classes


class AgentRegistry:
    def __init__(self, ctx: AgentContext) -> None:
        self.ctx = ctx
        self._agents: dict[str, BaseAgent] = {}
        ctx.registry = self

    @classmethod
    def build(cls, ctx: AgentContext, package: str = "app.agents") -> "AgentRegistry":
        registry = cls(ctx)
        for agent_cls in discover_agent_classes(package):
            if not ctx.settings.agent_config(agent_cls.name).enabled:
                logger.info("agent %s disabled by config", agent_cls.name)
                continue
            registry.register(agent_cls)
        for required in ("manager", "planner"):
            if required not in registry._agents:
                raise RuntimeError(f"Required control agent '{required}' is missing or disabled")
        return registry

    def register(self, agent: type[BaseAgent] | BaseAgent) -> BaseAgent:
        instance = agent(self.ctx) if isinstance(agent, type) else agent
        if instance.name in self._agents:
            logger.warning("agent %s re-registered; replacing previous instance", instance.name)
        self._agents[instance.name] = instance
        return instance

    def unregister(self, name: str) -> None:
        self._agents.pop(name, None)

    def get(self, name: str) -> BaseAgent:
        try:
            return self._agents[name]
        except KeyError:
            raise AgentNotFoundError(f"Unknown agent '{name}'. Available: {', '.join(self._agents)}") from None

    def has(self, name: str) -> bool:
        return name in self._agents

    def names(self) -> list[str]:
        return list(self._agents)

    def all(self) -> list[BaseAgent]:
        return list(self._agents.values())

    def workers(self) -> list[BaseAgent]:
        return [a for a in self._agents.values() if a.role == "worker"]

    def find_by_capability(self, capability: str) -> list[BaseAgent]:
        """Workers declaring `capability`, primary specialists (capability listed earlier) first."""
        matches = [a for a in self.workers() if capability in a.capabilities]
        return sorted(matches, key=lambda a: a.capabilities.index(capability))

    def first_for(self, capability: str) -> BaseAgent | None:
        matches = self.find_by_capability(capability)
        return matches[0] if matches else None

    def capabilities(self) -> dict[str, list[str]]:
        return {a.name: list(a.capabilities) for a in self._agents.values()}

    def all_capabilities(self) -> set[str]:
        return {cap for a in self.workers() for cap in a.capabilities}

    def info(self) -> list[AgentInfo]:
        return [a.info() for a in self._agents.values()]
