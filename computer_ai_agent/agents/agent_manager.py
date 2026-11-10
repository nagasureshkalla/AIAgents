"""Creates, caches and switches between application sub-agents.

Plugins live in applications/plugins/<name>/:
    agent.py        defines `AGENT` (a GenericApplicationAgent subclass) and `APP_KEYS` (names/ids it serves)
    knowledge.json  editable seed knowledge (copied to data/applications/<app>/knowledge.json on first use)
Apps without a plugin get a dynamically created `<Name>SubAgent` class built on GenericApplicationAgent.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agents.application_agent import GenericApplicationAgent

if TYPE_CHECKING:
    from applications.application_detector import AppEntry
    from core.runtime import Runtime

logger = logging.getLogger(__name__)

PLUGIN_DIR = Path(__file__).resolve().parent.parent / "applications" / "plugins"


def class_name_for(app_name: str) -> str:
    words = re.findall(r"[A-Za-z0-9]+", app_name) or ["App"]
    name = "".join(w[:1].upper() + w[1:] for w in words)
    if name[0].isdigit():
        name = "App" + name
    return f"{name}SubAgent"


class AgentManager:
    def __init__(self, runtime: Runtime, plugin_dir: Path = PLUGIN_DIR) -> None:
        self.rt = runtime
        self.plugin_dir = plugin_dir
        self.plugins: dict[str, type[GenericApplicationAgent]] = {}   # key (lower) -> class
        self.agents: dict[str, GenericApplicationAgent] = {}           # desktop id -> live sub-agent
        self.active: str | None = None
        self.load_plugins()

    def load_plugins(self) -> None:
        if not self.plugin_dir.is_dir():
            return
        for agent_py in sorted(self.plugin_dir.glob("*/agent.py")):
            folder = agent_py.parent
            try:
                spec = importlib.util.spec_from_file_location(f"cua_plugin_{folder.name}", agent_py)
                assert spec and spec.loader
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                cls: type[GenericApplicationAgent] = mod.AGENT
                keys = [k.lower() for k in getattr(mod, "APP_KEYS", [folder.name])]
            except Exception:  # noqa: BLE001 - a broken plugin must not take the system down
                logger.exception("failed to load plugin %s", folder.name)
                continue
            kfile = folder / "knowledge.json"
            if kfile.exists():
                try:
                    cls.knowledge_seed = json.loads(kfile.read_text())
                except json.JSONDecodeError:
                    logger.warning("invalid knowledge.json in plugin %s", folder.name)
            cls.plugin_name = folder.name
            for k in keys:
                self.plugins[k] = cls
            logger.info("plugin %s -> %s (%s)", folder.name, cls.__name__, ", ".join(keys))

    def plugin_for(self, entry: AppEntry) -> type[GenericApplicationAgent] | None:
        candidates = {entry.id.lower(), entry.name.lower(), entry.binary.lower()} | entry.window_keys()
        for c in candidates:
            if c in self.plugins:
                return self.plugins[c]
        return None

    def get(self, entry: AppEntry) -> tuple[GenericApplicationAgent, bool]:
        """Returns (agent, created)."""
        agent = self.agents.get(entry.id)
        if agent is not None:
            self.active = entry.id
            return agent, False
        cls = self.plugin_for(entry)
        if cls is None:
            # Unknown app: build a dedicated sub-agent class at runtime. Its knowledge starts empty and grows.
            cls = type(class_name_for(entry.name), (GenericApplicationAgent,),
                       {"plugin_name": "", "knowledge_seed": {}, "__module__": __name__})
        agent = cls(self.rt, entry)
        self.agents[entry.id] = agent
        self.active = entry.id
        return agent, True

    def launch_options(self, entry: AppEntry) -> tuple[list[str], dict[str, str]]:
        """Launch flags come from the plugin (e.g. Chrome's accessibility flag) without creating the agent yet."""
        cls = self.plugin_for(entry)
        if cls is None:
            return [], {}
        probe = cls.__new__(cls)
        probe.rt, probe.entry, probe.app_name = self.rt, entry, entry.name
        return probe.launch_args(), probe.launch_env()

    def list(self) -> list[dict[str, Any]]:
        out = [a.info() | {"active": k == self.active} for k, a in self.agents.items()]
        return out

    def plugin_names(self) -> list[str]:
        return sorted({c.plugin_name for c in self.plugins.values()})
