"""Discovers installed applications (freedesktop .desktop entries + PATH) and matches names and windows to them."""

from __future__ import annotations

import difflib
import logging
import os
import re
import shlex
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from core.types import WindowInfo

logger = logging.getLogger(__name__)

XDG_DIRS = [
    Path.home() / ".local/share/applications",
    Path("/usr/local/share/applications"),
    Path("/usr/share/applications"),
    Path("/var/lib/snapd/desktop/applications"),
    Path("/var/lib/flatpak/exports/share/applications"),
    Path.home() / ".local/share/flatpak/exports/share/applications",
]

# Everyday names -> desktop ids (first installed one wins). Unknown names fall through to fuzzy matching.
ALIASES: dict[str, list[str]] = {
    "chrome": ["google-chrome", "chromium", "chromium-browser", "chromium_chromium"],
    "google chrome": ["google-chrome"],
    "chromium": ["chromium", "chromium-browser", "chromium_chromium"],
    "firefox": ["firefox", "firefox_firefox", "org.mozilla.firefox"],
    "browser": ["@default-browser", "google-chrome", "firefox", "firefox_firefox"],
    "web browser": ["@default-browser", "google-chrome", "firefox"],
    "vs code": ["code", "code_code", "com.visualstudio.code", "code-oss"],
    "vscode": ["code", "code_code", "com.visualstudio.code", "code-oss"],
    "visual studio code": ["code", "code_code", "com.visualstudio.code"],
    "code": ["code", "code_code"],
    "terminal": ["org.gnome.Terminal", "org.gnome.Console", "org.gnome.Ptyxis", "xterm", "konsole"],
    "console": ["org.gnome.Console", "org.gnome.Terminal"],
    "file manager": ["org.gnome.Nautilus", "nautilus", "thunar", "org.kde.dolphin"],
    "files": ["org.gnome.Nautilus", "nautilus", "thunar"],
    "file explorer": ["org.gnome.Nautilus", "thunar"],
    "nautilus": ["org.gnome.Nautilus"],
    "calculator": ["org.gnome.Calculator", "gnome-calculator", "org.kde.kcalc"],
    "text editor": ["org.gnome.TextEditor", "org.gnome.gedit", "gedit", "mousepad"],
    "gedit": ["org.gnome.gedit", "gedit"],
    "settings": ["org.gnome.Settings", "gnome-control-center"],
    "system monitor": ["org.gnome.SystemMonitor", "gnome-system-monitor"],
    "libreoffice": ["libreoffice-startcenter"], "writer": ["libreoffice-writer"],
    "word": ["libreoffice-writer"], "excel": ["libreoffice-calc"], "spreadsheet": ["libreoffice-calc"],
    "blender": ["blender", "blender_blender", "org.blender.Blender"],
    "steam": ["steam", "com.valvesoftware.Steam"],
    "spotify": ["spotify", "spotify_spotify", "com.spotify.Client"],
    "photoshop": ["photoshop"], "gimp": ["gimp", "org.gimp.GIMP", "gimp_gimp"],
}

FIELD_CODES = re.compile(r"%[fFuUdDnNickvm]")


@dataclass
class AppEntry:
    id: str                    # desktop id without .desktop, or "path:<exe>"
    name: str
    exec: str
    generic_name: str = ""
    keywords: list[str] = field(default_factory=list)
    wm_class: str = ""
    path: str = ""
    no_display: bool = False
    terminal: bool = False

    @property
    def argv(self) -> list[str]:
        parts = shlex.split(FIELD_CODES.sub("", self.exec))
        return [p for p in parts if p]

    @property
    def binary(self) -> str:
        argv = self.argv
        i = 0
        if argv and Path(argv[0]).name == "env":  # env VAR=1 /snap/bin/code ...
            i = 1
            while i < len(argv) and "=" in argv[i]:
                i += 1
        return Path(argv[i]).name if i < len(argv) else ""

    def window_keys(self) -> set[str]:
        """Lower-case strings that may appear as a window's WM_CLASS for this app."""
        keys = {self.binary.lower(), self.id.lower(), self.id.split(".")[-1].lower(), self.name.lower()}
        if self.wm_class:
            keys.add(self.wm_class.lower())
        if self.binary.endswith("-stable"):
            keys.add(self.binary[: -len("-stable")].lower())
        if "_" in self.id:  # snap: code_code -> code
            keys.add(self.id.split("_")[0].lower())
        return {k for k in keys if k}

    def to_dict(self) -> dict:
        return {"id": self.id, "name": self.name, "exec": self.exec, "generic_name": self.generic_name,
                "wm_class": self.wm_class, "keywords": self.keywords}


def _parse_desktop(path: Path) -> AppEntry | None:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None
    section, data = None, {}
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("["):
            section = line
            continue
        if section != "[Desktop Entry]" or "=" not in line or line.startswith("#"):
            continue
        k, v = line.split("=", 1)
        data.setdefault(k.strip(), v.strip())  # first (unlocalized) value wins
    if data.get("Type") != "Application" or not data.get("Exec") or data.get("Hidden") == "true":
        return None
    return AppEntry(
        id=path.stem,
        name=data.get("Name", path.stem),
        exec=data["Exec"],
        generic_name=data.get("GenericName", ""),
        keywords=[k for k in data.get("Keywords", "").split(";") if k],
        wm_class=data.get("StartupWMClass", ""),
        path=str(path),
        no_display=data.get("NoDisplay") == "true",
        terminal=data.get("Terminal") == "true",
    )


class ApplicationDetector:
    def __init__(self, dirs: list[Path] | None = None) -> None:
        self.dirs = dirs or XDG_DIRS
        self.apps: dict[str, AppEntry] = {}

    def discover(self) -> dict[str, AppEntry]:
        apps: dict[str, AppEntry] = {}
        for d in self.dirs:
            if not d.is_dir():
                continue
            for f in sorted(d.glob("*.desktop")):
                entry = _parse_desktop(f)
                if entry and entry.id not in apps:  # earlier dirs (user) override system ones
                    apps[entry.id] = entry
        self.apps = apps
        logger.info("discovered %d applications", len(apps))
        return apps

    def _default_browser(self) -> str | None:
        try:
            out = subprocess.run(["xdg-settings", "get", "default-web-browser"], capture_output=True, text=True,
                                 timeout=3).stdout.strip()
            return out.removesuffix(".desktop") or None
        except Exception:  # noqa: BLE001
            return None

    def find(self, name: str) -> AppEntry | None:
        """Best installed match for a spoken app name ("chrome", "VS Code", "files", "blender"...)."""
        if not self.apps:
            self.discover()
        q = name.strip().lower().removesuffix(" app").removesuffix(" application").strip()
        if not q:
            return None
        for alias_id in ALIASES.get(q, []):
            if alias_id == "@default-browser":
                alias_id = self._default_browser() or ""
            if alias_id in self.apps:
                return self.apps[alias_id]
        best: tuple[float, AppEntry | None] = (0.0, None)
        for app in self.apps.values():
            names = [app.name.lower(), app.id.lower(), app.binary.lower(), app.generic_name.lower()]
            score = 0.0
            if q in (app.name.lower(), app.id.lower(), app.binary.lower()):
                score = 1.0
            elif any(n.startswith(q) or q in n.split() for n in names if n):
                score = 0.85
            elif q in [k.lower() for k in app.keywords]:
                score = 0.7
            else:
                score = max(difflib.SequenceMatcher(None, q, n).ratio() for n in names if n) * 0.8
            if app.no_display:
                score -= 0.1
            if score > best[0]:
                best = (score, app)
        if best[1] is not None and best[0] >= 0.6:
            return best[1]
        exe = shutil.which(q.replace(" ", "-")) or shutil.which(q.replace(" ", ""))
        if exe:
            return AppEntry(id=f"path:{Path(exe).name}", name=name, exec=exe)
        return None

    def matches_window(self, app: AppEntry, win: WindowInfo, pids: set[int] | None = None) -> bool:
        if pids and win.pid in pids:
            return True
        cls = win.wm_class.lower()
        if cls and (cls in app.window_keys() or any(k and (k in cls or cls in k) for k in app.window_keys()
                                                     if len(k) > 3)):
            return True
        return False

    @staticmethod
    def resolve_exec(app: AppEntry) -> list[str]:
        argv = app.argv
        if argv and not os.path.isabs(argv[0]) and Path(argv[0]).name != "env" and not shutil.which(argv[0]):
            raise FileNotFoundError(f"{app.name}: executable {argv[0]!r} not found")
        return argv
