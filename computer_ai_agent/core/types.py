"""Data types shared by every layer: UI elements, observations, actions and inter-agent messages."""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

# Where an element came from, best first. The planner prefers elements from earlier sources.
SOURCES = ("accessibility", "ocr", "vision")


@dataclass
class Element:
    """One interactable or readable thing on screen, in absolute screen pixels."""

    id: int
    type: str                      # button, textbox, link, menu, menuitem, tab, checkbox, list, icon, text, ...
    text: str
    bbox: tuple[int, int, int, int]  # x, y, width, height
    source: str = "accessibility"  # accessibility | ocr | vision
    states: list[str] = field(default_factory=list)  # focused, focusable, editable, checked, selected, ...
    confidence: float = 1.0
    window: str = ""

    @property
    def center(self) -> tuple[int, int]:
        x, y, w, h = self.bbox
        return x + w // 2, y + h // 2

    def describe(self, limit: int = 80) -> str:
        x, y, w, h = self.bbox
        flags = f" [{', '.join(self.states)}]" if self.states else ""
        text = self.text.replace("\n", "⏎").strip()
        if len(text) > limit:
            # say explicitly that this is a shortened view, so the model never "continues" text that is complete
            head, tail = text[: int(limit * 0.6)], text[-int(limit * 0.3):]
            text = f"{head} … {tail}\" (shortened view of {len(self.text)} characters)"
            return f'[{self.id}] {self.type} "{text} at ({x},{y},{w}x{h}){flags} <{self.source}>'
        return f'[{self.id}] {self.type} "{text}" at ({x},{y},{w}x{h}){flags} <{self.source}>'

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["center"] = list(self.center)
        return d


@dataclass
class WindowInfo:
    id: int
    title: str
    wm_class: str
    pid: int | None
    geometry: tuple[int, int, int, int]  # x, y, width, height
    active: bool = False
    desktop: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Observation:
    """What the agent knows about the screen at one moment."""

    timestamp: float
    screen_size: tuple[int, int]
    active_window: WindowInfo | None
    elements: list[Element]
    image: Any = None              # PIL.Image of the full screen (never serialized)
    image_hash: str = ""
    ocr_text: str = ""
    focused: Element | None = None
    sources: list[str] = field(default_factory=list)

    def element(self, element_id: int) -> Element | None:
        return next((e for e in self.elements if e.id == element_id), None)

    def summary(self, limit: int = 120, text_limit: int = 80) -> str:
        """Compact text the planner reads instead of raw pixels."""
        editable = ("textbox", "terminal", "document")
        f = self.focused
        if not (f is not None and f.type in editable and len(f.text) > 80):  # else: the biggest text field on screen
            f = max((e for e in self.elements if e.type in editable), key=lambda e: len(e.text), default=None)
        block = f is not None and len(f.text) > 80

        def line(e: Element) -> str:
            if block and e is f:
                x, y, w, h = e.bbox
                return (f'[{e.id}] {e.type} <{len(e.text)} characters: full content shown below> at ({x},{y},{w}x{h}) '
                        f'[{", ".join(e.states)}] <{e.source}>')
            return e.describe()

        w = self.active_window
        lines = [f"Screen: {self.screen_size[0]}x{self.screen_size[1]}"]
        if w:
            lines.append(f'Active window: "{w.title}" (class={w.wm_class}, pid={w.pid}, geometry={w.geometry})')
        else:
            lines.append("Active window: none")
        if self.focused:
            lines.append(f"Keyboard focus: {line(self.focused)}")
        lines.append(f"Visible elements ({len(self.elements)}, sources={'+'.join(self.sources) or 'none'}):")
        shown = self.elements
        if len(shown) > limit:
            # keep controls (things you can act on) before plain text, then restore screen order
            ranked = sorted(shown, key=lambda e: (e.type == "text", e.source != "accessibility", e.id))
            keep = {e.id for e in ranked[:limit]}
            shown = [e for e in self.elements if e.id in keep]
        for e in shown:
            lines.append("  " + line(e))
        if len(self.elements) > limit:
            lines.append(f"  ... {len(self.elements) - limit} less important elements not shown")
        # The focused text field's content, verbatim with real line breaks (a one-line "⏎" rendering or a head/tail
        # cut makes models "see" duplicated or broken text and rewrite correct content).
        if block:
            text = f.text
            if len(text) > text_limit:
                keep = text_limit // 2
                omitted = len(text) - 2 * keep
                text = f"{text[:keep]}\n[... {omitted} characters omitted here ...]\n{text[-keep:]}"
            label = "focused field" if f is self.focused else "text field"
            lines.append(f"Content of {label} [{f.id}] ({len(f.text)} characters, exact):\n<<<\n{text}\n>>>")
        return "\n".join(lines)

    def to_dict(self, with_elements: bool = True) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "screen_size": list(self.screen_size),
            "active_window": self.active_window.to_dict() if self.active_window else None,
            "elements": [e.to_dict() for e in self.elements] if with_elements else [],
            "focused": self.focused.to_dict() if self.focused else None,
            "sources": self.sources,
            "image_hash": self.image_hash,
        }


ACTION_TYPES = (
    "mouse_move", "click", "double_click", "right_click", "mouse_drag", "drag_and_drop", "scroll",
    "type_text", "key_press", "hotkey", "wait", "screenshot", "copy", "paste", "select_text",
    "open_application", "close_application", "switch_application",
    # control actions the planner may emit; the agent handles them instead of the OS
    "done", "fail", "ask_user",
)

# Model output is forgiving: these spellings map onto the canonical action names.
ACTION_ALIASES = {
    "mouse_click": "click", "left_click": "click", "tap": "click", "press": "key_press", "key": "key_press",
    "type": "type_text", "write": "type_text", "input": "type_text", "enter_text": "type_text",
    "shortcut": "hotkey", "keys": "hotkey", "sleep": "wait", "move": "mouse_move", "drag": "mouse_drag",
    "open": "open_application", "launch": "open_application", "close": "close_application",
    "switch": "switch_application", "focus_application": "switch_application", "finish": "done",
    "complete": "done", "success": "done", "give_up": "fail", "ask": "ask_user", "doubleclick": "double_click",
    "rightclick": "right_click", "select_all": "select_text",
}


@dataclass
class Action:
    action: str
    element: int | None = None      # element id from the observation: preferred over raw coordinates
    x: int | None = None
    y: int | None = None
    x2: int | None = None           # drag target
    y2: int | None = None
    target: int | None = None       # drag target element
    text: str = ""
    keys: list[str] = field(default_factory=list)
    amount: int = 0                 # scroll clicks (+down / -up) or wait seconds
    app: str = ""
    reason: str = ""
    expect: str = ""                # what should be true after the action (used by the verifier)
    question: str = ""              # ask_user
    choices: list[str] = field(default_factory=list)
    target_label: str = ""          # human-readable target, filled in when resolved

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Action:
        name = str(d.get("action") or d.get("type") or "").strip().lower().replace(" ", "_").replace("-", "_")
        name = ACTION_ALIASES.get(name, name)
        if name not in ACTION_TYPES:
            raise ValueError(f"unknown action {name!r}")

        def _int(v: Any) -> int | None:
            if isinstance(v, dict):  # {"id": 27} / {"element": 27}
                v = v.get("id", v.get("element", v.get("element_id")))
            if isinstance(v, str):
                v = v.strip().lstrip("[#").rstrip("]")
            try:
                return None if v is None or v == "" else int(round(float(v)))
            except (TypeError, ValueError):
                return None

        keys = d.get("keys") or d.get("key") or []
        if isinstance(keys, str):
            keys = [k for k in keys.replace("+", " ").split() if k]
        coords = d.get("coordinates") or d.get("coordinate") or d.get("location")
        x, y = _int(d.get("x")), _int(d.get("y"))
        if (x is None or y is None) and isinstance(coords, (list, tuple)) and len(coords) >= 2:
            x, y = _int(coords[0]), _int(coords[1])
        element = d.get("element", d.get("element_id", d.get("id", d.get("index"))))
        target = d.get("target", d.get("target_element"))
        if element is None and name not in ("mouse_drag", "drag_and_drop") and target is not None:
            element, target = target, None  # models often call the clicked element "target"
        choices = d.get("choices") or d.get("options") or []
        return cls(
            action=name,
            element=_int(element),
            x=x,
            y=y,
            x2=_int(d.get("x2")),
            y2=_int(d.get("y2")),
            target=_int(target),
            text=str(d.get("text") or d.get("value") or d.get("command") or ""),
            keys=[str(k) for k in keys],
            amount=_int(d.get("amount", d.get("clicks", d.get("seconds", 0)))) or 0,
            app=str(d.get("app") or d.get("application") or ""),
            reason=str(d.get("reason") or d.get("thought") or ""),
            expect=str(d.get("expect") or d.get("expected") or d.get("expectation") or ""),
            question=str(d.get("question") or ""),
            choices=[str(c) for c in choices] if isinstance(choices, list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, "", [], 0) or k == "action"}

    def short(self) -> str:
        bits = [self.action]
        if self.target_label:
            bits.append(f"→ {self.target_label}")
        elif self.element is not None:
            bits.append(f"→ element {self.element}")
        elif self.x is not None:
            bits.append(f"→ ({self.x},{self.y})")
        if self.text:
            t = self.text if len(self.text) <= 40 else self.text[:37] + "..."
            bits.append(f'"{t}"')
        if self.keys:
            bits.append("+".join(self.keys))
        if self.app:
            bits.append(self.app)
        return " ".join(bits)


@dataclass
class ActionResult:
    ok: bool
    message: str = ""
    point: tuple[int, int] | None = None   # where the pointer acted, for the UI overlay
    bbox: tuple[int, int, int, int] | None = None
    duration_ms: float = 0.0
    data: dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentMessage:
    """Standard envelope for Master <-> sub-agent communication (logged and streamed to the UI)."""

    sender: str
    recipient: str
    task: str = ""
    status: str = ""               # for responses: completed | failed | needs_user | stopped
    result: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    priority: str = "normal"
    id: str = field(default_factory=lambda: uuid.uuid4().hex[:10])
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["from"] = d.pop("sender")
        d["to"] = d.pop("recipient")
        return d
