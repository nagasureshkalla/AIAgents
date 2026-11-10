"""Real-desktop integration tests (no model needed). They drive an X display, so they only run when
AGENT_TEST_DISPLAY points at a throwaway screen, e.g. a nested Xephyr:

    Xephyr :7 -screen 1280x800 -ac & DISPLAY=:7 xfwm4 &
    AGENT_TEST_DISPLAY=:7 pytest -m desktop
"""

from __future__ import annotations

import os
import time

import pytest

DISPLAY = os.getenv("AGENT_TEST_DISPLAY", "")
pytestmark = [pytest.mark.desktop,
              pytest.mark.skipif(not DISPLAY, reason="set AGENT_TEST_DISPLAY to a test X display to run")]


@pytest.fixture(scope="module")
def desktop():  # noqa: ANN201
    from actions.platform import create_desktop

    return create_desktop(DISPLAY)


@pytest.fixture(scope="module")
def calculator(desktop):  # noqa: ANN001, ANN201
    from applications.application_manager import ApplicationManager

    apps = ApplicationManager(desktop.windows, DISPLAY, 20)
    entry, win, _ = apps.ensure_open("calculator")
    return apps, entry, win


def test_capture_and_windows(desktop):  # noqa: ANN001
    t = time.perf_counter()
    img = desktop.screen.grab()
    assert (time.perf_counter() - t) < 0.5
    assert img.size == desktop.screen.size()
    assert isinstance(desktop.windows.list_windows(), list)


def test_launch_type_and_read_back_through_accessibility(desktop, calculator):  # noqa: ANN001
    from config.settings import Settings
    from vision.accessibility import AccessibilityReader
    from vision.ui_detector import UIDetector

    apps, entry, win = calculator
    assert desktop.windows.activate(win.id)
    desktop.keyboard.press("Escape")
    desktop.keyboard.type_text("12*12\n")  # real XTest keys: GTK4 ignores synthetic XSendEvent keys
    time.sleep(0.6)
    det = UIDetector(desktop, Settings(ocr_enabled=False), None, AccessibilityReader())
    obs = det.observe(use_cache=False)
    assert obs.active_window and obs.active_window.id == win.id
    assert "accessibility" in obs.sources
    assert any("144" in e.text for e in obs.elements), obs.summary(40)
    buttons = [e for e in obs.elements if e.type == "button" and e.text == "7"]
    assert buttons, "GTK4 window-relative coordinates must map to screen coordinates"
    x, y, w, h = buttons[0].bbox
    wx, wy, ww, wh = win.geometry
    assert wx <= x <= wx + ww and wy <= y <= wy + wh


def test_click_via_element_center(desktop, calculator):  # noqa: ANN001
    from actions.action_engine import ActionEngine
    from config.settings import Settings
    from core.types import Action
    from vision.accessibility import AccessibilityReader
    from vision.ui_detector import UIDetector

    apps, entry, win = calculator
    desktop.windows.activate(win.id)
    desktop.keyboard.press("Escape")
    det = UIDetector(desktop, Settings(ocr_enabled=False), None, AccessibilityReader())
    obs = det.observe(use_cache=False)
    seven = next(e for e in obs.elements if e.type == "button" and e.text == "7")
    import asyncio

    res = asyncio.run(ActionEngine(desktop, 0.2).execute(Action("click", element=seven.id), obs))
    assert res.ok and res.duration_ms < 500
    time.sleep(0.4)
    obs2 = det.observe(use_cache=False)
    assert any(e.type == "textbox" and e.text.strip() == "7" for e in obs2.elements), obs2.summary(30)
