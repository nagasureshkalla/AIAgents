"""Speech agent: the Control Room's voice front door.

The dashboard listens all the time (voice activity detection in the browser). Every utterance is transcribed by the
local Whisper model and checked for the wake word ("Hey CC"); only then is the request handed to the manager, and the
reply is spoken with the local Piper voice. Anything said without the wake word is dropped and never stored.
"""

from __future__ import annotations

import asyncio
import re
import uuid
from dataclasses import dataclass
from typing import Any

from app.agents.base_agent import BaseAgent
from app.schemas.agent import AgentTask

# Whisper writes "Hey CC" as "Hey CC", "Hey C.C.", "Hey, see see", "ACC" ... (checked against real transcripts).
_GREETING = r"(?:hey|hay|hei|hi|heh|eh|ay|okay|ok)"
# ... and through Bluetooth (16 kHz) mics often "Hey Casey" / "Hey KC" / "Hey Kacey" / "Hey Sissi".
_NAME = (r"(?:c\.?[\s-]?c\b\.?|see[\s-]?see|sea[\s-]?sea|see[\s-]?c\b|c[\s-]?see|si[\s-]?si|sis+[iy]|cece|cici|ceci|"
         r"k\.?[\s-]?c\b\.?|ka[cs]e?y|kaycee|kayce|casey|kc)")
_SEP = r"[\s,.!?:;-]*"
# "hey cc" anywhere in the utterance; the fused forms ("ACC", "A C C") only at its start, so "send a cc" is no wake.
_WAKE_ANYWHERE = re.compile(rf"\b{_GREETING}{_SEP}{_NAME}", re.I)
_WAKE_AT_START = re.compile(rf"^\s*(?:(?:um|uh|so)[\s,]+)?(?:a[\s-]?{_NAME}|acc\b|hcc\b|heycc\b|hacc\b)", re.I)
_LEADING_FILLER = re.compile(r"^(?:[\s,.!?:;-]|\b(?:at|and|um|uh)\b)+", re.I)


# "Hey, ..." at the start: maybe "Hey CC" with the name lost in transcription (checked acoustically).
_GREETING_ONLY = re.compile(rf"^\W*{_GREETING}\b[\s,.!?:;-]*", re.I)


def maybe_lost_wake(text: str) -> bool:
    return bool(_GREETING_ONLY.match(text)) and not match_wake(text).woke


def after_greeting(text: str) -> str:
    return _GREETING_ONLY.sub("", text, count=1).strip().rstrip(" ,;:-")


# What Whisper writes for noise, breathing and stray sounds: never sent to the agents in conversation mode.
_FILLER = {"thank you", "thanks", "thank you very much", "thanks for watching", "thank you for watching", "you",
           "bye", "okay", "ok", "oh", "uh", "um", "hmm", "mm", "ah", "yeah", "so", "the end", "silence", "music"}


def is_request(text: str) -> bool:
    """Conversation mode: whether an utterance is worth sending to the agents (not noise / a filler word)."""
    words = re.findall(r"[a-z0-9']+", text.lower())
    if len(words) < 2:
        return False
    return " ".join(words) not in _FILLER


@dataclass(frozen=True)
class WakeMatch:
    woke: bool
    command: str = ""  # what was said after the wake word ("" = the wake word alone: listen for the request)


def match_wake(text: str) -> WakeMatch:
    m = _WAKE_AT_START.search(text) or _WAKE_ANYWHERE.search(text)
    if not m:
        return WakeMatch(False)
    command = _LEADING_FILLER.sub("", text[m.end():]).strip()
    return WakeMatch(True, command.rstrip(" ,;:-"))


class SpeechAgent(BaseAgent):
    name = "speech"
    description = (
        'Voice front door: always listening in the dashboard, wakes on "Hey CC", turns your speech into text with '
        "local Whisper, hands it to the manager and reads the answer aloud with a local Piper voice."
    )
    role = "control"  # never routed to like a worker; it sits before the manager
    capabilities = ["speech_to_text", "wake_word", "text_to_speech"]

    async def run(self, task: AgentTask) -> dict[str, Any]:
        """Read the prompt aloud: writes a WAV under data/output/speech/ and returns its path."""
        tts = self.ctx.tts
        if tts is None:
            raise RuntimeError("Text to speech is not configured")
        audio = await tts.synthesize(task.prompt)
        target = self.ctx.settings.output_dir / "speech" / f"{uuid.uuid4().hex[:10]}.wav"
        target.parent.mkdir(parents=True, exist_ok=True)
        await asyncio.to_thread(target.write_bytes, audio)
        return {"text": task.prompt, "audio_path": str(target)}
