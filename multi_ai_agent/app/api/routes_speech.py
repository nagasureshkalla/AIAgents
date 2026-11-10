"""Speech agent API: local Whisper speech-to-text, the "Hey CC" wake word, and local Piper text-to-speech."""

from __future__ import annotations

import logging
import re

from fastapi import APIRouter, File, Form, Response, UploadFile
from pydantic import BaseModel, Field

from app.agents.speech_agent import WakeMatch, after_greeting, is_request, match_wake, maybe_lost_wake
from app.api.deps import ContainerDep
from app.core.container import Container
from app.core.exceptions import InvalidRequestError

router = APIRouter(prefix="/api/speech", tags=["speech"])
logger = logging.getLogger(__name__)


class SpeakRequest(BaseModel):
    text: str = Field(..., min_length=1)


async def _read(container: Container, file: UploadFile) -> bytes:
    max_bytes = int(container.settings.speech.max_audio_mb * 1024 * 1024)
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise InvalidRequestError(f"Recording exceeds {container.settings.speech.max_audio_mb} MB")
    if not data:
        raise InvalidRequestError("Empty recording")
    return data


@router.get("", summary="Speech status: Whisper and Piper installed / loaded, wake phrase")
async def speech_status(container: ContainerDep) -> dict:
    cfg = container.settings.speech
    return {
        **container.speech.status(),
        "tts": container.tts.status(),
        "wake_phrase": cfg.wake_phrase,
        "require_wake_word": cfg.require_wake_word,
        "end_of_speech_seconds": cfg.end_of_speech_seconds,
        "follow_up_seconds": cfg.follow_up_seconds,
    }


@router.post("/transcribe", summary="Turn a recording (webm/ogg/mp4/wav) into text, locally")
async def transcribe(
    container: ContainerDep,
    file: UploadFile = File(..., description="Audio recorded by the dashboard microphone button"),
) -> dict:
    return await container.speech.transcribe(await _read(container, file))


@router.post("/listen", summary="Transcribe one utterance and check it for the wake word")
async def listen(
    container: ContainerDep,
    file: UploadFile = File(..., description="One utterance cut by the dashboard's voice activity detection"),
    follow_up: bool = Form(default=False, description="The wake word was just said alone: this is the request"),
) -> dict:
    """`woke` is true when the utterance starts the agent. `command` is the request to send ("" = only the wake
    word was said; the next utterance should be posted with follow_up=true). Text without the wake word is
    returned to the caller only, never stored."""
    cfg = container.settings.speech
    if not cfg.require_wake_word:  # conversation mode: every real utterance is a request
        heard = await container.speech.transcribe(await _read(container, file))
        wake = match_wake(heard["text"])  # "Hey CC, ..." still works: the greeting is dropped from the request
        command = (wake.command if wake.woke else heard["text"]) if is_request(heard["text"]) else ""
        logger.info("listen: %.1fs of audio, %d chars heard, request=%s%s", heard["seconds"], len(heard["text"]),
                    bool(command), f", text={command!r}" if command else "")
        return {**heard, "woke": bool(command), "command": command}
    heard = await container.speech.transcribe(await _read(container, file),
                                              wake_check=None if follow_up else maybe_lost_wake,
                                              phrase=" " + cfg.wake_phrase)
    wake = match_wake(heard["text"])
    if not wake.woke and heard.get("wake_score", -99.0) >= cfg.wake_score_threshold:
        wake = WakeMatch(True, after_greeting(heard["text"]))  # "Hey, open ..." that was really "Hey CC, open ..."
    # What was said stays out of the log unless it was addressed to the agent. A missed wake attempt (starts with a
    # greeting) logs only its first three words, so a misheard "Hey CC" can be added to the matcher.
    woke = wake.woke or follow_up
    attempt = not woke and re.match(r"\W*(?:hey|hay|hi|ok|okay|a|eh|ay)\b", heard["text"], re.I)
    shown = heard["text"] if woke else " ".join(heard["text"].split()[:3]) if attempt else ""
    logger.info("listen: %.1fs of audio, %d chars heard, woke=%s%s%s", heard["seconds"], len(heard["text"]), woke,
                f", wake_score={heard['wake_score']}" if "wake_score" in heard else "",
                f", {'text' if woke else 'starts'}={shown!r}" if shown else "")
    if follow_up:  # everything said is the request (minus a repeated "Hey CC")
        command = wake.command if wake.woke else heard["text"]
        return {**heard, "woke": bool(command), "command": command}
    return {**heard, "woke": wake.woke, "command": wake.command}


@router.post("/speak", summary="Speak text with the local Piper voice (audio/wav)", response_class=Response)
async def speak(request: SpeakRequest, container: ContainerDep) -> Response:
    audio = await container.tts.synthesize(request.text)
    return Response(audio, media_type="audio/wav", headers={"Cache-Control": "no-store"})
