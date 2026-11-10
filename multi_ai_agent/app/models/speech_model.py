"""Local speech for the dashboard: speech-to-text (Whisper via transformers, audio decoded by ffmpeg) and
text-to-speech (Piper voices, ONNX on the CPU)."""

from __future__ import annotations

import asyncio
import gc
import io
import logging
import shutil
import subprocess
import threading
import time
import wave
from collections.abc import Callable
from pathlib import Path
from typing import Any

from app.core.config import Settings
from app.core.device import DeviceInfo
from app.core.exceptions import InvalidRequestError

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16_000


class SpeechUnavailableError(InvalidRequestError):
    """Voice input is disabled, or the Whisper model / ffmpeg is missing."""

    status_code = 503
    code = "speech_unavailable"


def decode_audio(data: bytes, max_seconds: float) -> Any:
    """Any browser recording (webm/ogg/mp4/wav) -> 16 kHz mono float32 samples."""
    import numpy as np

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise SpeechUnavailableError("ffmpeg is not installed; it is needed to read microphone recordings")
    proc = subprocess.run(
        [ffmpeg, "-nostdin", "-loglevel", "error", "-i", "pipe:0", "-t", str(max_seconds),
         "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, timeout=60, check=False,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise InvalidRequestError(f"Could not read the recording: {proc.stderr.decode(errors='replace')[-200:]}")
    return np.frombuffer(proc.stdout, dtype=np.float32)


def loudest_frame(audio: Any, frame: int = SAMPLE_RATE // 33) -> float:
    """RMS of the loudest ~30 ms frame."""
    import numpy as np

    usable = len(audio) // frame * frame
    if not usable:
        return 0.0
    frames = audio[:usable].reshape(-1, frame)
    return float(np.sqrt((frames**2).mean(axis=1)).max())


class SpeechToText:
    """Loads the Whisper model on first use and unloads it after `speech.idle_unload_seconds` without requests."""

    def __init__(self, settings: Settings, device_info: DeviceInfo) -> None:
        self.settings = settings
        self.config = settings.speech
        self.device = device_info.torch_device
        self.path = settings.resolve_path(self.config.path)
        self._pipe: Any = None
        self._lock = threading.Lock()
        self._unload_timer: asyncio.TimerHandle | None = None

    @property
    def installed(self) -> bool:
        return (self.path / "config.json").is_file() and any(self.path.glob("*.safetensors"))

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    def status(self) -> dict:
        return {
            "enabled": self.config.enabled,
            "installed": self.installed,
            "loaded": self.loaded,
            "model": self.path.name,
            "language": self.config.language,
            "max_seconds": self.config.max_seconds,
        }

    def _check(self) -> None:
        if not self.config.enabled:
            raise SpeechUnavailableError("Voice input is disabled (speech.enabled: false)")
        if not self.installed:
            raise SpeechUnavailableError(
                f"No speech model at {self.path}. Download one with: python scripts/download_models.py "
                "--model openai/whisper-large-v3-turbo --type speech --name whisper"
            )

    def _load(self) -> None:
        import torch
        from transformers import pipeline

        dtype = torch.float16 if self.device.startswith("cuda") else torch.float32
        started = time.perf_counter()
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=str(self.path),
            dtype=dtype,
            device=self.device,
            model_kwargs={"local_files_only": True},
        )
        logger.info("loaded speech model %s on %s in %.1fs", self.path.name, self.device, time.perf_counter() - started)

    def _wake_score(self, audio: Any, phrase: str) -> float:
        """Mean log-probability that the opening of `audio` says `phrase` (Whisper scoring the phrase, not decoding).

        On a Bluetooth (16 kHz) mic Whisper often drops the short "CC" and writes just "Hey, ...". Scoring " Hey CC"
        against the first seconds tells the two apart: ~-1 when it was said, below -7 when it was not (measured)."""
        import torch

        model, tok = self._pipe.model, self._pipe.tokenizer
        lang = f"<|{self.config.language or 'en'}|>"
        prefix = tok.convert_tokens_to_ids(["<|startoftranscript|>", lang, "<|transcribe|>", "<|notimestamps|>"])
        target = tok(phrase, add_special_tokens=False).input_ids
        feats = self._pipe.feature_extractor(audio[: SAMPLE_RATE * 3], sampling_rate=SAMPLE_RATE, return_tensors="pt")
        feats = feats.input_features.to(model.device, model.dtype)
        ids = torch.tensor([prefix + target], device=model.device)
        with torch.no_grad():
            logits = model(input_features=feats, decoder_input_ids=ids).logits.float()
        logp = torch.log_softmax(logits[0, len(prefix) - 1:-1], -1)
        return float(logp[torch.arange(len(target)), torch.tensor(target)].mean())

    def _run(self, data: bytes, wake_check: Callable[[str], bool] | None = None, phrase: str = "") -> dict:
        audio = decode_audio(data, self.config.max_seconds)
        seconds = len(audio) / SAMPLE_RATE
        if seconds < 0.3 or loudest_frame(audio) < self.config.min_rms:
            return {"text": "", "seconds": round(seconds, 2)}  # silence: Whisper would invent "Thank you."
        generate_kwargs: dict[str, Any] = {"task": "transcribe"}
        if self.config.language:
            generate_kwargs["language"] = self.config.language
        with self._lock:
            if self._pipe is None:
                self._load()
            out = self._pipe(
                {"raw": audio, "sampling_rate": SAMPLE_RATE},
                chunk_length_s=30,
                batch_size=4,
                generate_kwargs=generate_kwargs,
            )
            text = (out.get("text") or "").strip()
            if len(text) > 40 + seconds * 30:  # nobody speaks that fast: a Whisper repetition loop on noise
                text = ""
            result: dict[str, Any] = {"text": text, "seconds": round(seconds, 2)}
            if wake_check is not None and phrase and wake_check(text):
                result["wake_score"] = round(self._wake_score(audio, phrase), 2)
        return result

    async def transcribe(self, data: bytes, wake_check: Callable[[str], bool] | None = None, phrase: str = "") -> dict:
        """Transcribe; when wake_check(text) is true, also score how likely the audio opens with `phrase`."""
        self._check()
        if self._unload_timer is not None:
            self._unload_timer.cancel()
        try:
            return await asyncio.to_thread(self._run, data, wake_check, phrase)
        finally:
            if self.config.idle_unload_seconds:
                loop = asyncio.get_running_loop()
                self._unload_timer = loop.call_later(self.config.idle_unload_seconds, self._unload_soon)

    def _unload_soon(self) -> None:
        asyncio.get_running_loop().run_in_executor(None, self.unload)

    def unload(self) -> None:
        with self._lock:
            if self._pipe is None:
                return
            self._pipe = None
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("unloaded speech model %s", self.path.name)


class TextToSpeech:
    """Piper voice (models/speech/piper/**/*.onnx). Loads on first use; synthesis takes ~0.1 s per sentence on CPU."""

    def __init__(self, settings: Settings) -> None:
        self.config = settings.speech
        self.path = settings.resolve_path(self.config.voice)
        self._voice: Any = None
        self._lock = threading.Lock()

    @property
    def model_file(self) -> Path | None:
        """The configured .onnx, or the best-quality voice in the folder (-high, then -medium, then -low)."""
        if self.path.is_file():
            return self.path
        rank = {"high": 0, "medium": 1, "low": 2, "x_low": 3}
        found = sorted(self.path.rglob("*.onnx"), key=lambda p: (rank.get(p.stem.rsplit("-", 1)[-1], 9), str(p))) \
            if self.path.is_dir() else []
        return found[0] if found else None

    @property
    def installed(self) -> bool:
        model = self.model_file
        return model is not None and Path(f"{model}.json").is_file()

    def status(self) -> dict:
        model = self.model_file
        return {"installed": self.installed, "loaded": self._voice is not None, "voice": model.stem if model else None}

    def _synthesize(self, text: str) -> bytes:
        from piper import PiperVoice, SynthesisConfig

        with self._lock:
            if self._voice is None:
                self._voice = PiperVoice.load(self.model_file)
                logger.info("loaded Piper voice %s", self.model_file.stem)
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav:
                self._voice.synthesize_wav(text, wav, syn_config=SynthesisConfig(length_scale=self.config.voice_length_scale))
        if not self.config.lead_in_ms:
            return buf.getvalue()
        buf.seek(0)
        with wave.open(buf, "rb") as src:
            params, frames = src.getparams(), src.readframes(src.getnframes())
        out = io.BytesIO()
        with wave.open(out, "wb") as dst:
            dst.setparams(params)
            dst.writeframes(b"\x00" * (params.sampwidth * params.nchannels * params.framerate * self.config.lead_in_ms
                                       // 1000) + frames)
        return out.getvalue()

    async def synthesize(self, text: str) -> bytes:
        if not self.config.enabled:
            raise SpeechUnavailableError("Speech is disabled (speech.enabled: false)")
        if not self.installed:
            raise SpeechUnavailableError(
                f"No Piper voice at {self.path}. Download one, e.g. en_US-lessac-medium.onnx and its .onnx.json from "
                "huggingface.co/rhasspy/piper-voices, into that folder"
            )
        text = text.strip()
        if not text:
            raise InvalidRequestError("Nothing to say")
        if len(text) > self.config.max_speak_chars:
            raise InvalidRequestError(f"Text exceeds {self.config.max_speak_chars} characters")
        return await asyncio.to_thread(self._synthesize, text)
