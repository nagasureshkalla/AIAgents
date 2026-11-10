from __future__ import annotations

import io
import math
import shutil
import struct
import wave

import pytest
from fastapi.testclient import TestClient

from app.agents.speech_agent import WakeMatch, match_wake
from app.main import create_app
from app.models.speech_model import SpeechToText, TextToSpeech
from tests.fakes import FakeFactory, build_container, make_settings

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg decodes the recordings")


def make_wav(seconds: float, rate: int = 22_050) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            struct.pack("<h", int(8000 * math.sin(2 * math.pi * 440 * i / rate))) for i in range(int(seconds * rate))
        ))
    return buf.getvalue()


def _client(tmp_path, *, installed: bool, wake_word: bool = True) -> TestClient:
    settings = make_settings(tmp_path)
    settings.speech.require_wake_word = wake_word
    model_dir = tmp_path / "models" / "speech" / "whisper"
    settings.speech.path = str(model_dir)
    settings.speech.voice = str(tmp_path / "models" / "speech" / "piper")
    if installed:
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"")
    factory = FakeFactory()
    return TestClient(create_app(settings, container_factory=lambda s: build_container(s, factory=factory)))


class Calls(list):
    """Recorded Whisper calls; tests queue what "Whisper" hears next in `heard`."""

    def __init__(self) -> None:
        super().__init__()
        self.heard: list[str] = []


@pytest.fixture
def fake_whisper(monkeypatch):
    calls = Calls()

    def load(self):
        def pipe(inputs, **kwargs):
            calls.append((inputs, kwargs))
            return {"text": calls.heard.pop(0) if calls.heard else "  open the calculator  "}

        self._pipe = pipe

    monkeypatch.setattr(SpeechToText, "_load", load)
    return calls


def test_status_and_missing_model(tmp_path):
    with _client(tmp_path, installed=False) as client:
        status = client.get("/api/speech").json()
        assert status["enabled"] and not status["installed"]
        r = client.post("/api/speech/transcribe", files={"file": ("voice.wav", make_wav(1), "audio/wav")})
        assert r.status_code == 503
        assert "download_models.py" in r.json()["message"]


def test_transcribes_recording_locally(tmp_path, fake_whisper):
    with _client(tmp_path, installed=True) as client:
        r = client.post("/api/speech/transcribe", files={"file": ("voice.wav", make_wav(1.5), "audio/wav")})
        assert r.status_code == 200, r.text
        assert r.json()["text"] == "open the calculator"
        assert r.json()["seconds"] == pytest.approx(1.5, abs=0.05)
        (inputs, kwargs), = fake_whisper
        assert inputs["sampling_rate"] == 16_000 and len(inputs["raw"]) == pytest.approx(24_000, abs=400)
        assert kwargs["generate_kwargs"] == {"task": "transcribe"}
        assert client.get("/api/speech").json()["loaded"]


def test_too_short_and_bad_audio(tmp_path, fake_whisper):
    with _client(tmp_path, installed=True) as client:
        r = client.post("/api/speech/transcribe", files={"file": ("voice.wav", make_wav(0.1), "audio/wav")})
        assert r.status_code == 200 and r.json()["text"] == ""
        assert not fake_whisper  # the model is not even loaded for a click
        r = client.post("/api/speech/transcribe", files={"file": ("voice.webm", b"not audio", "audio/webm")})
        assert r.status_code == 400


@pytest.mark.parametrize(
    ("text", "command"),
    [
        ("Hey CC, what is the capital of France?", "what is the capital of France?"),
        ("hey c.c. open the calculator", "open the calculator"),
        ("Hey, see see. Summarize this.", "Summarize this."),
        ("Hey CC at OpenCalculator.", "OpenCalculator."),  # how Whisper wrote "Hey CC, open calculator"
        ("ACC.", ""),                                        # how it wrote a bare "Hey C C"
        ("A C C", ""),
        ("Hey CC.", ""),
        ("um, hey cc, what's the time", "what's the time"),
        ("So I told her, hey CC, play some music", "play some music"),
    ],
)
def test_wake_word_variants(text, command):
    wake = match_wake(text)
    assert wake.woke, text
    assert wake.command == command


@pytest.mark.parametrize(
    "text",
    ["Please send a copy to the team.", "I see. See you later.", "Send a CC to John.", "Thank you.", "Okay.", "",
     "The ACC conference is next week."],
)
def test_no_wake_word(text):
    assert not match_wake(text).woke


def test_listen_only_acts_on_the_wake_word(tmp_path, fake_whisper):
    wav = make_wav(1.5)
    with _client(tmp_path, installed=True) as client:
        def listen(heard, follow_up=False):
            fake_whisper.heard.append(heard)
            r = client.post("/api/speech/listen", files={"file": ("u.wav", wav, "audio/wav")},
                            data={"follow_up": str(follow_up).lower()})
            assert r.status_code == 200, r.text
            return r.json()

        assert listen("I was telling my friend about the weather.")["woke"] is False
        r = listen("Hey CC, what is two plus two?")
        assert r["woke"] and r["command"] == "what is two plus two?"
        r = listen("Hey CC.")
        assert r["woke"] and r["command"] == ""
        r = listen("Open the calculator.", follow_up=True)  # the request after a bare "Hey CC"
        assert r["woke"] and r["command"] == "Open the calculator."
        r = listen("Hey CC, open the calculator.", follow_up=True)
        assert r["command"] == "open the calculator."
        status = client.get("/api/speech").json()
        assert status["wake_phrase"] == "Hey CC" and status["tts"]["installed"] is False


def test_silence_is_not_transcribed(tmp_path, fake_whisper):
    quiet = io.BytesIO()
    with wave.open(quiet, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16_000)
        w.writeframes(b"\x00\x00" * 32_000)
    with _client(tmp_path, installed=True) as client:
        r = client.post("/api/speech/listen", files={"file": ("u.wav", quiet.getvalue(), "audio/wav")})
        assert r.json() == {"text": "", "seconds": 2.0, "woke": False, "command": ""}
        assert not fake_whisper  # Whisper would have "heard" "Thank you." in silence


def test_speak_with_piper(tmp_path, monkeypatch):
    voice_dir = tmp_path / "models" / "speech" / "piper" / "en"
    voice_dir.mkdir(parents=True)
    (voice_dir / "v.onnx").write_bytes(b"")
    (voice_dir / "v.onnx.json").write_text("{}")
    said = []
    monkeypatch.setattr(TextToSpeech, "_synthesize", lambda self, text: said.append(text) or make_wav(0.2))
    with _client(tmp_path, installed=False) as client:
        assert client.get("/api/speech").json()["tts"] == {"installed": True, "loaded": False, "voice": "v"}
        r = client.post("/api/speech/speak", json={"text": " Hello there. "})
        assert r.status_code == 200 and r.headers["content-type"] == "audio/wav"
        assert r.content[:4] == b"RIFF" and said == ["Hello there."]
        assert client.post("/api/speech/speak", json={"text": "x" * 5000}).status_code == 400


def test_speak_without_voice(tmp_path):
    with _client(tmp_path, installed=False) as client:
        r = client.post("/api/speech/speak", json={"text": "Hello"})
        assert r.status_code == 503 and "piper-voices" in r.json()["message"]


def test_speech_agent_is_not_routed_to(tmp_path):
    with _client(tmp_path, installed=False) as client:
        agents = {a["name"]: a for a in client.get("/api/agents").json()}
        assert agents["speech"]["role"] == "control"
        assert "wake_word" in agents["speech"]["capabilities"]


@pytest.mark.parametrize(("text", "command"), [
    ("Hey Casey, what time is it?", "what time is it?"),   # Bluetooth-mic transcripts of "Hey CC"
    ("Hey KC open calculator", "open calculator"),
    ("Hey, Kacey. Tell me a joke", "Tell me a joke"),
])
def test_bluetooth_wake_spellings(text, command):
    assert match_wake(text) == WakeMatch(True, command)


@pytest.mark.parametrize("text", ["I talked to Casey yesterday", "KC and the Sunshine Band"])
def test_names_alone_do_not_wake(text):
    assert not match_wake(text).woke


def test_hey_sissi_wakes():
    assert match_wake("Hey Sissi, what time is it?") == WakeMatch(True, "what time is it?")


@pytest.mark.parametrize(("score", "woke", "command"), [
    (-0.8, True, "open the calculator."),   # the audio says "Hey CC": Whisper only dropped the name from the text
    (-8.5, False, ""),                     # it really was just "Hey, ..."
])
def test_name_lost_in_transcription_is_checked_acoustically(tmp_path, fake_whisper, monkeypatch, score, woke, command):
    scored = []
    monkeypatch.setattr(SpeechToText, "_wake_score", lambda self, audio, phrase: scored.append(phrase) or score)
    with _client(tmp_path, installed=True) as client:
        fake_whisper.heard.append("Hey, open the calculator.")
        r = client.post("/api/speech/listen", files={"file": ("u.wav", make_wav(1.5), "audio/wav")}).json()
        assert (r["woke"], r["command"], r["wake_score"]) == (woke, command, score)
        assert scored == [" Hey CC"]
        fake_whisper.heard.append("What a nice day.")  # no greeting: not scored at all
        r = client.post("/api/speech/listen", files={"file": ("u.wav", make_wav(1.5), "audio/wav")}).json()
        assert not r["woke"] and "wake_score" not in r and len(scored) == 1


def test_repetition_loop_is_dropped(tmp_path, fake_whisper):
    with _client(tmp_path, installed=True) as client:
        fake_whisper.heard.append("CC, " * 300)
        r = client.post("/api/speech/listen", files={"file": ("u.wav", make_wav(1.5), "audio/wav")}).json()
        assert r["text"] == "" and not r["woke"]


def test_conversation_mode_sends_everything_real(tmp_path, fake_whisper, monkeypatch):
    scored = []
    monkeypatch.setattr(SpeechToText, "_wake_score", lambda self, audio, phrase: scored.append(1) or -9.0)
    with _client(tmp_path, installed=True, wake_word=False) as client:
        assert client.get("/api/speech").json()["require_wake_word"] is False

        def listen(heard):
            fake_whisper.heard.append(heard)
            return client.post("/api/speech/listen", files={"file": ("u.wav", make_wav(1.5), "audio/wav")}).json()

        r = listen("Open the calculator and add 2 and 3.")
        assert r["woke"] and r["command"] == "Open the calculator and add 2 and 3."
        r = listen("Hey CC, what time is it?")           # the old wake phrase still works, and is dropped
        assert r["command"] == "what time is it?"
        for noise in ("Thank you.", "Okay.", "Um.", "you", "Thanks for watching!"):
            assert not listen(noise)["woke"], noise
        assert not scored  # no acoustic wake check in conversation mode


@pytest.mark.parametrize(("text", "ok"), [("Close calculator", True), ("Thank you.", False), ("Okay", False),
                                           ("What is two plus two?", True), ("hmm", False)])
def test_is_request(text, ok):
    from app.agents.speech_agent import is_request

    assert is_request(text) is ok
