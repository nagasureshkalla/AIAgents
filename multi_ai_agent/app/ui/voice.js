// Speech agent, browser side. The microphone is read through an AudioWorklet; a small energy-based voice activity
// detector cuts it into utterances, which are sent as 16 kHz WAV to the server's local Whisper model
// (/api/speech/listen or /transcribe). Replies are spoken by the server's local Piper voice (/api/speech/speak),
// falling back to the browser's on-device voices (never network voices). Nothing leaves this computer.

const RATE = 16_000;
const BLOCK_MS = 30;
const PREROLL_MS = 450;        // audio kept from just before speech started, so the first word is not clipped
const START_BLOCKS = 3;        // ~90 ms above the threshold counts as speech
const MIN_SPEECH_MS = 300;
const MAX_UTTERANCE_MS = 60_000;   // a spoken request with pauses to think; longer means the room never went quiet
const MIN_THRESHOLD = 0.015;

// Listens to the microphone and calls onUtterance(wavBlob) each time you finish saying something.
export class VoiceListener {
  constructor({ onUtterance, onLevel, onSpeech, silenceMs = 900, once = false, noSpeechMs = 0 } = {}) {
    Object.assign(this, { onUtterance, onLevel, onSpeech, silenceMs, once, noSpeechMs });
    this.muted = false;
    this.running = false;
  }

  async start() {
    this.stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true, channelCount: 1 },
    });
    this.deviceLabel = this.stream.getAudioTracks()[0]?.label || '';
    this.ctx = new AudioContext();
    await this.ctx.audioWorklet.addModule(new URL('./pcm-worklet.js', import.meta.url));
    const src = this.ctx.createMediaStreamSource(this.stream);
    this.node = new AudioWorkletNode(this.ctx, 'pcm-tap');
    const sink = this.ctx.createGain();
    sink.gain.value = 0; // the worklet must reach the destination to run; keep it silent
    src.connect(this.node).connect(sink).connect(this.ctx.destination);
    this.blockSize = Math.round((this.ctx.sampleRate * BLOCK_MS) / 1000);
    this._reset();
    this.noise = 0.004;
    this.startedAt = performance.now();
    this.heardAnything = false;
    this.node.port.onmessage = (e) => this._samples(e.data);
    this.running = true;
    if (this.ctx.state === 'suspended') await this.ctx.resume();
    return this.ctx.state === 'running'; // false: the browser wants a click on the page first
  }

  resume() { return this.ctx?.resume(); }

  stop(flush = false) {
    if (!this.running) return;
    this.running = false;
    if (flush && this.speaking) this._end();
    this.node?.port.close();
    this.stream?.getTracks().forEach((t) => t.stop());
    this.ctx?.close();
    this.onLevel?.(0);
  }

  // While muted (the reply is being spoken) the microphone is ignored, so the agent does not hear itself.
  mute(on) {
    this.muted = on;
    if (on) this._reset();
  }

  _reset() {
    this.pending = new Float32Array(0);
    this.preroll = [];
    this.segment = [];
    this.speaking = false;
    this.loud = 0;
    this.quietMs = 0;
    this.voicedMs = 0;
    this.peak = 0;
  }

  _samples(chunk) {
    const merged = new Float32Array(this.pending.length + chunk.length);
    merged.set(this.pending);
    merged.set(chunk, this.pending.length);
    let i = 0;
    for (; i + this.blockSize <= merged.length; i += this.blockSize) this._block(merged.slice(i, i + this.blockSize));
    this.pending = merged.slice(i);
  }

  _block(block) {
    if (!this.running) return;
    if (this.once && this.noSpeechMs && !this.heardAnything && performance.now() - this.startedAt > this.noSpeechMs) {
      this.stop();
      this.onUtterance?.(null);
      return;
    }
    let sum = 0;
    for (const v of block) sum += v * v;
    const rms = Math.sqrt(sum / block.length);
    this.onLevel?.(this.muted ? 0 : Math.min(1, rms * 12));
    if (this.muted) return;
    const threshold = Math.max(MIN_THRESHOLD, this.noise * 3.5);

    if (!this.speaking) {
      this.noise = Math.max(0.002, this.noise * 0.97 + rms * 0.03); // follows the room's background level
      this.preroll.push(block);
      if (this.preroll.length * BLOCK_MS > PREROLL_MS) this.preroll.shift();
      this.loud = rms > threshold ? this.loud + 1 : 0;
      if (this.loud >= START_BLOCKS) {
        this.speaking = true;
        this.heardAnything = true;
        this.segment = [...this.preroll];
        this.voicedMs = this.loud * BLOCK_MS;
        this.peak = rms;
        this.quietMs = 0;
        this.onSpeech?.(true);
      }
      return;
    }
    this.segment.push(block);
    // "Quiet" is relative to how loud you are speaking, not just to the room: a Bluetooth mic with automatic gain
    // keeps its hiss above any fixed level, so a fixed cut-off never ends the utterance.
    this.peak = Math.max(rms, this.peak * 0.999);  // slow decay: still "you" after a few seconds of thinking
    // ... and above the room's own noise, measured before you started (frozen while you talk)
    const quiet = Math.max(threshold * 0.6, this.peak * 0.3, this.noise * 1.6);
    if (rms > quiet) { this.quietMs = 0; this.voicedMs += BLOCK_MS; } else this.quietMs += BLOCK_MS;
    if (this.quietMs >= this.silenceMs || this.segment.length * BLOCK_MS >= MAX_UTTERANCE_MS) this._end();
  }

  _end() {
    const keep = this.voicedMs >= MIN_SPEECH_MS;
    const wav = keep ? encodeWav(this.segment, this.ctx.sampleRate) : null;
    this._reset();
    this.onSpeech?.(false);
    if (this.once) {
      if (!keep) return; // a click or a cough: keep waiting for real speech
      this.stop();
    }
    if (wav) this.onUtterance?.(wav);
  }
}

// Float32 blocks at the AudioContext rate -> 16 kHz 16-bit mono WAV.
function encodeWav(blocks, rate) {
  const total = blocks.reduce((n, b) => n + b.length, 0);
  const ratio = rate / RATE;
  const out = new Int16Array(Math.floor(total / ratio));
  const flat = new Float32Array(total);
  let o = 0;
  for (const b of blocks) { flat.set(b, o); o += b.length; }
  for (let i = 0; i < out.length; i++) {
    const a = Math.floor(i * ratio), z = Math.min(total, Math.floor((i + 1) * ratio));
    let s = 0;
    for (let j = a; j < z; j++) s += flat[j];
    const v = Math.max(-1, Math.min(1, s / Math.max(1, z - a)));
    out[i] = v < 0 ? v * 0x8000 : v * 0x7fff;
  }
  const buf = new ArrayBuffer(44 + out.length * 2);
  const dv = new DataView(buf);
  const str = (off, s) => [...s].forEach((c, k) => dv.setUint8(off + k, c.charCodeAt(0)));
  str(0, 'RIFF'); dv.setUint32(4, 36 + out.length * 2, true); str(8, 'WAVE');
  str(12, 'fmt '); dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
  dv.setUint32(24, RATE, true); dv.setUint32(28, RATE * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
  str(36, 'data'); dv.setUint32(40, out.length * 2, true);
  new Int16Array(buf, 44).set(out);
  return new Blob([buf], { type: 'audio/wav' });
}

// A "Monitor of …" input records the speakers, not a microphone: what Linux falls back to when a Bluetooth
// headset is in its A2DP (music) profile, which has no microphone.
export const isMonitorInput = (label) => /monitor/i.test(label || '');

// Push-to-talk: records one utterance. Resolves with a WAV Blob, or null if nothing was said.
export function recordUtterance({ onLevel, signal } = {}) {
  return new Promise((resolve, reject) => {
    const l = new VoiceListener({ once: true, silenceMs: 1500, noSpeechMs: 8000, onLevel, onUtterance: resolve });
    signal?.addEventListener('abort', () => { l.stop(true); resolve(null); }, { once: true });
    l.start().catch(reject);
  });
}

async function postAudio(path, blob, fields = {}) {
  const fd = new FormData();
  fd.append('file', blob, 'voice.wav');
  for (const [k, v] of Object.entries(fields)) fd.append(k, v);
  const res = await fetch(path, { method: 'POST', body: fd });
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.message || body.detail || `HTTP ${res.status}`);
  return body;
}
export const transcribe = async (blob) => (await postAudio('/api/speech/transcribe', blob)).text || '';
// -> {text, woke, command}
export const listen = (blob, followUp = false) => postAudio('/api/speech/listen', blob, { follow_up: String(followUp) });

// Markdown answer -> something pleasant to listen to.
export function speechText(markdown) {
  return markdown
    .split(/\n#{1,6}\s*Sources\b/)[0]
    .replace(/```[\s\S]*?```/g, ' (code shown on screen) ')
    .replace(/`([^`]+)`/g, '$1')
    .replace(/!\[[^\]]*\]\([^)]*\)/g, '')
    .replace(/\[([^\]]+)\]\([^)]*\)/g, '$1')
    .replace(/\[\d+\]/g, '')
    .replace(/https?:\/\/\S+/g, 'link')
    .replace(/^\s*#{1,6}\s*/gm, '')
    .replace(/^\s*[-*+]\s+/gm, '')
    .replace(/[*_~]{1,3}([^*_~]+)[*_~]{1,3}/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

function chunks(text, max = 260) {
  const out = [];
  for (const p of text.match(/[^.!?]+[.!?]+["')\]]*|[^.!?]+$/g) || [text]) {
    const last = out.length - 1;
    if (last >= 0 && out[last].length + p.length < max) out[last] += p;
    else out.push(p);
  }
  return out.map((c) => c.trim()).filter(Boolean);
}

// ---- speaking
let playing = null; // {cancelled, audio}
export const canSpeak = () => true;
export const isSpeaking = () => !!playing;
export function stopSpeaking() {
  if (playing) {
    playing.cancelled = true;
    playing.audio.pause();
    playing.resolve?.();
  }
  if ('speechSynthesis' in window) speechSynthesis.cancel();
}

async function piperChunk(text) {
  const res = await fetch('/api/speech/speak', {
    method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ text }),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).message || `HTTP ${res.status}`);
  return res.blob();
}

function playBlob(job, blob) {
  return new Promise((resolve) => {
    const url = URL.createObjectURL(blob);
    job.resolve = resolve;
    job.audio.src = url;
    job.audio.onended = job.audio.onerror = () => { URL.revokeObjectURL(url); resolve(); };
    job.audio.play().catch(() => resolve());
  });
}

function browserSpeak(job, parts) {
  const voices = 'speechSynthesis' in window ? speechSynthesis.getVoices().filter((v) => v.localService) : [];
  const lang = navigator.language || 'en-US';
  const voice = voices.find((v) => v.lang === lang) || voices.find((v) => v.lang?.startsWith(lang.slice(0, 2))) || voices[0];
  if (!voice) return Promise.resolve(false);
  return new Promise((resolve) => {
    job.resolve = () => resolve(true);
    parts.forEach((c, i) => {
      const u = new SpeechSynthesisUtterance(c);
      u.voice = voice;
      u.lang = voice.lang;
      if (i === parts.length - 1) u.onend = () => resolve(true);
      u.onerror = () => resolve(true);
      speechSynthesis.speak(u);
    });
  });
}
if ('speechSynthesis' in window) speechSynthesis.getVoices(); // the fallback voices load asynchronously

// Speaks text with Piper, sentence groups at a time (the next one is synthesized while this one plays).
// Resolves when finished or stopped; false if no voice could speak at all.
export async function speak(text) {
  stopSpeaking();
  const parts = chunks(text);
  if (!parts.length) return true;
  const job = { cancelled: false, audio: new Audio() };
  playing = job;
  try {
    let next = piperChunk(parts[0]);
    for (let i = 0; i < parts.length && !job.cancelled; i++) {
      let blob;
      try {
        blob = await next;
      } catch (err) {
        if (i > 0 || job.cancelled) throw err;
        return await browserSpeak(job, parts); // Piper missing: on-device browser voice
      }
      if (i + 1 < parts.length) { next = piperChunk(parts[i + 1]); next.catch(() => {}); }
      if (!job.cancelled) await playBlob(job, blob);
    }
    return true;
  } catch {
    return job.cancelled; // a later chunk failed: what was said so far stands
  } finally {
    if (playing === job) playing = null;
  }
}
