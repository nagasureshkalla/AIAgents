// AudioWorklet: forwards raw microphone samples to the page (voice.js does the voice activity detection).
class PcmTap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0]?.[0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor('pcm-tap', PcmTap);
