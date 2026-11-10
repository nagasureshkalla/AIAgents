// Floating rune cubes: one glowing, slowly tumbling cube per agent, each in its own colour.
// Faces are drawn procedurally (glowing glyphs on a dark tinted face). Drop an image at
// ui/assets/agent-cube.png to use it as the face texture instead; it is tinted per agent.
import * as THREE from 'three';

const FACE = 256;
const FACE_IMAGE_URL = './assets/agent-cube.png';

// Deterministic pseudo-random numbers so each agent keeps the same glyphs between reloads.
function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => ((s = (s * 1664525 + 1013904223) >>> 0) / 4294967296);
}
function hash(text) {
  let h = 2166136261;
  for (const ch of text) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return h >>> 0;
}

// One rune glyph made of strokes, arcs and dots inside a cell of size `s` centred at (x, y).
function drawGlyph(ctx, x, y, s, rand) {
  ctx.beginPath();
  const pts = [];
  for (let i = 0; i < 3 + Math.floor(rand() * 3); i++) {
    pts.push([x + (Math.round(rand() * 4) / 4 - 0.5) * s, y + (Math.round(rand() * 4) / 4 - 0.5) * s]);
  }
  ctx.moveTo(...pts[0]);
  for (const p of pts.slice(1)) ctx.lineTo(...p);
  if (rand() < 0.5) {
    ctx.moveTo(x + s * 0.32, y);
    ctx.arc(x, y, s * 0.32, 0, Math.PI * (0.6 + rand() * 1.4));
  }
  ctx.stroke();
  if (rand() < 0.3) {
    ctx.beginPath();
    ctx.arc(x + (rand() - 0.5) * s * 0.6, y + (rand() - 0.5) * s * 0.6, s * 0.06, 0, Math.PI * 2);
    ctx.fill();
  }
}

function runeFace(name, tint) {
  const c = document.createElement('canvas');
  c.width = c.height = FACE;
  const ctx = c.getContext('2d');
  const base = tint.clone();
  const dark = base.clone().multiplyScalar(0.28);
  const g = ctx.createRadialGradient(FACE / 2, FACE / 2, FACE * 0.1, FACE / 2, FACE / 2, FACE * 0.75);
  g.addColorStop(0, `#${base.clone().multiplyScalar(0.55).getHexString()}`);
  g.addColorStop(1, `#${dark.getHexString()}`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, FACE, FACE);

  // Bright inner frame.
  const glow = `#${base.clone().lerp(new THREE.Color('#ffffff'), 0.55).getHexString()}`;
  ctx.strokeStyle = glow;
  ctx.fillStyle = glow;
  ctx.shadowColor = glow;
  ctx.shadowBlur = 14;
  ctx.lineWidth = 6;
  ctx.strokeRect(14, 14, FACE - 28, FACE - 28);

  // 3x3 grid of glyphs.
  const rand = rng(hash(name));
  ctx.lineWidth = 7;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  const cell = (FACE - 56) / 3;
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      if (rand() < 0.22) continue;
      drawGlyph(ctx, 28 + cell * (i + 0.5), 28 + cell * (j + 0.5), cell * 0.62, rand);
    }
  }
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.anisotropy = 4;
  return tex;
}

// Loaded once and shared by every cube; null until (and unless) the optional image exists.
let faceImage = null;
const waiting = new Set();
new THREE.TextureLoader().load(
  FACE_IMAGE_URL,
  (tex) => {
    tex.colorSpace = THREE.SRGBColorSpace;
    faceImage = tex;
    for (const cube of waiting) cube.useImage(tex);
    waiting.clear();
  },
  undefined,
  () => waiting.clear(), // no custom image: keep the procedural runes
);

export class RuneCube {
  constructor(name, tint) {
    this.tint = tint.clone();
    this.phase = (hash(name) % 1000) / 100;
    this.object = new THREE.Group();

    this.runes = runeFace(name, tint);
    this.material = new THREE.MeshStandardMaterial({
      map: this.runes,
      emissive: new THREE.Color('#ffffff'),
      emissiveMap: this.runes,
      emissiveIntensity: 0.55,
      roughness: 0.35,
      metalness: 0.15,
    });
    this.cube = new THREE.Mesh(new THREE.BoxGeometry(1.7, 1.7, 1.7), this.material);
    this.object.add(this.cube);

    // Glowing edges plus a faint outer shell give the "energy cube" look.
    this.edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(this.cube.geometry),
      new THREE.LineBasicMaterial({ color: tint.clone().lerp(new THREE.Color('#ffffff'), 0.5), transparent: true }),
    );
    this.cube.add(this.edges);
    this.shell = new THREE.Mesh(
      new THREE.BoxGeometry(2.05, 2.05, 2.05),
      new THREE.MeshBasicMaterial({ color: tint, transparent: true, opacity: 0.12, depthWrite: false }),
    );
    this.cube.add(this.shell);

    this.fx = null;
    this.spin = 0;
    if (faceImage) this.useImage(faceImage);
    else waiting.add(this);
  }

  useImage(tex) {
    this.material.map = tex;
    this.material.emissiveMap = tex;
    this.material.color.copy(this.tint).lerp(new THREE.Color('#ffffff'), 0.35); // tint the shared image per agent
    this.material.emissive.copy(this.tint);
    this.material.needsUpdate = true;
  }

  /** One-off flourish: 'tip' (quick flip on success) or 'knock' (shake on failure). */
  trigger(kind) {
    this.fx = { kind, t: 0 };
  }

  /** `stateColor` drives the edge glow; the cube itself keeps the agent's colour. */
  update(dt, t, state, stateColor) {
    const running = state === 'running';
    const ph = this.phase;
    this.spin += dt * (running ? 2.4 : 0.35);
    this.cube.rotation.set(0.45 + 0.15 * Math.sin(t * 0.6 + ph), this.spin, 0.2 * Math.sin(t * 0.4 + ph));
    this.object.position.y = 0.22 * Math.sin(t * 1.4 + ph) + (running ? 0.15 * Math.abs(Math.sin(t * 6 + ph)) : 0);

    const pulse = running ? 0.5 + 0.5 * Math.sin(t * 6 + ph) : 0;
    this.material.emissiveIntensity = (state === 'idle' ? 0.5 : 0.7) + 0.6 * pulse;
    this.edges.material.color.copy(state === 'idle' ? this.tint : stateColor).lerp(new THREE.Color('#ffffff'), 0.35);
    this.shell.material.color.copy(state === 'idle' ? this.tint : stateColor);
    this.shell.material.opacity = 0.1 + 0.18 * pulse + (state === 'failed' ? 0.12 : 0);
    let scale = 1 + 0.06 * pulse;

    if (this.fx) {
      this.fx.t += dt;
      const k = Math.min(1, this.fx.t / 0.9);
      if (this.fx.kind === 'tip') {
        this.cube.rotation.x += Math.PI * 2 * (1 - Math.pow(1 - k, 3)); // one full flip, easing out
        scale += 0.15 * Math.sin(Math.PI * k);
      } else {
        this.object.position.x = 0.25 * Math.sin(this.fx.t * 40) * Math.exp(-this.fx.t * 4);
      }
      if (this.fx.t > 1.2) {
        this.fx = null;
        this.object.position.x = 0;
      }
    }
    this.cube.scale.setScalar(scale);
  }
}
