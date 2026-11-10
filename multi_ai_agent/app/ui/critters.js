// Cartoon animal mascots in black top hats, drawn procedurally on a canvas and shown as a camera-facing sprite.
import * as THREE from 'three';

const SIZE = 256;
const FPS = 24;
const INK = '#1e293b';
const HAT = '#121212';

// One parameter set per species; the drawing code below is shared.
const ANIMALS = {
  penguin: { main: '#334155', belly: '#f8fafc', face: '#f8fafc', feet: '#f59e0b', ears: 'none', snout: 'beak', beak: '#f59e0b' },
  bear: { main: '#8b5a2b', belly: '#d9b38c', feet: '#6b4423', ears: 'round', snout: 'muzzle', muzzle: '#e8cfa8' },
  owl: { main: '#8d6e63', belly: '#d7ccc8', face: '#efebe9', feet: '#f59e0b', ears: 'tufts', snout: 'beak', beak: '#f59e0b', bigEyes: true },
  parrot: { main: '#22c55e', belly: '#fde047', feet: '#64748b', ears: 'none', snout: 'beak', beak: '#f97316', cheeks: '#ef4444' },
  cat: { main: '#f59e0b', belly: '#fff7ed', feet: '#fff7ed', ears: 'pointy', snout: 'nose', nose: '#f472b6', whiskers: true, bigEyes: true },
  fox: { main: '#ea580c', belly: '#fff7ed', feet: '#431407', ears: 'pointy', snout: 'muzzle', muzzle: '#fff7ed', tail: true },
  raccoon: { main: '#9ca3af', belly: '#e5e7eb', feet: '#374151', ears: 'round', snout: 'muzzle', muzzle: '#f3f4f6', mask: '#374151' },
  frog: { main: '#4ade80', belly: '#dcfce7', feet: '#22c55e', ears: 'none', snout: 'none', frogEyes: true, hatY: -16, cheeks: '#f9a8d4' },
  squirrel: { main: '#b45309', belly: '#fde68a', feet: '#78350f', ears: 'pointy', snout: 'nose', nose: '#451a03', tail: true, cheeks: '#fca5a5' },
};

const AGENT_ANIMALS = {
  user: 'penguin', manager: 'bear', planner: 'owl', text: 'parrot', vision: 'cat',
  multimodal: 'fox', reasoning: 'raccoon', validation: 'frog', aggregator: 'squirrel',
};

function animalFor(name) {
  if (AGENT_ANIMALS[name]) return ANIMALS[AGENT_ANIMALS[name]];
  const keys = Object.keys(ANIMALS);
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) >>> 0;
  return ANIMALS[keys[h % keys.length]];
}

// Fill + cartoon outline for whatever path `build` traces.
function blob(ctx, fill, build, line = 4) {
  ctx.beginPath();
  build();
  ctx.fillStyle = fill;
  ctx.fill();
  if (line) {
    ctx.lineWidth = line;
    ctx.strokeStyle = INK;
    ctx.stroke();
  }
}

const oval = (ctx, x, y, rx, ry, rot = 0) => () => ctx.ellipse(x, y, rx, ry, rot, 0, Math.PI * 2);

export class Critter {
  constructor(name, tint) {
    this.a = animalFor(name);
    this.tint = tint;
    this.phase = Math.random() * 10;
    this.canvas = document.createElement('canvas');
    this.canvas.width = this.canvas.height = SIZE;
    this.ctx = this.canvas.getContext('2d');
    this.texture = new THREE.CanvasTexture(this.canvas);
    this.texture.colorSpace = THREE.SRGBColorSpace;
    this.sprite = new THREE.Sprite(new THREE.SpriteMaterial({ map: this.texture, transparent: true, alphaTest: 0.05 }));
    this.sprite.renderOrder = 1; // draw over the glow halo sharing the same position
    this.acc = Infinity;
    this.nextBlink = 1 + Math.random() * 3;
    this.blinkT = 1;
    this.fx = null;
  }

  /** One-off flourish: 'tip' (hat tip on success) or 'knock' (hat knocked askew on failure). */
  trigger(kind) {
    this.fx = { kind, t: 0 };
  }

  update(dt, t, state, color) {
    this.nextBlink -= dt;
    if (this.nextBlink <= 0) {
      this.blinkT = 0;
      this.nextBlink = 3 + Math.random() * 2;
    }
    this.blinkT += dt;
    if (this.fx) {
      this.fx.t += dt;
      if (this.fx.t > 1.6) this.fx = null;
    }
    this.acc += dt;
    if (this.acc < 1 / FPS) return;
    this.acc = 0;
    this._draw(this._pose(t, state), '#' + color.getHexString());
    this.texture.needsUpdate = true;
  }

  _pose(t, state) {
    const ph = this.phase;
    const breath = Math.sin(t * 2 + ph);
    const p = {
      bob: 0, sx: 1 - 0.02 * breath, sy: 1 + 0.025 * breath, shake: 0,
      hatLift: 0, hatRot: 0.04 * Math.sin(t * 1.3 + ph), hatX: 0,
      eyeX: 2 * Math.sin(t * 0.7 + ph), eyes: this.blinkT < 0.12 ? 'closed' : 'open', mouth: 'smile',
      lines: false, sweat: false, tailWag: 0.08 * Math.sin(t * 1.5 + ph),
    };
    if (state === 'running') {
      const h = Math.abs(Math.sin(t * 7 + ph));
      const g = Math.pow(1 - h, 3);
      p.bob = h * 16;
      p.sx = 1 + 0.1 * g;
      p.sy = 1 - 0.12 * g + 0.05 * h;
      p.hatLift = 4 + 8 * Math.abs(Math.sin(t * 7 + ph - 0.5)); // trails the body's hop
      p.hatRot = 0.12 * Math.sin(t * 7 + ph);
      p.eyeX = 4 * Math.sin(t * 3 + ph);
      p.mouth = 'o';
      p.lines = true;
      p.tailWag = 0.35 * Math.sin(t * 14 + ph);
    } else if (state === 'done') {
      p.eyes = 'happy';
      p.mouth = 'grin';
    } else if (state === 'failed') {
      p.eyes = 'x';
      p.mouth = 'frown';
      p.hatRot = -0.45;
      p.hatX = -10;
      p.hatLift = -4;
      p.sweat = true;
      p.eyeX = 0;
    }
    if (this.fx) {
      const k = Math.min(1, this.fx.t);
      const arc = Math.sin(Math.PI * k);
      if (this.fx.kind === 'tip') {
        p.hatLift += 22 * arc;
        p.hatRot -= 0.5 * arc;
        p.hatX += 14 * arc;
      } else if (this.fx.kind === 'knock') {
        p.hatLift += 26 * arc;
        p.hatRot += 0.6 * arc;
        p.shake = 8 * Math.sin(this.fx.t * 40) * Math.exp(-this.fx.t * 4);
      }
    }
    return p;
  }

  _draw(p, band) {
    const { ctx, a } = this;
    ctx.clearRect(0, 0, SIZE, SIZE);
    ctx.lineJoin = ctx.lineCap = 'round';

    // Ground shadow shrinks while airborne.
    ctx.fillStyle = 'rgba(15,23,42,0.18)';
    ctx.beginPath();
    ctx.ellipse(128, 244, 40 * (1 - p.bob / 40), 6, 0, 0, Math.PI * 2);
    ctx.fill();

    if (p.lines) {
      ctx.strokeStyle = 'rgba(30,41,59,0.35)';
      ctx.lineWidth = 3;
      for (const [y, w] of [[150, 22], [175, 30], [200, 18]]) {
        ctx.beginPath();
        ctx.moveTo(30, y - p.bob);
        ctx.lineTo(30 + w, y - p.bob);
        ctx.stroke();
      }
    }

    ctx.save();
    ctx.translate(128 + p.shake, 240 - p.bob);
    ctx.scale(0.92 * p.sx, 0.92 * p.sy); // headroom so a hopping, hat-lifting critter stays inside the canvas

    if (a.tail) {
      ctx.save();
      ctx.translate(36, -40);
      ctx.rotate(p.tailWag);
      blob(ctx, a.main, () => {
        ctx.moveTo(0, 0);
        ctx.bezierCurveTo(40, 10, 60, -40, 40, -80);
        ctx.bezierCurveTo(30, -95, 10, -80, 18, -60);
        ctx.bezierCurveTo(26, -40, 10, -20, 0, -14);
        ctx.closePath();
      });
      ctx.restore();
    }

    // Feet, body, belly, arms.
    blob(ctx, a.feet, oval(ctx, -20, -4, 15, 7));
    blob(ctx, a.feet, oval(ctx, 20, -4, 15, 7));
    blob(ctx, a.main, oval(ctx, 0, -46, 46, 44));
    blob(ctx, a.belly, oval(ctx, 0, -40, 30, 31), 0);
    blob(ctx, a.main, oval(ctx, -44, -50, 9, 20, 0.35));
    blob(ctx, a.main, oval(ctx, 44, -50, 9, 20, -0.35));

    // Ears sit behind the head.
    if (a.ears === 'round') {
      for (const s of [-1, 1]) {
        blob(ctx, a.main, oval(ctx, 34 * s, -142, 15, 15));
        blob(ctx, a.belly, oval(ctx, 34 * s, -142, 7, 7), 0);
      }
    } else if (a.ears === 'pointy' || a.ears === 'tufts') {
      const tall = a.ears === 'pointy' ? 40 : 26;
      for (const s of [-1, 1]) {
        blob(ctx, a.main, () => {
          ctx.moveTo(16 * s, -146);
          ctx.lineTo(42 * s, -146 - tall);
          ctx.lineTo(44 * s, -122);
          ctx.closePath();
        });
      }
    }
    if (a.frogEyes) {
      blob(ctx, a.main, oval(ctx, -20, -142, 17, 16));
      blob(ctx, a.main, oval(ctx, 20, -142, 17, 16));
    }

    // Head and face.
    blob(ctx, a.main, oval(ctx, 0, -112, 44, 42));
    if (a.face) blob(ctx, a.face, oval(ctx, 0, -106, 34, 30), 0);
    if (a.mask) blob(ctx, a.mask, oval(ctx, 0, -118, 38, 13), 0);
    if (a.cheeks) {
      ctx.globalAlpha = 0.55;
      blob(ctx, a.cheeks, oval(ctx, -28, -98, 8, 5), 0);
      blob(ctx, a.cheeks, oval(ctx, 28, -98, 8, 5), 0);
      ctx.globalAlpha = 1;
    }
    this._eyes(p, a.frogEyes ? -140 : -118, a.bigEyes ? 13 : 10);
    this._snout(p);

    if (p.sweat) {
      blob(ctx, '#7dd3fc', () => {
        ctx.moveTo(46, -146);
        ctx.quadraticCurveTo(38, -128, 46, -126);
        ctx.quadraticCurveTo(54, -128, 46, -146);
      }, 2);
    }

    this._hat(p, band);
    ctx.restore();
  }

  _eyes(p, y, r) {
    const ctx = this.ctx;
    ctx.strokeStyle = this.a.mask ? '#f8fafc' : INK; // line eyes must show on a dark mask
    ctx.lineWidth = 4;
    for (const s of [-1, 1]) {
      const x = 17 * s;
      if (p.eyes === 'open') {
        blob(ctx, '#ffffff', oval(ctx, x, y, r, r + 1), 3);
        blob(ctx, INK, oval(ctx, x + p.eyeX, y + 1, r * 0.5, r * 0.55), 0);
        blob(ctx, '#ffffff', oval(ctx, x + p.eyeX - 2, y - 2, r * 0.16, r * 0.16), 0);
      } else if (p.eyes === 'closed') {
        ctx.beginPath();
        ctx.moveTo(x - r, y);
        ctx.lineTo(x + r, y);
        ctx.stroke();
      } else if (p.eyes === 'happy') {
        ctx.beginPath();
        ctx.arc(x, y + 4, r * 0.8, Math.PI * 1.1, Math.PI * 1.9);
        ctx.stroke();
      } else {
        const d = r * 0.6;
        ctx.beginPath();
        ctx.moveTo(x - d, y - d); ctx.lineTo(x + d, y + d);
        ctx.moveTo(x + d, y - d); ctx.lineTo(x - d, y + d);
        ctx.stroke();
      }
    }
  }

  _snout(p) {
    const { ctx, a } = this;
    let mouthY = -88;
    if (a.snout === 'beak') {
      blob(ctx, a.beak, () => {
        ctx.moveTo(-10, -102);
        ctx.lineTo(10, -102);
        ctx.lineTo(0, -86);
        ctx.closePath();
      }, 3);
      if (p.mouth === 'o') blob(ctx, '#7f1d1d', oval(ctx, 0, -84, 4, 3), 0);
      return;
    }
    if (a.snout === 'muzzle') {
      blob(ctx, a.muzzle, oval(ctx, 0, -96, 18, 13), 3);
      blob(ctx, INK, oval(ctx, 0, -102, 6, 4), 0);
      mouthY = -92;
    } else if (a.snout === 'nose') {
      blob(ctx, a.nose, oval(ctx, 0, -100, 5, 4), 2);
      mouthY = -94;
    } else {
      mouthY = -94;
    }
    if (a.whiskers) {
      ctx.strokeStyle = INK;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const s of [-1, 1]) {
        ctx.moveTo(12 * s, -98); ctx.lineTo(38 * s, -104);
        ctx.moveTo(12 * s, -94); ctx.lineTo(38 * s, -92);
      }
      ctx.stroke();
    }
    const wide = a.frogEyes ? 1.8 : 1;
    ctx.strokeStyle = INK;
    ctx.lineWidth = 3;
    ctx.beginPath();
    if (p.mouth === 'o') {
      blob(ctx, '#7f1d1d', oval(ctx, 0, mouthY + 4, 5 * wide, 5), 2);
      return;
    } else if (p.mouth === 'grin') {
      ctx.arc(0, mouthY - 2, 9 * wide, 0.15 * Math.PI, 0.85 * Math.PI);
    } else if (p.mouth === 'frown') {
      ctx.arc(0, mouthY + 10, 7 * wide, 1.2 * Math.PI, 1.8 * Math.PI);
    } else {
      ctx.arc(0, mouthY - 2, 6 * wide, 0.25 * Math.PI, 0.75 * Math.PI);
    }
    ctx.stroke();
  }

  _hat(p, band) {
    const ctx = this.ctx;
    ctx.save();
    ctx.translate(p.hatX, -148 + (this.a.hatY || 0) - p.hatLift);
    ctx.rotate(-0.1 + p.hatRot); // a jaunty default tilt
    blob(ctx, HAT, () => {
      ctx.moveTo(-24, 0);
      ctx.lineTo(-27, -46);
      ctx.quadraticCurveTo(0, -52, 27, -46);
      ctx.lineTo(24, 0);
      ctx.closePath();
    }, 3);
    blob(ctx, band, () => ctx.rect(-25, -14, 50, 9), 0);
    ctx.fillStyle = 'rgba(255,255,255,0.14)';
    ctx.fillRect(-19, -44, 5, 28);
    blob(ctx, HAT, oval(ctx, 0, 0, 40, 7), 3);
    ctx.restore();
  }

  dispose() {
    this.texture.dispose();
    this.sprite.material.dispose();
  }
}
