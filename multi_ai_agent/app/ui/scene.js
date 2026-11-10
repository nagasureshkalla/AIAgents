// Three.js view of the agent pipeline: one floating rune cube per agent, arranged in 3D, with particles flowing
// along the edges a task uses. Only agents are drawn; the user and the result aggregator are not.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { RuneCube } from './cubes.js';

export const COLORS = {
  idle: new THREE.Color('#94a3b8'),
  running: new THREE.Color('#f59e0b'),
  done: new THREE.Color('#10b981'),
  failed: new THREE.Color('#ef4444'),
};

// Non-agent actors that appear in task events but are not drawn.
const NOT_AGENTS = new Set(['user', 'aggregator']);

// The pipeline runs along the x axis and spirals around it; parallel specialists sit on a ring around the
// axis (in the y-z plane), so the formation has real depth from every orbit angle.
const RING_X = 0;
const RING_Y = 2.0; // ring centre, lifted so its lowest agent still floats above the ground
const RING_R = 5.6;
const RING_ORDER = ['text', 'vision', 'multimodal', 'research'];
const RING_SLOTS = 8;
// Every agent floats above the ground. Add a model class here to stand an agent on the floor instead
// (it must provide object/update/trigger/height like RuneCube).
const GROUND_Y = -4.6;
const GROUNDED = {};
const GROUND_SCALE = 1.3;
const LAYOUT = {
  manager: [-11, 1.6, -1.8],
  planner: [-5.8, -1.2, 2.0],
  reasoning: [5.8, 1.4, -2.0],
  validation: [11, -1.4, 1.8],
  control_panel: [6.4, -2.4, 2.8],  // below reasoning, clear of the specialist ring from every orbit angle
  web_tester: [-9.6, -3.4, 3.4],    // lower left, in front of the manager: clear of the planner and the ring
  job_agent: [9.4, 5.2, 0.6],       // upper right, above reasoning/validation: clear of the ring and the manager
  speech: [-15.4, 4.4, 0.6],        // far left, before the manager: the voice front door
  airllm: [-5.6, 6.2, -2.4],        // upper left, above the planner: clear of the specialist ring
};
// Extra guide lines for agents outside BASE_EDGES, drawn once both ends exist.
const LINKS = [['speech', 'manager']];
function ringPosition(slot) {
  const a = Math.PI / 2 - (slot * 2 * Math.PI) / RING_SLOTS;
  return [RING_X, RING_Y + RING_R * Math.sin(a), RING_R * Math.cos(a)];
}
// Fog and idle-edge colors follow the page theme (--scene-fog / --scene-edge in styles.css).
const EDGE_COLOR = new THREE.Color('#b8c3d9');
const cssColor = (name, fallback) =>
  getComputedStyle(document.documentElement).getPropertyValue(name).trim() || fallback;

// Soft radial gradient used as a glow halo behind each node (bloom washes out on a light background).
// Radial dark blob for contact shadows.
function makeShadowTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(0,0,0,0.75)');
  g.addColorStop(0.5, 'rgba(0,0,0,0.3)');
  g.addColorStop(1, 'rgba(0,0,0,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

// Floor disc: the theme's floor colour, fading to transparent at the rim.
function makeFloorTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const ctx = c.getContext('2d');
  const col = new THREE.Color(cssColor('--scene-floor', '#e6ebf5'));
  const rgb = `${Math.round(col.r * 255)},${Math.round(col.g * 255)},${Math.round(col.b * 255)}`;
  const g = ctx.createRadialGradient(128, 128, 0, 128, 128, 128);
  g.addColorStop(0, `rgba(${rgb},0.85)`);
  g.addColorStop(0.65, `rgba(${rgb},0.45)`);
  g.addColorStop(1, `rgba(${rgb},0)`);
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 256, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

function makeHaloTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 128;
  const ctx = c.getContext('2d');
  const g = ctx.createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, 'rgba(255,255,255,1)');
  g.addColorStop(0.35, 'rgba(255,255,255,0.45)');
  g.addColorStop(1, 'rgba(255,255,255,0)');
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, 128, 128);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  return tex;
}

// Every agent cube has the same colour; state (running / done / failed) is shown by COLORS on top of it.
const AGENT_TINT = '#16a34a';
const DISPLAY_NAMES = { control_panel: 'AI DYOT', web_tester: 'Web Tester', job_agent: 'Job Agent', speech: 'Speech', airllm: 'AirLLM · 120B' }; // AI DYOT: the computer agent (computer_ai_agent service)
const tintFor = () => AGENT_TINT;

// The canonical flow drawn as faint guide lines between agents.
const BASE_EDGES = [
  ['manager', 'planner'],
  ['planner', 'text'], ['planner', 'vision'], ['planner', 'multimodal'],
  ['text', 'reasoning'], ['vision', 'reasoning'], ['multimodal', 'reasoning'],
  ['reasoning', 'validation'],
];

export class AgentScene {
  constructor(canvas, labelsEl) {
    this.canvas = canvas;
    this.labelsEl = labelsEl;
    this.nodes = new Map();
    this.edges = new Map();
    this.particles = [];
    this.emitters = new Map(); // agent -> {sources, color, acc}
    this.clock = new THREE.Clock();
    this.extraSlot = RING_ORDER.length;

    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.setClearColor(0x000000, 0); // transparent: the light CSS gradient shows through

    this.scene = new THREE.Scene();
    this.scene.fog = new THREE.Fog(0xf1f4fa, 30, 90);
    this._applyTheme();
    window.addEventListener('themechange', () => this._applyTheme()); // fired by app.js on OS change or toggle
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 400);
    this.camera.position.set(9, 6, 25);

    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.06;
    this.controls.minDistance = 8;
    this.controls.maxDistance = 70;
    this.controls.target.set(0, -0.6, 0);
    this.controls.autoRotate = true; // slow drift shows the depth; stops once the user grabs the view
    this.controls.autoRotateSpeed = 0.35;
    this.controls.addEventListener('start', () => { this.controls.autoRotate = false; });

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0xcfd8ea, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 1.4);
    key.position.set(-6, 12, 14);
    this.scene.add(key);
    this.haloTex = makeHaloTexture();

    this.shadowTex = makeShadowTexture();
    this._buildBackdrop();
    this._buildGround();

    this.particleGeo = new THREE.SphereGeometry(0.1, 10, 10);

    for (const name of ['manager', 'planner', 'text', 'vision', 'multimodal', 'reasoning', 'validation']) {
      this.ensureNode(name);
    }
    for (const [a, b] of BASE_EDGES) this._edge(a, b);

    new ResizeObserver(() => this._resize()).observe(canvas.parentElement);
    this._resize();
    this.renderer.setAnimationLoop(() => this._tick());
  }

  _buildGround() {
    // A subtle circular floor: soft disc + grid, both fading out towards the edge (and with the fog).
    const disc = new THREE.Mesh(
      new THREE.CircleGeometry(24, 96),
      new THREE.MeshBasicMaterial({ map: makeFloorTexture(), transparent: true, depthWrite: false }));
    disc.rotation.x = -Math.PI / 2;
    disc.position.y = GROUND_Y;
    disc.renderOrder = -3;
    this.floor = disc;
    this.grid = new THREE.GridHelper(44, 44, EDGE_COLOR, EDGE_COLOR);
    this.grid.position.y = GROUND_Y + 0.01;
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.35;
    this.grid.material.depthWrite = false;
    // fade the grid towards its rim
    this.grid.material.onBeforeCompile = (shader) => {
      shader.vertexShader = shader.vertexShader
        .replace('void main() {', 'varying float vR;\nvoid main() {')
        .replace('#include <begin_vertex>', '#include <begin_vertex>\nvR = length(position.xz) / 22.0;');
      shader.fragmentShader = shader.fragmentShader
        .replace('void main() {', 'varying float vR;\nvoid main() {')
        .replace('#include <opaque_fragment>', '#include <opaque_fragment>\ngl_FragColor.a *= clamp(1.0 - vR, 0.0, 1.0);');
    };
    this.scene.add(disc, this.grid);
  }

  _buildBackdrop() {
    const n = 1400;
    const pos = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) {
      const r = 40 + Math.random() * 60;
      const th = Math.random() * Math.PI * 2;
      const ph = Math.acos(2 * Math.random() - 1);
      pos.set([r * Math.sin(ph) * Math.cos(th), r * Math.sin(ph) * Math.sin(th), r * Math.cos(ph)], i * 3);
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    this.stars = new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xa9b6d3, size: 0.16, transparent: true, opacity: 0.5 }));
    this.scene.add(this.stars);

  }

  ensureNode(name, meta = {}) {
    if (NOT_AGENTS.has(name)) return null;
    if (this.nodes.has(name)) {
      Object.assign(this.nodes.get(name).meta, meta);
      return this.nodes.get(name);
    }
    let p = LAYOUT[name];
    if (!p) {
      const known = RING_ORDER.indexOf(name);
      p = ringPosition(known >= 0 ? known : this.extraSlot++ % RING_SLOTS);
    }
    const tint = new THREE.Color(tintFor(name));
    const group = new THREE.Group();
    group.position.set(...p);

    // Grounded agents (factory, bookshelf) have their origin on the floor; floating cubes are centred.
    const Model = GROUNDED[name];
    const grounded = !!Model;
    const cube = grounded ? new Model(name, tint) : new RuneCube(name, tint);
    group.add(cube.object);
    if (grounded) {
      cube.object.scale.setScalar(GROUND_SCALE);
      cube.height *= GROUND_SCALE;
    }
    const lift = grounded ? cube.height * 0.55 : 0; // where links attach and the halo glows

    const halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: this.haloTex, color: tint, transparent: true, opacity: 0.25, depthWrite: false,
    }));
    halo.scale.setScalar(grounded ? 5.2 : 3.6);
    halo.position.y = lift;
    halo.renderOrder = -1;
    group.add(halo);

    // While running: a spinning ring around a floating cube, or a glowing ring on the floor around a building.
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(grounded ? 2.9 : 1.75, grounded ? 0.05 : 0.04, 8, 96),
      new THREE.MeshBasicMaterial({ color: COLORS.running, transparent: true, opacity: 0 }));
    ring.rotation.x = grounded ? Math.PI / 2 : Math.PI / 2.4;
    ring.position.y = grounded ? 0.04 : 0;
    group.add(ring);

    const burst = new THREE.Mesh(
      new THREE.RingGeometry(0.9, 1.0, 64),
      new THREE.MeshBasicMaterial({ color: COLORS.done, transparent: true, opacity: 0, side: THREE.DoubleSide }));
    burst.position.y = lift;
    group.add(burst);

    // Soft contact shadow on the ground: dark under buildings, faint and wider the higher a cube floats.
    const height = p[1] - GROUND_Y;
    const shadow = new THREE.Mesh(
      new THREE.PlaneGeometry(1, 1),
      new THREE.MeshBasicMaterial({ map: this.shadowTex, transparent: true, depthWrite: false,
        opacity: grounded ? 0.55 : Math.max(0.08, 0.32 - height * 0.025) }));
    shadow.rotation.x = -Math.PI / 2;
    shadow.position.set(p[0], GROUND_Y + 0.02, p[2]);
    shadow.scale.setScalar(grounded ? 4.8 : 2.4 + height * 0.12);
    shadow.renderOrder = -2;
    this.scene.add(shadow);

    this.scene.add(group);

    const label = document.createElement('div');
    label.className = 'node-label';
    label.innerHTML = `<span class="nm"></span><span class="st"></span>`;
    label.querySelector('.nm').textContent = meta.title || DISPLAY_NAMES[name] || name;
    this.labelsEl.appendChild(label);

    const node = {
      name, group, cube, halo, ring, burst, label, tint, meta, grounded, shadow,
      anchor: group.position.clone().add(new THREE.Vector3(0, lift, 0)),
      state: 'idle', running: 0, glow: 0.15, targetGlow: 0.15,
      color: tint.clone(), targetColor: tint.clone(), burstT: 1, phase: Math.random() * 10,
    };
    this.nodes.set(name, node);
    if (!LAYOUT[name] && this.nodes.has('planner')) this._edge('planner', name); // ring agents are fed by the planner
    for (const [a, b] of LINKS) if ((name === a || name === b) && this.nodes.has(a) && this.nodes.has(b)) this._edge(a, b);
    return node;
  }

  _edge(a, b) {
    if (NOT_AGENTS.has(a) || NOT_AGENTS.has(b)) return null;
    const k = `${a}>${b}`;
    if (this.edges.has(k)) return this.edges.get(k);
    const A = this.ensureNode(a).anchor;
    const B = this.ensureNode(b).anchor;
    // Bow each edge away from the pipeline axis so links between ring agents don't cut through the middle.
    const ctrl = A.clone().add(B).multiplyScalar(0.5);
    const out = new THREE.Vector3(0, ctrl.y, ctrl.z);
    if (out.lengthSq() < 0.01) out.set(0, 1, 0);
    ctrl.add(out.normalize().multiplyScalar(1.6));
    const curve = new THREE.QuadraticBezierCurve3(A.clone(), ctrl, B.clone());
    const geo = new THREE.BufferGeometry().setFromPoints(curve.getPoints(48));
    const mat = new THREE.LineBasicMaterial({ color: EDGE_COLOR, transparent: true, opacity: 0.7 });
    const line = new THREE.Line(geo, mat);
    this.scene.add(line);
    const edge = { curve, line, mat, heat: 0 };
    this.edges.set(k, edge);
    return edge;
  }

  /** Visual state of an agent: idle | running | done | failed. `status` is the small text under the name. */
  setState(name, state, status = '') {
    const node = this.ensureNode(name);
    if (!node) return;
    const prev = node.state;
    node.state = state;
    node.label.className = `node-label ${state}`;
    node.label.querySelector('.st').textContent = status;
    if (state === 'running') {
      node.targetColor.copy(COLORS.running);
      node.targetGlow = 0.55;
    } else if (state === 'done') {
      node.targetColor.copy(COLORS.done);
      node.targetGlow = 0.35;
      if (prev === 'running') {
        this._burst(node, COLORS.done);
        node.cube.trigger('tip');
      }
    } else if (state === 'failed') {
      node.targetColor.copy(COLORS.failed);
      node.targetGlow = 0.45;
      this._burst(node, COLORS.failed);
      if (prev !== 'failed') node.cube.trigger('knock');
    } else {
      node.targetColor.copy(node.tint);
      node.targetGlow = 0.15;
    }
  }

  /** While an agent runs, keep particles streaming from its upstream agents into it. */
  setEmitter(name, sources) {
    const agents = (sources || []).filter((s) => !NOT_AGENTS.has(s));
    if (!NOT_AGENTS.has(name) && agents.length) this.emitters.set(name, { sources: agents, acc: 0 });
    else this.emitters.delete(name);
  }

  /** One-off burst of particles along a -> b (e.g. result delivered to the user). */
  flow(a, b, color = COLORS.done, count = 6) {
    const edge = this._edge(a, b);
    if (edge) {
      for (let i = 0; i < count; i++) this._spawn(edge, color, -i * 0.12);
      return;
    }
    // A flow from/to a non-agent (a new message arriving): flash the agent end instead.
    const node = this.nodes.get(NOT_AGENTS.has(b) ? a : b);
    if (node) this._burst(node, color);
  }

  _spawn(edge, color, t0 = 0) {
    const mesh = new THREE.Mesh(this.particleGeo, new THREE.MeshBasicMaterial({ color, transparent: true }));
    mesh.visible = false;
    this.scene.add(mesh);
    this.particles.push({ mesh, edge, t: t0, speed: 0.55 + Math.random() * 0.25 });
    edge.heat = 1;
    edge.mat.color.copy(color);
  }

  _burst(node, color) {
    node.burst.material.color.copy(color);
    node.burstT = 0;
  }

  _applyTheme() {
    this.scene.fog.color.set(cssColor('--scene-fog', '#f1f4fa'));
    EDGE_COLOR.set(cssColor('--scene-edge', '#b8c3d9'));
    for (const edge of this.edges.values()) if (edge.heat === 0) edge.mat.color.copy(EDGE_COLOR);
    if (this.grid) this.grid.material.color?.copy(EDGE_COLOR);
    if (this.floor) {
      this.floor.material.map = makeFloorTexture();
      this.floor.material.needsUpdate = true;
    }
  }

  _resize() {
    const host = this.canvas.parentElement;
    const w = host.clientWidth, h = host.clientHeight;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    // Fit the pipeline's width (±15.5 units at the default distance) in view, but never zoom in past 42°.
    const halfW = 17.5 / this.camera.position.distanceTo(this.controls.target);
    this.camera.fov = Math.max(42, (2 * Math.atan(halfW / this.camera.aspect) * 180) / Math.PI);
    this.camera.updateProjectionMatrix();
    this.size = { w, h };
  }

  _tick() {
    const dt = Math.min(this.clock.getDelta(), 0.05);
    const t = this.clock.elapsedTime;
    this.controls.update();
    this.stars.rotation.y += dt * 0.01;

    for (const node of this.nodes.values()) {
      node.color.lerp(node.targetColor, 1 - Math.pow(0.002, dt));
      node.glow += (node.targetGlow - node.glow) * (1 - Math.pow(0.01, dt));
      const running = node.state === 'running';
      node.cube.update(dt, t, node.state, node.color);
      if (node.cube.facesCamera) {
        const want = Math.atan2(this.camera.position.x - node.group.position.x,
          this.camera.position.z - node.group.position.z);
        let d = want - node.cube.object.rotation.y;
        d = Math.atan2(Math.sin(d), Math.cos(d));
        node.cube.object.rotation.y += d * Math.min(1, dt * 3);
      }
      node.halo.material.color.copy(node.state === 'idle' ? node.tint : node.color);
      node.halo.material.opacity = running ? 0.55 + 0.2 * Math.sin(t * 6 + node.phase) : node.state === 'idle' ? 0.22 : 0.4;
      node.halo.scale.setScalar(running ? 4.4 + 0.4 * Math.sin(t * 6 + node.phase) : 3.6);
      node.ring.material.opacity += ((running ? 0.9 : 0) - node.ring.material.opacity) * Math.min(1, dt * 6);
      node.ring.rotation.z += dt * 2.2;
      if (node.burstT < 1) {
        node.burstT = Math.min(1, node.burstT + dt * 1.3);
        node.burst.scale.setScalar(1 + node.burstT * 2.2);
        node.burst.material.opacity = 0.8 * (1 - node.burstT);
        node.burst.lookAt(this.camera.position);
      }
    }

    for (const [name, em] of this.emitters) {
      em.acc += dt;
      if (em.acc > 0.28) {
        em.acc = 0;
        for (const src of em.sources) this._spawn(this._edge(src, name), COLORS.running);
      }
    }

    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.t += dt * p.speed;
      if (p.t >= 1) {
        this.scene.remove(p.mesh);
        p.mesh.material.dispose();
        this.particles.splice(i, 1);
        continue;
      }
      if (p.t < 0) continue;
      p.mesh.visible = true;
      p.edge.curve.getPoint(p.t, p.mesh.position);
      p.mesh.material.opacity = Math.sin(Math.PI * p.t);
    }

    for (const edge of this.edges.values()) {
      edge.heat = Math.max(0, edge.heat - dt * 0.6);
      edge.mat.opacity = 0.7 + edge.heat * 0.3;
      if (edge.heat === 0) edge.mat.color.lerp(EDGE_COLOR, Math.min(1, dt * 2));
    }

    this.renderer.render(this.scene, this.camera);
    this._placeLabels();
  }

  _placeLabels() {
    if (!this.size) return;
    const v = new THREE.Vector3();
    const dists = [...this.nodes.values()].map((n) => n.group.position.distanceTo(this.camera.position));
    const near = Math.min(...dists), far = Math.max(...dists);
    for (const node of this.nodes.values()) {
      v.copy(node.group.position);
      v.y -= node.grounded ? 0.35 : 1.9;
      v.project(this.camera);
      const behind = v.z > 1;
      const x = (v.x * 0.5 + 0.5) * this.size.w;
      const y = (-v.y * 0.5 + 0.5) * this.size.h;
      // Nearer agents' labels sit on top; labels of agents at the back of the formation fade out a little.
      const depth = far > near ? (node.group.position.distanceTo(this.camera.position) - near) / (far - near) : 0;
      node.label.style.zIndex = String(100 - Math.round(depth * 99));
      node.label.style.opacity = behind ? 0 : String(node.state === 'idle' ? 1 - 0.55 * depth : 1);
      node.label.style.left = `${x}px`;
      node.label.style.top = `${y}px`;
    }
  }
}
