const MODEL_URL = "./assets/van_mobile.rtsp";
const HEADER_SIZE = 64;
const MAGIC = "RTSPGS1\u0000";

const canvas = document.querySelector("#view");
const statusEl = document.querySelector("#status");
const messageEl = document.querySelector("#message");
const counterEl = document.querySelector("#counter");
const resetBtn = document.querySelector("#reset");
const orbitBtn = document.querySelector("#orbit");
const openAssetBtn = document.querySelector("#openAsset");
const assetInput = document.querySelector("#assetInput");
const sizeInput = document.querySelector("#size");

const state = {
  device: null,
  context: null,
  pipeline: null,
  bindGroup: null,
  uniformBuffer: null,
  splatBuffer: null,
  format: null,
  splatCount: 0,
  center: [0, 0, 0],
  radius: 1,
  yaw: 0.45,
  pitch: 0.26,
  distance: 3.0,
  autoOrbit: true,
  sizeScale: 1.0,
  pointer: new Map(),
  lastPinch: 0,
  lastTime: performance.now(),
  raf: 0
};

const shader = /* wgsl */ `
struct Camera {
  view_proj: mat4x4<f32>,
  params: vec4<f32>,
};

struct Splat {
  pos: vec4<f32>,
  axis0: vec4<f32>,
  axis1: vec4<f32>,
  color: vec4<f32>,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) color: vec4<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;

const corners = array<vec2<f32>, 6>(
  vec2<f32>(-2.0, -2.0),
  vec2<f32>( 2.0, -2.0),
  vec2<f32>(-2.0,  2.0),
  vec2<f32>(-2.0,  2.0),
  vec2<f32>( 2.0, -2.0),
  vec2<f32>( 2.0,  2.0)
);

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOut {
  let splat = splats[instance_index];
  let uv = corners[vertex_index];
  let world = splat.pos.xyz + (splat.axis0.xyz * uv.x + splat.axis1.xyz * uv.y) * camera.params.x;

  var out: VertexOut;
  out.position = camera.view_proj * vec4<f32>(world, 1.0);
  out.uv = uv;
  out.color = splat.color;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 4.0) {
    discard;
  }
  let alpha = clamp(in.color.a * exp(-0.5 * r2), 0.0, 1.0);
  if (alpha < 0.004) {
    discard;
  }
  return vec4<f32>(in.color.rgb * alpha, alpha);
}
`;

function setStatus(text) {
  statusEl.textContent = text;
}

function showMessage(text) {
  messageEl.textContent = text;
  messageEl.classList.remove("hidden");
}

function hideMessage() {
  messageEl.classList.add("hidden");
}

function formatCount(count) {
  return new Intl.NumberFormat("en-US").format(count);
}

async function loadArrayBuffer(url) {
  const response = await fetch(url, { cache: "force-cache" });
  if (!response.ok || !response.body) {
    throw new Error(`Failed to load ${url}: ${response.status}`);
  }

  const total = Number(response.headers.get("content-length") || 0);
  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.byteLength;
    if (total > 0) {
      setStatus(`Loading ${Math.round((received / total) * 100)}%`);
    } else {
      setStatus(`Loading ${(received / 1048576).toFixed(1)} MB`);
    }
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return buffer.buffer;
}

function parseSplatAsset(buffer) {
  const bytes = new Uint8Array(buffer, 0, 8);
  const magic = String.fromCharCode(...bytes);
  if (magic !== MAGIC) {
    throw new Error("Invalid splat asset header");
  }

  const view = new DataView(buffer);
  const version = view.getUint32(8, true);
  const count = view.getUint32(12, true);
  const stride = view.getUint32(16, true);
  if (version !== 1 || stride !== 16) {
    throw new Error(`Unsupported splat asset version=${version} stride=${stride}`);
  }

  const center = [view.getFloat32(20, true), view.getFloat32(24, true), view.getFloat32(28, true)];
  const radius = view.getFloat32(32, true);
  const data = new Float32Array(buffer, HEADER_SIZE, count * stride);
  return { count, stride, center, radius, data };
}

async function initGpu() {
  if (!navigator.gpu) {
    throw new Error("This browser does not expose WebGPU. Use Safari on iOS 26+ or a current desktop browser.");
  }

  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) {
    throw new Error("No WebGPU adapter was found on this device.");
  }

  const device = await adapter.requestDevice();
  const context = canvas.getContext("webgpu");
  const format = navigator.gpu.getPreferredCanvasFormat();
  context.configure({
    device,
    format,
    alphaMode: "opaque"
  });

  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: {
      module,
      entryPoint: "vs_main"
    },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [
        {
          format,
          blend: {
            color: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add"
            },
            alpha: {
              srcFactor: "one",
              dstFactor: "one-minus-src-alpha",
              operation: "add"
            }
          }
        }
      ]
    },
    primitive: {
      topology: "triangle-list",
      cullMode: "none"
    }
  });

  state.device = device;
  state.context = context;
  state.pipeline = pipeline;
  state.format = format;
}

function uploadAsset(asset) {
  const { device, pipeline } = state;
  if (state.splatBuffer) state.splatBuffer.destroy();
  if (state.uniformBuffer) state.uniformBuffer.destroy();
  const splatBuffer = device.createBuffer({
    label: "splat data",
    size: asset.data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST
  });
  device.queue.writeBuffer(splatBuffer, 0, asset.data);

  const uniformBuffer = device.createBuffer({
    label: "camera uniforms",
    size: 80,
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST
  });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: splatBuffer } }
    ]
  });

  state.splatBuffer = splatBuffer;
  state.uniformBuffer = uniformBuffer;
  state.bindGroup = bindGroup;
  state.splatCount = asset.count;
  state.center = asset.center;
  state.radius = asset.radius;
  state.distance = asset.radius * 2.7;
  counterEl.textContent = `${formatCount(asset.count)} splats`;
}

function startRendering() {
  if (state.raf) {
    cancelAnimationFrame(state.raf);
  }
  state.lastTime = performance.now();
  state.raf = requestAnimationFrame(render);
}

function loadAssetBuffer(buffer, label = "Ready") {
  const asset = parseSplatAsset(buffer);
  uploadAsset(asset);
  hideMessage();
  setStatus(label);
  startRendering();
}

function resizeCanvas() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.floor(canvas.clientWidth * dpr));
  const height = Math.max(1, Math.floor(canvas.clientHeight * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function perspective(fovy, aspect, near, far) {
  const f = 1.0 / Math.tan(fovy / 2);
  const nf = 1 / (near - far);
  return new Float32Array([
    f / aspect,
    0,
    0,
    0,
    0,
    f,
    0,
    0,
    0,
    0,
    (far + near) * nf,
    -1,
    0,
    0,
    2 * far * near * nf,
    0
  ]);
}

function normalize(v) {
  const len = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / len, v[1] / len, v[2] / len];
}

function cross(a, b) {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function lookAt(eye, center, up) {
  const z = normalize([eye[0] - center[0], eye[1] - center[1], eye[2] - center[2]]);
  const x = normalize(cross(up, z));
  const y = cross(z, x);

  return new Float32Array([
    x[0],
    y[0],
    z[0],
    0,
    x[1],
    y[1],
    z[1],
    0,
    x[2],
    y[2],
    z[2],
    0,
    -dot(x, eye),
    -dot(y, eye),
    -dot(z, eye),
    1
  ]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] =
        a[0 * 4 + row] * b[col * 4 + 0] +
        a[1 * 4 + row] * b[col * 4 + 1] +
        a[2 * 4 + row] * b[col * 4 + 2] +
        a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function updateUniforms() {
  const aspect = canvas.width / Math.max(canvas.height, 1);
  const pitch = Math.max(-1.25, Math.min(1.25, state.pitch));
  state.pitch = pitch;

  const cosPitch = Math.cos(pitch);
  const eye = [
    state.center[0] + Math.sin(state.yaw) * cosPitch * state.distance,
    state.center[1] + Math.sin(pitch) * state.distance,
    state.center[2] + Math.cos(state.yaw) * cosPitch * state.distance
  ];
  const proj = perspective(47 * (Math.PI / 180), aspect, Math.max(0.01, state.radius * 0.02), state.radius * 20);
  const view = lookAt(eye, state.center, [0, 1, 0]);
  const viewProj = multiply(proj, view);

  const uniforms = new Float32Array(20);
  uniforms.set(viewProj, 0);
  uniforms[16] = state.sizeScale;
  state.device.queue.writeBuffer(state.uniformBuffer, 0, uniforms);
}

function render(now) {
  const dt = Math.min(0.05, (now - state.lastTime) / 1000);
  state.lastTime = now;
  if (state.autoOrbit && state.pointer.size === 0) {
    state.yaw += dt * 0.16;
  }

  resizeCanvas();
  updateUniforms();

  const encoder = state.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [
      {
        view: state.context.getCurrentTexture().createView(),
        clearValue: { r: 0.063, g: 0.067, b: 0.059, a: 1 },
        loadOp: "clear",
        storeOp: "store"
      }
    ]
  });
  pass.setPipeline(state.pipeline);
  pass.setBindGroup(0, state.bindGroup);
  pass.draw(6, state.splatCount);
  pass.end();
  state.device.queue.submit([encoder.finish()]);

  state.raf = requestAnimationFrame(render);
}

function setAutoOrbit(enabled) {
  state.autoOrbit = enabled;
  orbitBtn.classList.toggle("active", enabled);
}

function resetCamera() {
  state.yaw = 0.45;
  state.pitch = 0.26;
  state.distance = state.radius * 2.7;
}

function pointerDistance() {
  const points = [...state.pointer.values()];
  if (points.length < 2) return 0;
  return Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
}

function bindControls() {
  resetBtn.addEventListener("click", resetCamera);
  orbitBtn.addEventListener("click", () => setAutoOrbit(!state.autoOrbit));
  openAssetBtn.addEventListener("click", () => assetInput.click());
  assetInput.addEventListener("change", async () => {
    const file = assetInput.files && assetInput.files[0];
    if (!file) return;
    try {
      setStatus(`Opening ${file.name}`);
      const buffer = await file.arrayBuffer();
      loadAssetBuffer(buffer, file.name);
    } catch (error) {
      console.error(error);
      setStatus("Open failed");
      showMessage(error.message);
    }
  });
  sizeInput.addEventListener("input", () => {
    state.sizeScale = Number(sizeInput.value);
  });

  canvas.addEventListener("pointerdown", (event) => {
    canvas.setPointerCapture(event.pointerId);
    state.pointer.set(event.pointerId, { x: event.clientX, y: event.clientY });
    state.lastPinch = pointerDistance();
    setAutoOrbit(false);
  });

  canvas.addEventListener("pointermove", (event) => {
    const prev = state.pointer.get(event.pointerId);
    if (!prev) return;

    const current = { x: event.clientX, y: event.clientY };
    state.pointer.set(event.pointerId, current);

    if (state.pointer.size >= 2) {
      const dist = pointerDistance();
      if (state.lastPinch > 0 && dist > 0) {
        const ratio = state.lastPinch / dist;
        state.distance = Math.max(state.radius * 0.75, Math.min(state.radius * 8, state.distance * ratio));
      }
      state.lastPinch = dist;
      return;
    }

    const dx = current.x - prev.x;
    const dy = current.y - prev.y;
    state.yaw -= dx * 0.006;
    state.pitch -= dy * 0.006;
  });

  const release = (event) => {
    state.pointer.delete(event.pointerId);
    state.lastPinch = pointerDistance();
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);

  canvas.addEventListener(
    "wheel",
    (event) => {
      event.preventDefault();
      setAutoOrbit(false);
      const factor = Math.exp(event.deltaY * 0.001);
      state.distance = Math.max(state.radius * 0.75, Math.min(state.radius * 8, state.distance * factor));
    },
    { passive: false }
  );
}

async function main() {
  bindControls();
  hideMessage();

  await initGpu();

  if ("serviceWorker" in navigator && location.protocol === "https:") {
    navigator.serviceWorker.register("./sw.js").catch(() => {});
  }

  try {
    const buffer = await loadArrayBuffer(MODEL_URL);
    loadAssetBuffer(buffer);
  } catch (error) {
    console.warn(error);
    setStatus("Choose asset");
    showMessage("Tap the folder button and choose van_mobile.rtsp from Files.");
  }
}

main().catch((error) => {
  console.error(error);
  setStatus("Unavailable");
  showMessage(error.message);
});
