const MODEL_URL = new URLSearchParams(location.search).get("asset") || "./assets/van_mobile_full.rtsp";
const HEADER_SIZE = 64;
const MAGIC_LITE = "RTSPGS1\u0000";
const MAGIC_FULL = "RTSPGS2\u0000";
const FULL_STRIDE = 72;
const SH_C0 = 0.28209479177387814;

const canvas = document.querySelector("#view");
const statusEl = document.querySelector("#status");
const messageEl = document.querySelector("#message");
const counterEl = document.querySelector("#counter");
const resetBtn = document.querySelector("#reset");
const orbitBtn = document.querySelector("#orbit");
const sortBtn = document.querySelector("#sort");
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
  indexBuffer: null,
  format: null,
  splatCount: 0,
  center: [0, 0, 0],
  radius: 1,
  yaw: 0.45,
  pitch: 0.26,
  distance: 3.0,
  autoOrbit: true,
  sortEnabled: true,
  sizeScale: 1.0,
  pointer: new Map(),
  lastPinch: 0,
  lastTime: performance.now(),
  lastSortTime: 0,
  raf: 0,
  assetData: null,
  orderArray: null,
  depthArray: null,
  cameraEye: [0, 0, 0],
  cameraForward: [0, 0, -1]
};

const shader = /* wgsl */ `
struct Camera {
  view_proj: mat4x4<f32>,
  params: vec4<f32>,
  camera_pos: vec4<f32>,
  light_dir: vec4<f32>,
};

struct Splat {
  pos: vec4<f32>,
  axis0: vec4<f32>,
  axis1: vec4<f32>,
  material: vec4<f32>,
  aux: vec4<f32>,
  feature: vec4<f32>,
  sh_r0: vec4<f32>,
  sh_r1: vec4<f32>,
  sh_r2: vec4<f32>,
  sh_r3: vec4<f32>,
  sh_g0: vec4<f32>,
  sh_g1: vec4<f32>,
  sh_g2: vec4<f32>,
  sh_g3: vec4<f32>,
  sh_b0: vec4<f32>,
  sh_b1: vec4<f32>,
  sh_b2: vec4<f32>,
  sh_b3: vec4<f32>,
};

struct VertexOut {
  @builtin(position) position: vec4<f32>,
  @location(0) uv: vec2<f32>,
  @location(1) @interpolate(flat) instance: u32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var<storage, read> splats: array<Splat>;
@group(0) @binding(2) var<storage, read> draw_order: array<u32>;

const corners = array<vec2<f32>, 6>(
  vec2<f32>(-2.0, -2.0),
  vec2<f32>( 2.0, -2.0),
  vec2<f32>(-2.0,  2.0),
  vec2<f32>(-2.0,  2.0),
  vec2<f32>( 2.0, -2.0),
  vec2<f32>( 2.0,  2.0)
);

fn eval_sh(c0: vec4<f32>, c1: vec4<f32>, c2: vec4<f32>, c3: vec4<f32>, dir: vec3<f32>) -> f32 {
  let x = dir.x;
  let y = dir.y;
  let z = dir.z;
  let xx = x * x;
  let yy = y * y;
  let zz = z * z;
  let xy = x * y;
  let yz = y * z;
  let xz = x * z;

  var result = 0.28209479177387814 * c0.x;
  result = result - 0.4886025119029199 * y * c0.y + 0.4886025119029199 * z * c0.z - 0.4886025119029199 * x * c0.w;
  result = result
    + 1.0925484305920792 * xy * c1.x
    - 1.0925484305920792 * yz * c1.y
    + 0.31539156525252005 * (2.0 * zz - xx - yy) * c1.z
    - 1.0925484305920792 * xz * c1.w
    + 0.5462742152960396 * (xx - yy) * c2.x;
  result = result
    - 0.5900435899266435 * y * (3.0 * xx - yy) * c2.y
    + 2.890611442640554 * xy * z * c2.z
    - 0.4570457994644658 * y * (4.0 * zz - xx - yy) * c2.w
    + 0.3731763325901154 * z * (2.0 * zz - 3.0 * xx - 3.0 * yy) * c3.x
    - 0.4570457994644658 * x * (4.0 * zz - xx - yy) * c3.y
    + 1.445305721320277 * z * (xx - yy) * c3.z
    - 0.5900435899266435 * x * (xx - 3.0 * yy) * c3.w;
  return result;
}

fn splat_color(splat: Splat, view_to_point: vec3<f32>) -> vec3<f32> {
  let r = eval_sh(splat.sh_r0, splat.sh_r1, splat.sh_r2, splat.sh_r3, view_to_point);
  let g = eval_sh(splat.sh_g0, splat.sh_g1, splat.sh_g2, splat.sh_g3, view_to_point);
  let b = eval_sh(splat.sh_b0, splat.sh_b1, splat.sh_b2, splat.sh_b3, view_to_point);
  return clamp(vec3<f32>(r, g, b) + vec3<f32>(0.5), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn vs_main(@builtin(vertex_index) vertex_index: u32, @builtin(instance_index) instance_index: u32) -> VertexOut {
  let splat_index = draw_order[instance_index];
  let splat = splats[splat_index];
  let uv = corners[vertex_index];
  let world = splat.pos.xyz + (splat.axis0.xyz * uv.x + splat.axis1.xyz * uv.y) * camera.params.x;

  var out: VertexOut;
  out.position = camera.view_proj * vec4<f32>(world, 1.0);
  out.uv = uv;
  out.instance = splat_index;
  return out;
}

@fragment
fn fs_main(in: VertexOut) -> @location(0) vec4<f32> {
  let r2 = dot(in.uv, in.uv);
  if (r2 > 4.0) {
    discard;
  }

  let splat = splats[in.instance];
  let occupancy = clamp(splat.material.x, 0.0, 1.0);
  let opacity = clamp(splat.material.y, 0.0, 1.0);
  let transmissivity = clamp(splat.material.z * splat.aux.z, 0.0, 1.0);
  let roughness = clamp(splat.aux.x, 0.04, 1.0);
  let reflectance = clamp(splat.aux.y * splat.aux.z, 0.0, 1.0);

  let view_to_point = normalize(splat.pos.xyz - camera.camera_pos.xyz);
  let point_to_view = -view_to_point;
  var normal = normalize(cross(splat.axis0.xyz, splat.axis1.xyz));
  if (dot(normal, point_to_view) < 0.0) {
    normal = -normal;
  }

  let sh_color = splat_color(splat, view_to_point);
  let surface_weight = 1.0 - transmissivity;
  let volume_weight = transmissivity * opacity;
  let material_weight = max(surface_weight + volume_weight, 0.02);

  let light_dir = normalize(camera.light_dir.xyz);
  let half_dir = normalize(point_to_view + light_dir);
  let spec_power = mix(96.0, 8.0, roughness);
  let spec = pow(max(dot(normal, half_dir), 0.0), spec_power) * reflectance * (1.0 - transmissivity * 0.35);
  let feature_tint = clamp(vec3<f32>(0.5) + 0.5 * splat.feature.xyz, vec3<f32>(0.0), vec3<f32>(1.0));
  let spec_color = mix(vec3<f32>(1.0), feature_tint, 0.35) * spec;

  let gaussian = exp(-0.5 * r2);
  let alpha = clamp(occupancy * material_weight * gaussian, 0.0, 1.0);
  if (alpha < 0.003) {
    discard;
  }

  let color = sh_color * material_weight + spec_color;
  return vec4<f32>(color * alpha, alpha);
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

function convertLiteAsset(buffer, count, center, radius) {
  const source = new Float32Array(buffer, HEADER_SIZE, count * 16);
  const full = new Float32Array(count * FULL_STRIDE);
  for (let i = 0; i < count; i += 1) {
    const src = i * 16;
    const dst = i * FULL_STRIDE;
    full.set(source.subarray(src, src + 12), dst);
    const r = source[src + 12];
    const g = source[src + 13];
    const b = source[src + 14];
    const alpha = source[src + 15];
    full[dst + 12] = alpha;
    full[dst + 13] = 1.0;
    full[dst + 14] = 0.0;
    full[dst + 15] = alpha;
    full[dst + 16] = 0.5;
    full[dst + 17] = 0.0;
    full[dst + 18] = 1.0;
    full[dst + 19] = alpha;
    full[dst + 24] = (r - 0.5) / SH_C0;
    full[dst + 40] = (g - 0.5) / SH_C0;
    full[dst + 56] = (b - 0.5) / SH_C0;
  }
  return { count, stride: FULL_STRIDE, center, radius, data: full, version: 1 };
}

function parseSplatAsset(buffer) {
  const bytes = new Uint8Array(buffer, 0, 8);
  const magic = String.fromCharCode(...bytes);
  const view = new DataView(buffer);
  const version = view.getUint32(8, true);
  const count = view.getUint32(12, true);
  const stride = view.getUint32(16, true);
  const center = [view.getFloat32(20, true), view.getFloat32(24, true), view.getFloat32(28, true)];
  const radius = view.getFloat32(32, true);

  if (magic === MAGIC_LITE && version === 1 && stride === 16) {
    return convertLiteAsset(buffer, count, center, radius);
  }
  if (magic !== MAGIC_FULL || version !== 2 || stride !== FULL_STRIDE) {
    throw new Error(`Unsupported splat asset magic=${magic} version=${version} stride=${stride}`);
  }

  const data = new Float32Array(buffer, HEADER_SIZE, count * stride);
  return { count, stride, center, radius, data, version };
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
  context.configure({ device, format, alphaMode: "opaque" });

  const module = device.createShaderModule({ code: shader });
  const pipeline = device.createRenderPipeline({
    layout: "auto",
    vertex: { module, entryPoint: "vs_main" },
    fragment: {
      module,
      entryPoint: "fs_main",
      targets: [
        {
          format,
          blend: {
            color: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" },
            alpha: { srcFactor: "one", dstFactor: "one-minus-src-alpha", operation: "add" }
          }
        }
      ]
    },
    primitive: { topology: "triangle-list", cullMode: "none" }
  });

  state.device = device;
  state.context = context;
  state.pipeline = pipeline;
  state.format = format;
}

function uploadAsset(asset) {
  const { device, pipeline } = state;
  if (state.splatBuffer) state.splatBuffer.destroy();
  if (state.indexBuffer) state.indexBuffer.destroy();
  if (state.uniformBuffer) state.uniformBuffer.destroy();

  const splatBuffer = device.createBuffer({ label: "rt-splat data", size: asset.data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(splatBuffer, 0, asset.data);

  const orderArray = new Uint32Array(asset.count);
  for (let i = 0; i < asset.count; i += 1) orderArray[i] = i;
  const indexBuffer = device.createBuffer({ label: "depth sorted order", size: orderArray.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(indexBuffer, 0, orderArray);

  const uniformBuffer = device.createBuffer({ label: "camera uniforms", size: 128, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });

  const bindGroup = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uniformBuffer } },
      { binding: 1, resource: { buffer: splatBuffer } },
      { binding: 2, resource: { buffer: indexBuffer } }
    ]
  });

  state.splatBuffer = splatBuffer;
  state.indexBuffer = indexBuffer;
  state.uniformBuffer = uniformBuffer;
  state.bindGroup = bindGroup;
  state.splatCount = asset.count;
  state.center = asset.center;
  state.radius = asset.radius;
  state.distance = asset.radius * 2.7;
  state.assetData = asset.data;
  state.orderArray = orderArray;
  state.depthArray = new Float32Array(asset.count);
  counterEl.textContent = `${formatCount(asset.count)} splats`;
  setSortEnabled(state.sortEnabled, true);
}

function startRendering() {
  if (state.raf) cancelAnimationFrame(state.raf);
  state.lastTime = performance.now();
  state.raf = requestAnimationFrame(render);
}

function loadAssetBuffer(buffer, label = "Ready") {
  const asset = parseSplatAsset(buffer);
  uploadAsset(asset);
  hideMessage();
  setStatus(`${label} · ${asset.version === 2 ? "RT full" : "lite"}`);
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
  return new Float32Array([f / aspect, 0, 0, 0, 0, f, 0, 0, 0, 0, (far + near) * nf, -1, 0, 0, 2 * far * near * nf, 0]);
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
  return new Float32Array([x[0], y[0], z[0], 0, x[1], y[1], z[1], 0, x[2], y[2], z[2], 0, -dot(x, eye), -dot(y, eye), -dot(z, eye), 1]);
}

function multiply(a, b) {
  const out = new Float32Array(16);
  for (let col = 0; col < 4; col += 1) {
    for (let row = 0; row < 4; row += 1) {
      out[col * 4 + row] = a[0 * 4 + row] * b[col * 4 + 0] + a[1 * 4 + row] * b[col * 4 + 1] + a[2 * 4 + row] * b[col * 4 + 2] + a[3 * 4 + row] * b[col * 4 + 3];
    }
  }
  return out;
}

function updateUniforms() {
  const aspect = canvas.width / Math.max(canvas.height, 1);
  state.pitch = Math.max(-1.25, Math.min(1.25, state.pitch));
  const cosPitch = Math.cos(state.pitch);
  const eye = [
    state.center[0] + Math.sin(state.yaw) * cosPitch * state.distance,
    state.center[1] + Math.sin(state.pitch) * state.distance,
    state.center[2] + Math.cos(state.yaw) * cosPitch * state.distance
  ];
  const forward = normalize([state.center[0] - eye[0], state.center[1] - eye[1], state.center[2] - eye[2]]);
  state.cameraEye = eye;
  state.cameraForward = forward;

  const proj = perspective(47 * (Math.PI / 180), aspect, Math.max(0.01, state.radius * 0.02), state.radius * 20);
  const view = lookAt(eye, state.center, [0, 1, 0]);
  const viewProj = multiply(proj, view);
  const uniforms = new Float32Array(32);
  uniforms.set(viewProj, 0);
  uniforms[16] = state.sizeScale;
  uniforms[20] = eye[0];
  uniforms[21] = eye[1];
  uniforms[22] = eye[2];
  uniforms[24] = -0.35;
  uniforms[25] = 0.75;
  uniforms[26] = 0.55;
  state.device.queue.writeBuffer(state.uniformBuffer, 0, uniforms);
}

function sortSplats(now, force = false) {
  if (!state.sortEnabled || !state.assetData || !state.orderArray || !state.indexBuffer) return;
  const interval = state.pointer.size > 0 ? 90 : 220;
  if (!force && now - state.lastSortTime < interval) return;
  state.lastSortTime = now;
  const data = state.assetData;
  const depths = state.depthArray;
  const order = state.orderArray;
  const eye = state.cameraEye;
  const fwd = state.cameraForward;
  for (let i = 0; i < state.splatCount; i += 1) {
    const base = i * FULL_STRIDE;
    depths[i] = (data[base] - eye[0]) * fwd[0] + (data[base + 1] - eye[1]) * fwd[1] + (data[base + 2] - eye[2]) * fwd[2];
  }
  order.sort((a, b) => depths[b] - depths[a]);
  state.device.queue.writeBuffer(state.indexBuffer, 0, order);
}

function render(now) {
  const dt = Math.min(0.05, (now - state.lastTime) / 1000);
  state.lastTime = now;
  if (state.autoOrbit && state.pointer.size === 0) state.yaw += dt * 0.16;
  resizeCanvas();
  updateUniforms();
  sortSplats(now);
  const encoder = state.device.createCommandEncoder();
  const pass = encoder.beginRenderPass({
    colorAttachments: [{ view: state.context.getCurrentTexture().createView(), clearValue: { r: 0.063, g: 0.067, b: 0.059, a: 1 }, loadOp: "clear", storeOp: "store" }]
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

function setSortEnabled(enabled, resetOrder = false) {
  state.sortEnabled = enabled;
  sortBtn.classList.toggle("active", enabled);
  if (!enabled && state.orderArray && state.indexBuffer) {
    for (let i = 0; i < state.orderArray.length; i += 1) state.orderArray[i] = i;
    state.device.queue.writeBuffer(state.indexBuffer, 0, state.orderArray);
  } else if (enabled && resetOrder && state.orderArray) {
    sortSplats(performance.now(), true);
  }
}

function resetCamera() {
  state.yaw = 0.45;
  state.pitch = 0.26;
  state.distance = state.radius * 2.7;
  sortSplats(performance.now(), true);
}

function pointerDistance() {
  const points = [...state.pointer.values()];
  if (points.length < 2) return 0;
  return Math.hypot(points[0].x - points[1].x, points[0].y - points[1].y);
}

function bindControls() {
  resetBtn.addEventListener("click", resetCamera);
  orbitBtn.addEventListener("click", () => setAutoOrbit(!state.autoOrbit));
  sortBtn.addEventListener("click", () => setSortEnabled(!state.sortEnabled));
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
      if (state.lastPinch > 0 && dist > 0) state.distance = Math.max(state.radius * 0.75, Math.min(state.radius * 8, state.distance * (state.lastPinch / dist)));
      state.lastPinch = dist;
      return;
    }
    state.yaw -= (current.x - prev.x) * 0.006;
    state.pitch -= (current.y - prev.y) * 0.006;
  });
  const release = (event) => {
    state.pointer.delete(event.pointerId);
    state.lastPinch = pointerDistance();
    sortSplats(performance.now(), true);
  };
  canvas.addEventListener("pointerup", release);
  canvas.addEventListener("pointercancel", release);
  canvas.addEventListener("wheel", (event) => {
    event.preventDefault();
    setAutoOrbit(false);
    state.distance = Math.max(state.radius * 0.75, Math.min(state.radius * 8, state.distance * Math.exp(event.deltaY * 0.001)));
  }, { passive: false });
}

async function main() {
  bindControls();
  hideMessage();
  await initGpu();
  if ("serviceWorker" in navigator && location.protocol === "https:") navigator.serviceWorker.register("./sw.js").catch(() => {});
  try {
    const buffer = await loadArrayBuffer(MODEL_URL);
    loadAssetBuffer(buffer);
  } catch (error) {
    console.warn(error);
    setStatus("Choose asset");
    showMessage("Tap the folder button and choose van_mobile_full.rtsp from Files.");
  }
}

main().catch((error) => {
  console.error(error);
  setStatus("Unavailable");
  showMessage(error.message);
});
