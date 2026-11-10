import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const SPECS = {
  both: { label: "Both garments", parts: "--", tris: "--", verts: "--" },
  tshirt: { label: "Crewneck T-Shirt", parts: "--", tris: "--", verts: "--" },
  hoodie: { label: "Pullover Hoodie", parts: "--", tris: "--", verts: "--" },
};

const CAMERA_PRESETS = {
  "three-quarter": { dir: new THREE.Vector3(0.55, 0.24, 0.80) },
  front: { dir: new THREE.Vector3(0.0, 0.14, 0.99) },
  back: { dir: new THREE.Vector3(0.0, 0.16, -0.99) },
  side: { dir: new THREE.Vector3(0.99, 0.14, 0.04) },
};

const host = document.getElementById("canvasHost");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingSub = document.getElementById("loadingSub");
const activeModelLabel = document.getElementById("activeModelLabel");
const specParts = document.getElementById("specParts");
const specQuads = document.getElementById("specQuads");
const specVerts = document.getElementById("specVerts");

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(host.clientWidth, host.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.ACESFilmicToneMapping;
renderer.toneMappingExposure = 1.04;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xefefec);

const pmrem = new THREE.PMREMGenerator(renderer);
const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environment = envTex;
scene.environmentIntensity = 0.55;

const camera = new THREE.PerspectiveCamera(36, host.clientWidth / Math.max(1, host.clientHeight), 0.05, 50);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.7;
controls.maxDistance = 6.0;
controls.maxPolarAngle = Math.PI * 0.54;

// 3-point studio lighting
const keyLight = new THREE.DirectionalLight(0xfffcf5, 2.2);
keyLight.position.set(2.4, 3.2, 2.8);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 12;
keyLight.shadow.camera.left = -2.0;
keyLight.shadow.camera.right = 2.0;
keyLight.shadow.camera.top = 2.0;
keyLight.shadow.camera.bottom = -2.0;
keyLight.shadow.bias = -0.0005;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xeef4fa, 0.9);
fillLight.position.set(-2.6, 1.8, 2.0);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xffffff, 1.15);
rimLight.position.set(0.2, 2.4, -3.0);
scene.add(rimLight);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0xd8d7d2, 0.45);
scene.add(hemiLight);

// Subtle contact shadow floor
const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(12, 12),
  new THREE.ShadowMaterial({ opacity: 0.11 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.002;
floor.receiveShadow = true;
scene.add(floor);

const rootGroup = new THREE.Group();
scene.add(rootGroup);

let tshirtNodes = [];
let hoodieNodes = [];
const originalPosX = new Map();
const originalMaterials = new Map();
const clayMaterial = new THREE.MeshStandardMaterial({
  color: 0xd9d6ce,
  roughness: 0.82,
  metalness: 0.0,
});
const wireMaterial = new THREE.MeshBasicMaterial({
  color: 0x1b6c80,
  wireframe: true,
});

const state = {
  garment: "both",
  camera: "three-quarter",
  shading: "studio",
  tshirtColor: "#eae6de",
  hoodieColor: "#2f343d",
};

let camAnim = null;
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setCameraTarget(presetKey, immediate = false) {
  const preset = CAMERA_PRESETS[presetKey] || CAMERA_PRESETS["three-quarter"];
  const box = new THREE.Box3();
  const activeNodes = state.garment === "tshirt" ? tshirtNodes : state.garment === "hoodie" ? hoodieNodes : [...tshirtNodes, ...hoodieNodes];
  for (const n of activeNodes) {
    if (n.visible) box.expandByObject(n);
  }
  if (box.isEmpty()) return;

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const maxDim = Math.max(size.x, size.y, size.z);
  const fovRad = THREE.MathUtils.degToRad(camera.fov);
  const framePad = camera.aspect < 1.0 ? 1.25 : 1.08;
  const dist = (maxDim / (2 * Math.tan(fovRad / 2))) * framePad;

  const targetPos = center.clone().add(preset.dir.clone().normalize().multiplyScalar(dist));

  if (immediate || prefersReducedMotion) {
    camera.position.copy(targetPos);
    controls.target.copy(center);
    controls.update();
    camAnim = null;
  } else {
    camAnim = {
      startPos: camera.position.clone(),
      endPos: targetPos,
      startTarget: controls.target.clone(),
      endTarget: center,
      t: 0,
    };
  }
}

function applyColors() {
  const tshirtCol = new THREE.Color(state.tshirtColor);
  const collarCol = tshirtCol.clone().multiplyScalar(0.95);
  const hoodieCol = new THREE.Color(state.hoodieColor);
  const ribCol = hoodieCol.clone().multiplyScalar(0.88);

  for (const [mesh, mat] of originalMaterials.entries()) {
    if (!mat || !mat.name) continue;
    if (mat.name === "tshirt_cotton") {
      mat.color.copy(tshirtCol);
    } else if (mat.name === "tshirt_collar_rib") {
      mat.color.copy(collarCol);
    } else if (mat.name === "hoodie_fleece") {
      mat.color.copy(hoodieCol);
    } else if (mat.name === "hoodie_rib") {
      mat.color.copy(ribCol);
    }
    mesh.material.needsUpdate = true;
  }
}

function applyShading() {
  for (const [mesh, origMat] of originalMaterials.entries()) {
    if (state.shading === "studio") {
      mesh.material = origMat;
    } else if (state.shading === "clay") {
      mesh.material = clayMaterial;
    } else {
      mesh.material = wireMaterial;
    }
  }
}

function centerOffsetFor(nodes) {
  const box = new THREE.Box3();
  for (const n of nodes) {
    box.expandByObject(n);
  }
  if (box.isEmpty()) return 0;
  return -box.getCenter(new THREE.Vector3()).x;
}

function applyGarmentSelection(immediate = false) {
  for (const n of tshirtNodes) {
    n.visible = state.garment !== "hoodie";
    n.position.x = originalPosX.get(n) ?? 0;
  }
  for (const n of hoodieNodes) {
    n.visible = state.garment !== "tshirt";
    n.position.x = originalPosX.get(n) ?? 0;
  }

  if (state.garment === "tshirt") {
    const dx = centerOffsetFor(tshirtNodes);
    for (const n of tshirtNodes) n.position.x += dx;
  } else if (state.garment === "hoodie") {
    const dx = centerOffsetFor(hoodieNodes);
    for (const n of hoodieNodes) n.position.x += dx;
  }

  const info = SPECS[state.garment];
  activeModelLabel.textContent = info.label;
  specParts.textContent = info.parts;
  specQuads.textContent = info.tris;
  specVerts.textContent = info.verts;
  setCameraTarget(state.camera, immediate);
}

function computeStats(nodes) {
  let tris = 0;
  let verts = 0;
  for (const obj of nodes) {
    const g = obj.geometry;
    if (!g) continue;
    verts += g.attributes.position ? g.attributes.position.count : 0;
    tris += g.index ? Math.round(g.index.count / 3) : Math.round((g.attributes.position?.count || 0) / 3);
  }
  return { parts: `${nodes.length} parts`, tris: tris.toLocaleString("en-US"), verts: verts.toLocaleString("en-US") };
}

const loader = new GLTFLoader();
loader.load(
  "./shirt_and_hoodie.glb",
  (gltf) => {
    const model = gltf.scene;
    rootGroup.add(model);

    model.traverse((obj) => {
      if (!obj.isMesh) return;
      obj.castShadow = true;
      obj.receiveShadow = true;
      originalPosX.set(obj, obj.position.x);
      if (obj.material) {
        obj.material.roughness = Math.max(obj.material.roughness ?? 0.85, 0.72);
        originalMaterials.set(obj, obj.material);
      }
      if (obj.name.startsWith("TShirt")) {
        tshirtNodes.push(obj);
      } else if (obj.name.startsWith("Hoodie")) {
        hoodieNodes.push(obj);
      }
    });

    SPECS.tshirt = { label: "Crewneck T-Shirt", ...computeStats(tshirtNodes) };
    SPECS.hoodie = { label: "Pullover Hoodie", ...computeStats(hoodieNodes) };
    SPECS.both = { label: "Both garments", ...computeStats([...tshirtNodes, ...hoodieNodes]) };

    applyColors();
    applyShading();
    applyGarmentSelection(true);

    host.setAttribute("aria-busy", "false");
    loadingOverlay.classList.add("is-hidden");
  },
  (evt) => {
    if (evt.total > 0) {
      const pct = Math.round((evt.loaded / evt.total) * 100);
      loadingSub.textContent = `Loading shirt_and_hoodie.glb (${pct}%)`;
    }
  },
  () => {
    loadingSub.textContent = "Could not load .glb file. Switch to Cycles render tab or download .blend.";
    host.setAttribute("aria-busy", "false");
  }
);

// Bind UI controls
document.querySelectorAll("[data-garment]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-garment]").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    state.garment = btn.dataset.garment;
    applyGarmentSelection();
  });
});

document.querySelectorAll("[data-camera]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-camera]").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    state.camera = btn.dataset.camera;
    setCameraTarget(state.camera, false);
  });
});

document.querySelectorAll("[data-shading]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-shading]").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    state.shading = btn.dataset.shading;
    applyShading();
  });
});

document.querySelectorAll(".swatch-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.target;
    document.querySelectorAll(`.swatch-btn[data-target="${target}"]`).forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    if (target === "tshirt") {
      state.tshirtColor = btn.dataset.color;
      document.getElementById("tshirtColorName").textContent = btn.dataset.name;
    } else {
      state.hoodieColor = btn.dataset.color;
      document.getElementById("hoodieColorName").textContent = btn.dataset.name;
    }
    applyColors();
  });
});

const autoRotateToggle = document.getElementById("autoRotateToggle");
autoRotateToggle.addEventListener("change", () => {
  controls.autoRotate = autoRotateToggle.checked;
  controls.autoRotateSpeed = 1.6;
});

function dollyBy(factor) {
  const offset = camera.position.clone().sub(controls.target);
  const nextLen = THREE.MathUtils.clamp(offset.length() * factor, controls.minDistance, controls.maxDistance);
  offset.setLength(nextLen);
  camera.position.copy(controls.target).add(offset);
  controls.update();
}

document.getElementById("zoomInBtn").addEventListener("click", () => dollyBy(0.82));
document.getElementById("zoomOutBtn").addEventListener("click", () => dollyBy(1.22));
document.getElementById("resetCamBtn").addEventListener("click", () => {
  state.camera = "three-quarter";
  document.querySelectorAll("[data-camera]").forEach((b) => b.classList.toggle("is-active", b.dataset.camera === "three-quarter"));
  setCameraTarget("three-quarter", false);
});

// Topbar tabs: Interactive 3D vs Cycles render
const tabInteractive = document.getElementById("tabInteractive");
const tabCycles = document.getElementById("tabCycles");
const panel3d = document.getElementById("panel3d");
const panelCycles = document.getElementById("panelCycles");

function switchTab(mode) {
  const is3d = mode === "3d";
  tabInteractive.setAttribute("aria-selected", String(is3d));
  tabCycles.setAttribute("aria-selected", String(!is3d));
  panel3d.hidden = !is3d;
  panelCycles.hidden = is3d;
  if (is3d) resizeViewport();
}

tabInteractive.addEventListener("click", () => switchTab("3d"));
tabCycles.addEventListener("click", () => switchTab("cycles"));

function resizeViewport() {
  const w = host.clientWidth;
  const h = Math.max(1, host.clientHeight);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h);
}

window.addEventListener("resize", resizeViewport);

function animate() {
  requestAnimationFrame(animate);
  if (camAnim) {
    camAnim.t = Math.min(1, camAnim.t + 0.08);
    const ease = 1 - Math.pow(1 - camAnim.t, 3);
    camera.position.lerpVectors(camAnim.startPos, camAnim.endPos, ease);
    controls.target.lerpVectors(camAnim.startTarget, camAnim.endTarget, ease);
    if (camAnim.t >= 1) camAnim = null;
  }
  controls.update();
  renderer.render(scene, camera);
}

animate();
