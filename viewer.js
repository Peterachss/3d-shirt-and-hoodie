import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { RoomEnvironment } from "three/addons/environments/RoomEnvironment.js";

const CHAM_COLORWAYS = [
  { name: "Forest Green", hex: "#28382e", contrast: "light" },
  { name: "Navy", hex: "#242839", contrast: "light" },
  { name: "Cream", hex: "#e1cdbf", contrast: "dark" },
  { name: "Pastel Pink", hex: "#e9afb9", contrast: "dark" },
];

const SPECS = {
  lineup: { label: "CHAM Collection · All 4 Colorways", tris: "--", verts: "--" },
  both: { label: "CHAM Hoodie & Sports Shirt Pair", tris: "--", verts: "--" },
  tshirt: { label: "CHAM Sports Shirt · 160 GSM", tris: "--", verts: "--" },
  hoodie: { label: "CHAM Hoodie · 350 GSM", tris: "--", verts: "--" },
};

const CAMERA_PRESETS = {
  front: { dir: new THREE.Vector3(0.0, 0.08, 1.0) },
  back: { dir: new THREE.Vector3(0.0, 0.10, -1.0) },
  "three-quarter": { dir: new THREE.Vector3(0.46, 0.20, 0.86) },
  side: { dir: new THREE.Vector3(0.98, 0.12, 0.16) },
};

const host = document.getElementById("canvasHost");
const loadingOverlay = document.getElementById("loadingOverlay");
const loadingSub = document.getElementById("loadingSub");
const activeModelLabel = document.getElementById("activeModelLabel");
const specQuads = document.getElementById("specQuads");
const specVerts = document.getElementById("specVerts");

const renderer = new THREE.WebGLRenderer({ antialias: true, powerPreference: "high-performance" });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
renderer.setSize(host.clientWidth, host.clientHeight);
renderer.outputColorSpace = THREE.SRGBColorSpace;
renderer.toneMapping = THREE.NeutralToneMapping;
renderer.toneMappingExposure = 0.94;
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
host.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xe6dfd7);

const pmrem = new THREE.PMREMGenerator(renderer);
const envTex = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
scene.environment = envTex;
scene.environmentIntensity = 0.34;

const camera = new THREE.PerspectiveCamera(26, host.clientWidth / Math.max(1, host.clientHeight), 0.12, 30);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.07;
controls.minDistance = 0.7;
controls.maxDistance = 12.0;
controls.maxPolarAngle = Math.PI * 0.56;

// Studio lighting calibrated for crisp fold definition on both dark and pastel/cream colorways
const keyLight = new THREE.DirectionalLight(0xfffcf5, 1.55);
keyLight.position.set(2.8, 3.6, 2.8);
keyLight.castShadow = true;
keyLight.shadow.mapSize.set(2048, 2048);
keyLight.shadow.camera.near = 0.5;
keyLight.shadow.camera.far = 14;
keyLight.shadow.camera.left = -3.2;
keyLight.shadow.camera.right = 3.2;
keyLight.shadow.camera.top = 2.4;
keyLight.shadow.camera.bottom = -2.4;
keyLight.shadow.bias = -0.0005;
scene.add(keyLight);

const fillLight = new THREE.DirectionalLight(0xf3f6fa, 0.55);
fillLight.position.set(-2.8, 1.8, 2.2);
scene.add(fillLight);

const rimLight = new THREE.DirectionalLight(0xfffcf8, 0.95);
rimLight.position.set(0.0, 2.6, -3.4);
scene.add(rimLight);

const hemiLight = new THREE.HemisphereLight(0xffffff, 0xcabfb4, 0.36);
scene.add(hemiLight);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(16, 16),
  new THREE.ShadowMaterial({ opacity: 0.09 })
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.62;
floor.receiveShadow = true;
scene.add(floor);

const rootGroup = new THREE.Group();
scene.add(rootGroup);

const lineupGroup = new THREE.Group();
lineupGroup.visible = false;
scene.add(lineupGroup);

let tshirtNodes = [];
let hoodieNodes = [];
const originalPosX = new Map();
const originalMaterials = new Map();
const lineupInstances = [];

const clayMaterial = new THREE.MeshStandardMaterial({
  color: 0xd9d6ce,
  roughness: 0.82,
  metalness: 0.0,
});
const wireMaterial = new THREE.MeshBasicMaterial({
  color: 0x1b6c80,
  wireframe: true,
});

// Load CHAM high-resolution graphic decal textures for dynamic contrast switching
const texLoader = new THREE.TextureLoader();
const decalTextures = {};
for (const [key, url] of [
  ["hoodie_front", "./textures/cham_hoodie_front.png"],
  ["shirt_front", "./textures/cham_shirt_front.png"],
  ["back_dark", "./textures/cham_back_dark.png"],
  ["back_light", "./textures/cham_back_light.png"],
  ["vn_flag", "./textures/cham_vn_flag.png"],
]) {
  const t = texLoader.load(url);
  t.colorSpace = THREE.SRGBColorSpace;
  t.flipY = false;
  decalTextures[key] = t;
}

const FABRIC_PRESETS = {
  loopback: { roughness: 0.88, sheen: 0.20, sheenRoughness: 0.66, normalScale: 0.34, tile: 44 },
  mesh: { roughness: 0.80, sheen: 0.16, sheenRoughness: 0.54, normalScale: 0.44, tile: 46 },
  cotton: { roughness: 0.86, sheen: 0.16, sheenRoughness: 0.62, normalScale: 0.24, tile: 48 },
  fleece: { roughness: 0.93, sheen: 0.26, sheenRoughness: 0.74, normalScale: 0.28, tile: 48 },
  waffle: { roughness: 0.84, sheen: 0.16, sheenRoughness: 0.58, normalScale: 0.58, tile: 34 },
  nylon: { roughness: 0.52, sheen: 0.28, sheenRoughness: 0.38, normalScale: 0.30, tile: 50 },
};

const fabricTextures = new Map();

function getFabricNormalTexture(kind) {
  if (fabricTextures.has(kind)) return fabricTextures.get(kind);
  const size = 128;
  const canvas = document.createElement("canvas");
  canvas.width = size;
  canvas.height = size;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(size, size);
  const h = new Float32Array(size * size);

  for (let y = 0; y < size; y++) {
    const v = (y / size) * Math.PI * 2;
    for (let x = 0; x < size; x++) {
      const u = (x / size) * Math.PI * 2;
      let val = 0;
      if (kind === "mesh") {
        // 160 GSM Breathable Mesh (Vai thun me): staggered birdseye pinhole dimples
        const s1 = Math.sin(10 * u) * Math.sin(10 * u) + Math.sin(10 * v) * Math.sin(10 * v);
        const s2 = Math.cos(10 * u) * Math.cos(10 * u) + Math.cos(10 * v) * Math.cos(10 * v);
        val = -0.75 * (Math.exp(-s1 / 0.22) + Math.exp(-s2 / 0.22)) + 0.2 * Math.sin(20 * u);
      } else if (kind === "loopback") {
        // 350 GSM Loopback Knit (Vai chan cua): diagonal French terry knit wales
        val = 0.5 * Math.sin(14 * (u + 0.5 * v)) + 0.3 * Math.sin(14 * u) * Math.cos(14 * v);
      } else if (kind === "cotton") {
        val = 0.45 * Math.sin(16 * u + 0.5 * Math.sin(16 * v)) + 0.35 * Math.sin(16 * v);
      } else if (kind === "fleece") {
        val = 0.4 * Math.sin(14 * u) * Math.cos(14 * v) + 0.35 * Math.sin(28 * u + 19 * v);
      } else if (kind === "waffle") {
        const gu = Math.cos(8 * u);
        const gv = Math.cos(8 * v);
        val = 0.8 * Math.min(gu, gv) + 0.2 * gu * gv;
      } else {
        val = 0.6 * Math.sin(18 * u) + 0.6 * Math.sin(18 * v);
      }
      h[y * size + x] = val;
    }
  }

  const strength = kind === "waffle" ? 1.7 : kind === "mesh" ? 1.5 : 1.1;
  for (let y = 0; y < size; y++) {
    const ym = (y - 1 + size) % size;
    const yp = (y + 1 + size) % size;
    for (let x = 0; x < size; x++) {
      const xm = (x - 1 + size) % size;
      const xp = (x + 1 + size) % size;
      const dx = (h[y * size + xp] - h[y * size + xm]) * 0.5 * strength;
      const dy = (h[yp * size + x] - h[ym * size + x]) * 0.5 * strength;
      const inv = 1.0 / Math.sqrt(dx * dx + dy * dy + 1.0);
      const idx = (y * size + x) * 4;
      img.data[idx] = Math.round((-dx * inv * 0.5 + 0.5) * 255);
      img.data[idx + 1] = Math.round((-dy * inv * 0.5 + 0.5) * 255);
      img.data[idx + 2] = Math.round((inv * 0.5 + 0.5) * 255);
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);

  const tex = new THREE.CanvasTexture(canvas);
  tex.wrapS = THREE.RepeatWrapping;
  tex.wrapT = THREE.RepeatWrapping;
  const tile = FABRIC_PRESETS[kind]?.tile || 36;
  tex.repeat.set(tile, tile);
  tex.colorSpace = THREE.NoColorSpace;
  fabricTextures.set(kind, tex);
  return tex;
}

const state = {
  garment: "lineup",
  camera: "front",
  shading: "studio",
  tshirtColor: "#e1cdbf",
  tshirtContrast: "dark",
  hoodieColor: "#28382e",
  hoodieContrast: "light",
  tshirtFabric: "mesh",
  hoodieFabric: "loopback",
};

let camAnim = null;
const prefersReducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function setCameraTarget(presetKey, immediate = false) {
  const preset = CAMERA_PRESETS[presetKey] || CAMERA_PRESETS.front;
  const box = new THREE.Box3();
  if (state.garment === "lineup" && lineupGroup.visible) {
    box.expandByObject(lineupGroup);
  } else {
    const activeNodes =
      state.garment === "tshirt"
        ? tshirtNodes
        : state.garment === "hoodie"
          ? hoodieNodes
          : [...tshirtNodes, ...hoodieNodes];
    for (const n of activeNodes) {
      if (n.visible) box.expandByObject(n);
    }
  }
  if (box.isEmpty()) return;

  const center = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const fovV = THREE.MathUtils.degToRad(camera.fov);
  const fovH = 2 * Math.atan(Math.tan(fovV / 2) * camera.aspect);
  const distV = size.y / (2 * Math.tan(fovV / 2));
  const distH = size.x / (2 * Math.tan(fovH / 2));
  const pad = state.garment === "lineup" ? 1.12 : 1.16;
  const dist = Math.max(distV, distH, size.z) * pad;

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

function styleGarmentMaterial(mat, meshName, colorHex, contrast, fabricKind) {
  if (!mat || !mat.name) return;
  const baseCol = new THREE.Color(colorHex);
  // Slightly deepen light fabrics so studio highlights never blow out Cream or Pastel Pink
  const hsl = { h: 0, s: 0, l: 0 };
  baseCol.getHSL(hsl);
  if (hsl.l > 0.65) {
    baseCol.setHSL(hsl.h, Math.min(1.0, hsl.s * 1.18), hsl.l * 0.88);
  }
  const ribCol = baseCol.clone().multiplyScalar(0.92);
  const preset = FABRIC_PRESETS[fabricKind] || FABRIC_PRESETS.loopback;
  const nTex = getFabricNormalTexture(fabricKind);

  if (mat.name === "tshirt_cotton" || mat.name === "hoodie_fleece") {
    mat.color.copy(baseCol);
    mat.roughness = preset.roughness;
    mat.sheen = preset.sheen;
    mat.sheenRoughness = preset.sheenRoughness;
    mat.sheenColor.copy(baseCol).lerp(new THREE.Color(0xffffff), 0.08);
    mat.normalMap = nTex;
    mat.normalScale.set(preset.normalScale, preset.normalScale);
  } else if (mat.name === "tshirt_collar_rib" || mat.name === "hoodie_rib") {
    mat.color.copy(ribCol);
    mat.roughness = Math.min(0.97, preset.roughness + 0.03);
    mat.sheen = preset.sheen * 0.85;
    mat.sheenRoughness = preset.sheenRoughness;
    mat.sheenColor.copy(ribCol).lerp(new THREE.Color(0xffffff), 0.06);
    mat.normalMap = nTex;
    mat.normalScale.set(preset.normalScale * 1.15, preset.normalScale * 1.15);
  } else if (meshName.includes("Decal_Front")) {
    mat.map = meshName.startsWith("Hoodie") ? decalTextures.hoodie_front : decalTextures.shirt_front;
    mat.color.setHex(0xffffff);
    mat.roughness = 0.85;
    mat.metalness = 0.0;
    mat.transparent = true;
    mat.alphaTest = 0.08;
    mat.depthWrite = false;
  } else if (meshName.includes("Decal_Back")) {
    mat.map = contrast === "dark" ? decalTextures.back_dark : decalTextures.back_light;
    mat.color.setHex(0xffffff);
    mat.roughness = 0.85;
    mat.metalness = 0.0;
    mat.transparent = true;
    mat.alphaTest = 0.08;
    mat.depthWrite = false;
  } else if (meshName.includes("Decal_SleeveFlag")) {
    mat.map = decalTextures.vn_flag;
    mat.color.setHex(0xffffff);
    mat.roughness = 0.85;
    mat.metalness = 0.0;
    mat.transparent = true;
    mat.alphaTest = 0.08;
    mat.depthWrite = false;
  }
  mat.needsUpdate = true;
}

function applyColors() {
  for (const [mesh, mat] of originalMaterials.entries()) {
    const isHoodie = mesh.name.startsWith("Hoodie");
    styleGarmentMaterial(
      mat,
      mesh.name,
      isHoodie ? state.hoodieColor : state.tshirtColor,
      isHoodie ? state.hoodieContrast : state.tshirtContrast,
      isHoodie ? state.hoodieFabric : state.tshirtFabric
    );
  }

  for (const inst of lineupInstances) {
    styleGarmentMaterial(
      inst.mat,
      inst.mesh.name,
      inst.colorHex,
      inst.contrast,
      inst.isHoodie ? state.hoodieFabric : state.tshirtFabric
    );
  }
}

function applyShading() {
  for (const [mesh, origMat] of originalMaterials.entries()) {
    if (mesh.name.includes("Decal") && state.shading !== "studio") {
      mesh.visible = false;
      continue;
    }
    if (state.garment !== "lineup") {
      const showGarment =
        state.garment === "both" ||
        (state.garment === "tshirt" && mesh.name.startsWith("TShirt")) ||
        (state.garment === "hoodie" && mesh.name.startsWith("Hoodie"));
      mesh.visible = showGarment;
    }
    mesh.material =
      state.shading === "studio" ? origMat : state.shading === "clay" ? clayMaterial : wireMaterial;
  }

  for (const inst of lineupInstances) {
    if (inst.mesh.name.includes("Decal")) {
      inst.mesh.visible = state.shading === "studio";
    }
    inst.mesh.material =
      state.shading === "studio" ? inst.mat : state.shading === "clay" ? clayMaterial : wireMaterial;
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

function buildLineupScene() {
  lineupGroup.clear();
  lineupInstances.length = 0;

  const tDx = centerOffsetFor(tshirtNodes);
  const hDx = centerOffsetFor(hoodieNodes);
  const colSpacing = 0.96;
  const startX = -1.5 * colSpacing;

  CHAM_COLORWAYS.forEach((cw, idx) => {
    const colX = startX + idx * colSpacing;

    // Top row: CHAM Hoodie
    for (const srcMesh of hoodieNodes) {
      const clone = srcMesh.clone();
      const origMat = originalMaterials.get(srcMesh);
      const cloneMat = origMat.clone();
      clone.material = cloneMat;
      clone.position.x = (originalPosX.get(srcMesh) ?? 0) + hDx + colX;
      clone.position.y += 0.36;
      clone.castShadow = false;
      clone.receiveShadow = false;
      lineupGroup.add(clone);
      lineupInstances.push({
        mesh: clone,
        mat: cloneMat,
        colorHex: cw.hex,
        contrast: cw.contrast,
        isHoodie: true,
      });
    }

    // Bottom row: CHAM Sports Shirt
    for (const srcMesh of tshirtNodes) {
      const clone = srcMesh.clone();
      const origMat = originalMaterials.get(srcMesh);
      const cloneMat = origMat.clone();
      clone.material = cloneMat;
      clone.position.x = (originalPosX.get(srcMesh) ?? 0) + tDx + colX;
      clone.position.y -= 0.44;
      clone.castShadow = false;
      clone.receiveShadow = false;
      lineupGroup.add(clone);
      lineupInstances.push({
        mesh: clone,
        mat: cloneMat,
        colorHex: cw.hex,
        contrast: cw.contrast,
        isHoodie: false,
      });
    }
  });
}

function applyGarmentSelection(immediate = false) {
  const isLineup = state.garment === "lineup";
  rootGroup.visible = !isLineup;
  lineupGroup.visible = isLineup;
  floor.visible = !isLineup;
  floor.position.y = -0.002;

  if (!isLineup) {
    for (const n of tshirtNodes) {
      n.visible = state.garment !== "hoodie" && (state.shading === "studio" || !n.name.includes("Decal"));
      n.position.x = originalPosX.get(n) ?? 0;
    }
    for (const n of hoodieNodes) {
      n.visible = state.garment !== "tshirt" && (state.shading === "studio" || !n.name.includes("Decal"));
      n.position.x = originalPosX.get(n) ?? 0;
    }

    if (state.garment === "tshirt") {
      const dx = centerOffsetFor(tshirtNodes);
      for (const n of tshirtNodes) n.position.x += dx;
    } else if (state.garment === "hoodie") {
      const dx = centerOffsetFor(hoodieNodes);
      for (const n of hoodieNodes) n.position.x += dx;
    }
  }

  const info = SPECS[state.garment] || SPECS.both;
  activeModelLabel.textContent = info.label;
  specQuads.textContent = info.tris;
  specVerts.textContent = info.verts;
  setCameraTarget(state.camera, immediate);
}

function computeStats(nodes, multiplier = 1) {
  let tris = 0;
  let verts = 0;
  for (const obj of nodes) {
    const g = obj.geometry;
    if (!g) continue;
    verts += g.attributes.position ? g.attributes.position.count : 0;
    tris += g.index ? Math.round(g.index.count / 3) : Math.round((g.attributes.position?.count || 0) / 3);
  }
  return {
    tris: (tris * multiplier).toLocaleString("en-US"),
    verts: (verts * multiplier).toLocaleString("en-US"),
  };
}

const loader = new GLTFLoader();
loader.load(
  "./shirt_and_hoodie.glb",
  (gltf) => {
    const model = gltf.scene;
    rootGroup.add(model);

    model.traverse((obj) => {
      if (!obj.isMesh) return;
      const isDecal = obj.name.includes("Decal");
      obj.castShadow = !isDecal;
      obj.receiveShadow = !isDecal;
      originalPosX.set(obj, obj.position.x);
      if (obj.material) {
        const srcMat = obj.material;
        const phys = new THREE.MeshPhysicalMaterial({
          name: srcMat.name,
          color: isDecal ? new THREE.Color(0xffffff) : srcMat.color ? srcMat.color.clone() : new THREE.Color(0xffffff),
          roughness: isDecal ? 0.68 : srcMat.roughness ?? 0.86,
          metalness: 0.0,
          sheen: isDecal ? 0.05 : 0.22,
          sheenRoughness: 0.62,
          sheenColor: new THREE.Color(0xffffff),
          transparent: isDecal,
          alphaTest: isDecal ? 0.08 : 0.0,
        });
        obj.material = phys;
        originalMaterials.set(obj, phys);
      }
      if (obj.name.startsWith("TShirt")) {
        tshirtNodes.push(obj);
      } else if (obj.name.startsWith("Hoodie")) {
        hoodieNodes.push(obj);
      }
    });

    buildLineupScene();

    SPECS.tshirt = { label: "CHAM Sports Shirt · 160 GSM", ...computeStats(tshirtNodes) };
    SPECS.hoodie = { label: "CHAM Hoodie · 350 GSM", ...computeStats(hoodieNodes) };
    SPECS.both = { label: "CHAM Hoodie & Sports Shirt Pair", ...computeStats([...tshirtNodes, ...hoodieNodes]) };
    SPECS.lineup = { label: "CHAM Collection · All 4 Colorways", ...computeStats([...tshirtNodes, ...hoodieNodes], 4) };

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

document.querySelectorAll("[data-fabric]").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.fabricTarget;
    document.querySelectorAll(`[data-fabric-target="${target}"]`).forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    if (target === "tshirt") {
      state.tshirtFabric = btn.dataset.fabric;
      document.getElementById("tshirtFabricName").textContent = btn.dataset.label;
    } else {
      state.hoodieFabric = btn.dataset.fabric;
      document.getElementById("hoodieFabricName").textContent = btn.dataset.label;
    }
    applyColors();
  });
});

document.querySelectorAll("[data-cycles-src]").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll("[data-cycles-src]").forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    const img = document.getElementById("cyclesImg");
    if (img) {
      img.src = btn.dataset.cyclesSrc;
      img.alt = btn.dataset.cyclesAlt || img.alt;
    }
  });
});

document.querySelectorAll(".swatch-btn").forEach((btn) => {
  btn.addEventListener("click", () => {
    const target = btn.dataset.target;
    document.querySelectorAll(`.swatch-btn[data-target="${target}"]`).forEach((b) => b.classList.remove("is-active"));
    btn.classList.add("is-active");
    if (target === "tshirt") {
      state.tshirtColor = btn.dataset.color;
      state.tshirtContrast = btn.dataset.contrast || "dark";
      document.getElementById("tshirtColorName").textContent = btn.dataset.name;
    } else {
      state.hoodieColor = btn.dataset.color;
      state.hoodieContrast = btn.dataset.contrast || "light";
      document.getElementById("hoodieColorName").textContent = btn.dataset.name;
    }
    if (state.garment === "lineup") {
      state.garment = target === "tshirt" ? "tshirt" : "hoodie";
      document.querySelectorAll("[data-garment]").forEach((b) => b.classList.toggle("is-active", b.dataset.garment === state.garment));
      applyGarmentSelection();
    }
    applyColors();
  });
});

const autoRotateToggle = document.getElementById("autoRotateToggle");
autoRotateToggle.addEventListener("change", () => {
  controls.autoRotate = autoRotateToggle.checked;
  controls.autoRotateSpeed = 1.4;
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
  state.camera = "front";
  document.querySelectorAll("[data-camera]").forEach((b) => b.classList.toggle("is-active", b.dataset.camera === "front"));
  setCameraTarget("front", false);
});

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
