/** 3D Scan Web UI — Three.js viewer + API client */
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { STLLoader } from 'three/addons/loaders/STLLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';

// ── API ──────────────────────────────────────────────────

const API = '/api';

async function api(path, opts = {}) {
  const res = await fetch(API + path, {
    headers: { 'Accept': 'application/json' },
    ...opts,
  });
  if (!res.ok) throw new Error((await res.json()).detail || res.statusText);
  return res.json();
}

async function uploadImages(files) {
  const fd = new FormData();
  files.forEach(f => fd.append('files', f));
  return api('/tasks', { method: 'POST', body: fd });
}

async function processTask(id) { return api(`/tasks/${id}/process`, { method: 'POST' }); }

// ── Three.js Viewer ──────────────────────────────────────

class Viewer {
  constructor(canvas) {
    this.canvas = canvas;
    this.mode = 'solid';
    this.partGroups = [];
    this.setup();
  }

  setup() {
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;

    this.renderer = new THREE.WebGLRenderer({ canvas: this.canvas, antialias: true, alpha: true });
    this.renderer.setSize(w, h);
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;

    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(0x111122);
    this.scene.fog = new THREE.Fog(0x111122, 30, 80);

    this.camera = new THREE.PerspectiveCamera(45, w / h, 0.5, 200);
    this.camera.position.set(15, 10, 15);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.08;
    this.controls.target.set(0, 3, 0);

    // Lights
    this.scene.add(new THREE.AmbientLight(0x404060, 2));
    const key = new THREE.DirectionalLight(0xffffff, 3);
    key.position.set(10, 20, 10);
    key.castShadow = true;
    this.scene.add(key);
    const fill = new THREE.DirectionalLight(0x8888ff, 1.5);
    fill.position.set(-10, 5, -5);
    this.scene.add(fill);
    const rim = new THREE.DirectionalLight(0xff8844, 1);
    rim.position.set(0, 2, -15);
    this.scene.add(rim);

    // Grid
    const grid = new THREE.GridHelper(30, 30, 0x333355, 0x222244);
    this.scene.add(grid);

    // Resize
    window.addEventListener('resize', () => this.onResize());
    this.animate();
  }

  animate() {
    requestAnimationFrame(() => this.animate());
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  onResize() {
    const w = this.canvas.parentElement.clientWidth;
    const h = this.canvas.parentElement.clientHeight;
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }

  clear() {
    this.partGroups = [];
    while (this.scene.children.length > 5) { // keep lights + grid
      const c = this.scene.children[5];
      this.scene.remove(c);
    }
  }

  async loadSTL(url) {
    this.clear();
    const loader = new STLLoader();
    const geom = await loader.loadAsync(url);
    this._addMesh(geom, 0x4a90d9);
    this._centerView(geom);
  }

  async loadParts(parts) {
    this.clear();
    this.partGroups = [];
    const loader = new STLLoader();
    const colors = [0xe94560, 0x4a90d9, 0x4caf50, 0xff9800, 0x9c27b0, 0x00bcd4, 0xffeb3b, 0x795548];

    for (let i = 0; i < parts.length; i++) {
      try {
        const geom = await loader.loadAsync(API + `/tasks/${parts.taskId}/download/${parts[i].filename}`);
        const group = this._addMesh(geom, colors[i % colors.length]);
        group.userData = { partIndex: i, name: parts[i].name };
        this.partGroups.push(group);
      } catch (e) { /* skip missing files */ }
    }

    if (this.partGroups.length > 0) {
      const box = new THREE.Box3();
      this.partGroups.forEach(g => box.expandByObject(g));
      this._fitView(box);
    }
  }

  _addMesh(geom, colorHex) {
    const mat = new THREE.MeshStandardMaterial({
      color: colorHex,
      roughness: 0.6,
      metalness: 0.1,
      flatShading: false,
    });
    if (this.mode === 'wireframe') {
      mat.wireframe = true;
      mat.opacity = 0.3;
      mat.transparent = true;
    }
    const mesh = new THREE.Mesh(geom, mat);
    const group = new THREE.Group();
    group.add(mesh);
    this.scene.add(group);
    return group;
  }

  setMode(mode) {
    this.mode = mode;
    this.scene.traverse(child => {
      if (child.isMesh && child.material.isMaterial) {
        if (mode === 'wireframe') {
          child.material.wireframe = true;
          child.material.opacity = 0.3;
          child.material.transparent = true;
        } else {
          child.material.wireframe = false;
          child.material.opacity = 1;
          child.material.transparent = false;
        }
      }
    });

    // 模块模式: 高亮选中
    if (mode === 'parts') {
      this.partGroups.forEach((g, i) => {
        g.children.forEach(c => {
          if (c.isMesh) c.material.opacity = 0.6;
          c.material.transparent = true;
        });
      });
    }
    if (mode === 'solid') {
      this.partGroups.forEach(g => {
        g.children.forEach(c => {
          if (c.isMesh) { c.material.opacity = 1; c.material.transparent = false; }
        });
      });
    }
  }

  highlightPart(index) {
    this.partGroups.forEach((g, i) => {
      const isTarget = i === index;
      g.children.forEach(c => {
        if (c.isMesh) {
          c.material.opacity = isTarget ? 1 : 0.15;
          c.material.transparent = true;
        }
      });
    });
  }

  _centerView(geom) {
    geom.computeBoundingBox();
    const box = geom.boundingBox;
    this._fitView(box);
  }

  _fitView(box) {
    const center = new THREE.Vector3();
    box.getCenter(center);
    this.controls.target.copy(center);
    const size = new THREE.Vector3();
    box.getSize(size);
    const maxDim = Math.max(size.x, size.y, size.z);
    this.camera.position.set(center.x + maxDim, center.y + maxDim * 0.7, center.z + maxDim);
    this.camera.lookAt(center);
  }

  resetCamera() {
    this.camera.position.set(15, 10, 15);
    this.controls.target.set(0, 3, 0);
    this.controls.update();
  }
}

// ── App State ────────────────────────────────────────────

let viewer;
let currentTaskId = null;
const PART_COLORS = ['#e94560','#4a90d9','#4caf50','#ff9800','#9c27b0','#00bcd4','#ffeb3b','#795548'];

// ── Init ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  viewer = new Viewer(document.getElementById('viewer-canvas'));
  setupUpload();
  setupViewerControls();
  setupExportButtons();
  pollTasks();
});

// ── Upload ───────────────────────────────────────────────

function setupUpload() {
  const dropZone = document.getElementById('drop-zone');
  const fileInput = document.getElementById('file-input');
  const previewStrip = document.getElementById('preview-strip');
  const uploadActions = document.getElementById('upload-actions');
  const fileCount = document.getElementById('file-count');
  const btnUpload = document.getElementById('btn-upload');
  let files = [];

  dropZone.addEventListener('click', () => fileInput.click());
  dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('drag-over'); });
  dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
  dropZone.addEventListener('drop', e => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    handleFiles(Array.from(e.dataTransfer.files));
  });
  fileInput.addEventListener('change', () => handleFiles(Array.from(fileInput.files)));

  function handleFiles(newFiles) {
    files = newFiles.filter(f => f.type.startsWith('image/'));
    previewStrip.innerHTML = '';
    files.forEach(f => {
      const img = document.createElement('img');
      img.src = URL.createObjectURL(f);
      previewStrip.appendChild(img);
    });
    fileCount.textContent = `${files.length} 张`;
    uploadActions.style.display = 'flex';
  }

  btnUpload.addEventListener('click', async () => {
    if (files.length === 0) return;
    setProgress(true, 10);
    try {
      const { task_id } = await uploadImages(files);
      currentTaskId = task_id;
      setProgress(true, 30);
      const result = await processTask(task_id);
      setProgress(false, 0);
      handleTaskResult(task_id, result);
    } catch (e) {
      setProgress(false, 0);
      alert('处理失败: ' + e.message);
    }
  });
}

// ── Viewer Controls ──────────────────────────────────────

function setupViewerControls() {
  document.querySelectorAll('#viewer-controls .btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('#viewer-controls .btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      const mode = btn.dataset.mode;
      if (mode === 'reset-camera') {
        viewer.resetCamera();
      } else {
        viewer.setMode(mode);
      }
    });
  });
}

// ── Export ────────────────────────────────────────────────

function setupExportButtons() {
  document.getElementById('btn-download-all').addEventListener('click', async () => {
    if (!currentTaskId) return;
    // 触发下载所有 STL
    const exportStatus = document.getElementById('export-status');
    exportStatus.textContent = '准备下载...';
    // 生产环境用 zip 打包
    window.open(`${API}/tasks/${currentTaskId}/download/model.stl`, '_blank');
    exportStatus.textContent = '已触发下载';
  });
}

// ── Task Result Handling ─────────────────────────────────

function handleTaskResult(taskId, result) {
  updateTaskStatus('完成');
  if (result.output) {
    // Load preview mesh
    const stlUrl = `${API}/tasks/${taskId}/download/model.stl`;
    viewer.loadSTL(stlUrl).then(() => {
      updateViewerInfo('-', '-', '-');
    });

    // Update parts if decomposition was done
    if (result.parts && result.parts.length > 0) {
      renderPartList(result.parts);
      updateViewerInfo('-', '-', result.parts.length);
    }

    // Enable export
    document.getElementById('btn-download-all').disabled = false;
  }
}

function updateViewerInfo(verts, faces, parts) {
  document.getElementById('info-verts').textContent = verts;
  document.getElementById('info-faces').textContent = faces;
  document.getElementById('info-parts').textContent = parts;
}

function updateTaskStatus(text) {
  document.getElementById('task-status').textContent = text;
}

// ── Part List ─────────────────────────────────────────────

function renderPartList(parts) {
  const container = document.getElementById('part-list');
  container.innerHTML = '';
  parts.forEach((part, i) => {
    const div = document.createElement('div');
    div.className = 'part-item';
    div.innerHTML = `
      <span class="color-dot" style="background:${PART_COLORS[i % PART_COLORS.length]}"></span>
      <span>${part.name}</span>
      <span style="margin-left:auto;font-size:11px;color:var(--text-dim)">${part.face_count || '?'}面</span>
    `;
    div.addEventListener('click', () => {
      document.querySelectorAll('.part-item').forEach(el => el.classList.remove('selected'));
      div.classList.add('selected');
      viewer.highlightPart(i);
      document.getElementById('btn-download-part').disabled = false;
    });
    container.appendChild(div);
  });
  document.getElementById('btn-download-all').disabled = false;
}

// ── Progress ──────────────────────────────────────────────

function setProgress(show, pct) {
  const bar = document.getElementById('progress-bar');
  const fill = document.getElementById('progress-fill');
  const text = document.getElementById('progress-text');
  bar.style.display = show ? 'block' : 'none';
  fill.style.width = pct + '%';
  text.textContent = pct + '%';
  if (show) updateTaskStatus('处理中...');
}

// ── Task Poll ─────────────────────────────────────────────

async function pollTasks() {
  const container = document.getElementById('task-list');
  setInterval(async () => {
    // 简单显示当前任务状态
    if (currentTaskId) {
      try {
        const task = await api(`/tasks/${currentTaskId}`);
        container.innerHTML = `<div class="task-item">
          <span class="status-dot ${task.status}"></span>
          <span>${task.id}</span>
          <span style="margin-left:auto">${task.status}</span>
        </div>`;
      } catch (e) { /* task expired */ }
    }
  }, 5000);
}