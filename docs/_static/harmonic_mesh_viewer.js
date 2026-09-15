// Generic viewer for "harmonic mesh" animations exported by
// export_animation_data() in docs/examples/src/B7_boat_animation.py.
//
// The exported .bin file is a small binary blob describing any number of
// meshes ("components"), each with a mean shape and a single complex
// (real + imaginary) 3D motion vector per vertex. This viewer has no
// knowledge of what a component physically represents (a boat hull, a
// free surface, ...) — any mesh exported the same way can be plugged in
// by pointing initHarmonicMeshViewer() at a different .bin file.
//
// Binary format (little-endian):
//   uint32 num_components
//   float32 omega
//   float32[3] camera_position  (a suggested initial camera position)
//   float32[3] camera_target    (a suggested initial camera look-at point)
//   per component:
//     uint32 n_vertices, uint32 n_faces
//     float32[n_vertices*3]  mean_vertices
//     uint32[n_faces*3]      triangle_indices
//     float32[n_vertices*3]  motion_real
//     float32[n_vertices*3]  motion_imag
//
// At time t, a vertex position is:
//   pos = mean + motion_real * cos(omega * t) + motion_imag * sin(omega * t)
// (the real part of mean + motion * exp(-i * omega * t), matching
// capytaine.ui.vedo_animations.Animation.update()).

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

// Capytaine logo palette, used for the height-based color gradient: low
// points are navy, mid-height is cyan, peaks are gold.
const HEIGHT_GRADIENT = [
    [0x21, 0x35, 0x5d], // navy
    [0x05, 0xa9, 0xd9], // cyan
    [0xfb, 0xc0, 0x2d], // gold
];

function heightToColor(t, out) {
    // t expected in [-1, 1]; clamps outside that range.
    const u = Math.max(0, Math.min(1, (t + 1) / 2)); // -> [0, 1]
    const [c0, c1] = u < 0.5 ? [HEIGHT_GRADIENT[0], HEIGHT_GRADIENT[1]] : [HEIGHT_GRADIENT[1], HEIGHT_GRADIENT[2]];
    const localT = u < 0.5 ? u / 0.5 : (u - 0.5) / 0.5;
    out[0] = (c0[0] + (c1[0] - c0[0]) * localT) / 255;
    out[1] = (c0[1] + (c1[1] - c0[1]) * localT) / 255;
    out[2] = (c0[2] + (c1[2] - c0[2]) * localT) / 255;
}

function parseHarmonicMeshData(buffer) {
    const view = new DataView(buffer);
    let offset = 0;

    const numComponents = view.getUint32(offset, true); offset += 4;
    const omega = view.getFloat32(offset, true); offset += 4;
    const cameraPosition = [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)];
    offset += 12;
    const cameraTarget = [view.getFloat32(offset, true), view.getFloat32(offset + 4, true), view.getFloat32(offset + 8, true)];
    offset += 12;

    const components = [];
    for (let i = 0; i < numComponents; i++) {
        const nVertices = view.getUint32(offset, true); offset += 4;
        const nFaces = view.getUint32(offset, true); offset += 4;

        const meanVertices = new Float32Array(buffer, offset, nVertices * 3);
        offset += nVertices * 3 * 4;
        const triangleIndices = new Uint32Array(buffer, offset, nFaces * 3);
        offset += nFaces * 3 * 4;
        const motionReal = new Float32Array(buffer, offset, nVertices * 3);
        offset += nVertices * 3 * 4;
        const motionImag = new Float32Array(buffer, offset, nVertices * 3);
        offset += nVertices * 3 * 4;

        components.push({ nVertices, nFaces, meanVertices, triangleIndices, motionReal, motionImag });
    }

    return { omega, cameraPosition, cameraTarget, components };
}

// The largest oscillation amplitude (along `axis`) reached by any vertex of
// the component, used to normalize per-vertex height into [-1, 1] for the
// color gradient. `axis` is 0/1/2 for x/y/z (default z, the usual "up" axis).
function computeHeightScale(component, axis = 2) {
    let maxAmplitude = 0;
    for (let i = axis; i < component.motionReal.length; i += 3) {
        const re = component.motionReal[i];
        const im = component.motionImag[i];
        const amplitude = Math.sqrt(re * re + im * im);
        if (amplitude > maxAmplitude) maxAmplitude = amplitude;
    }
    return maxAmplitude || 1;
}

function buildComponentGroup(component, { color, opacity, colorByHeight, wireframe, wireframeColor }) {
    const geometry = new THREE.BufferGeometry();
    // Copy (not view) the mean vertices: this array is mutated every frame.
    geometry.setAttribute('position', new THREE.BufferAttribute(component.meanVertices.slice(), 3));
    geometry.setIndex(new THREE.BufferAttribute(component.triangleIndices, 1));
    geometry.computeVertexNormals();

    const materialOptions = {
        transparent: opacity < 1,
        opacity,
        side: THREE.DoubleSide,
        shininess: 30,
    };
    if (colorByHeight) {
        // Vertex colors are multiplied by the material color, so use white
        // to show the computed gradient unmodified.
        materialOptions.color = 0xffffff;
        materialOptions.vertexColors = true;
        geometry.setAttribute('color', new THREE.BufferAttribute(new Float32Array(component.nVertices * 3), 3));
        component.heightScale = computeHeightScale(component);
    } else {
        materialOptions.color = color;
    }

    const group = new THREE.Group();
    group.add(new THREE.Mesh(geometry, new THREE.MeshPhongMaterial(materialOptions)));
    if (wireframe) {
        group.add(new THREE.Mesh(geometry, new THREE.MeshBasicMaterial({
            color: wireframeColor,
            wireframe: true,
            transparent: true,
            opacity: 0.35,
        })));
    }
    return { geometry, group };
}

function updateComponent(component, geometry, t, omega) {
    const cosWt = Math.cos(omega * t);
    const sinWt = Math.sin(omega * t);
    const positions = geometry.attributes.position.array;
    const { meanVertices, motionReal, motionImag } = component;

    for (let i = 0; i < positions.length; i++) {
        positions[i] = meanVertices[i] + motionReal[i] * cosWt + motionImag[i] * sinWt;
    }
    geometry.attributes.position.needsUpdate = true;
    geometry.computeVertexNormals();

    if (component.heightScale) {
        const colors = geometry.attributes.color.array;
        const rgb = [0, 0, 0];
        for (let v = 0, i = 2; i < positions.length; v++, i += 3) {
            const height = motionReal[i] * cosWt + motionImag[i] * sinWt;
            heightToColor(height / component.heightScale, rgb);
            colors[v * 3] = rgb[0];
            colors[v * 3 + 1] = rgb[1];
            colors[v * 3 + 2] = rgb[2];
        }
        geometry.attributes.color.needsUpdate = true;
    }
}

/**
 * Set up an interactive, looping harmonic-mesh animation inside `canvas`,
 * loading its scene data from `binUrl`.
 *
 * `options`:
 *   - colors:          array of hex fill colors, one per component (cycled if shorter);
 *                       ignored for a component with colorByHeight set
 *   - opacities:        array of opacities (0-1), one per component (cycled if shorter)
 *   - colorByHeight:    array of booleans, one per component (cycled if shorter);
 *                       true colors the component by its own vertical motion using
 *                       a navy -> cyan -> gold gradient instead of a flat color.
 *                       Default: only the 2nd component (the free surface, in the
 *                       boat+wave scene) is colored by height.
 *   - wireframe:        array of booleans, one per component (cycled if shorter);
 *                       whether to overlay that component's mesh wireframe.
 *                       Default: only the 1st component (the floating body).
 *   - wireframeColors:  array of hex colors for each component's wireframe overlay
 *   - cameraPosition:  [x, y, z]; defaults to the suggestion packed in the
 *                       .bin file (see export_animation_data() in B7_boat_animation.py)
 *   - cameraTarget:     [x, y, z] look-at point; also defaults to the packed suggestion
 *   - autoRotate:       whether OrbitControls should idle-rotate the camera (default false)
 *   - onError:          called with the Error if `binUrl` can't be fetched or parsed,
 *                       so the caller can fall back to a static image/video
 *
 * Returns `{ start, stop }` to manually control the render loop, though by
 * default it already starts/stops itself based on the canvas' visibility.
 */
export function initHarmonicMeshViewer(canvas, binUrl, options = {}) {
    const colors = options.colors || [0x21355d, 0x05a9d9];
    const opacities = options.opacities || [1.0, 0.92];
    const colorByHeight = options.colorByHeight || [false, true];
    const wireframe = options.wireframe || [true, false];
    const wireframeColors = options.wireframeColors || [0xfbc02d, 0x21355d];

    const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 2000);
    camera.up.set(0, 0, 1);
    // Placeholder until the .bin file's suggested camera is loaded (or the
    // caller's own `options.cameraPosition`/`cameraTarget` are applied below).
    camera.position.set(70, 70, 100);
    camera.lookAt(0, 0, 0);

    scene.add(new THREE.AmbientLight(0xffffff, 0.7));
    const sun = new THREE.DirectionalLight(0xffffff, 0.9);
    sun.position.set(0, 0, 100);
    scene.add(sun);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.minDistance = 20;
    controls.maxDistance = 400;
    controls.autoRotate = !!options.autoRotate;
    controls.autoRotateSpeed = 0.6;

    let animationData = null;
    let geometries = [];
    let running = false;
    let rafId = null;
    const clockStart = performance.now();

    function resize() {
        const width = canvas.clientWidth;
        const height = canvas.clientHeight;
        if (width === 0 || height === 0) return;
        renderer.setSize(width, height, false);
        camera.aspect = width / height;
        camera.updateProjectionMatrix();
    }

    function frame(now) {
        if (!running) return;
        rafId = requestAnimationFrame(frame);
        const t = (now - clockStart) / 1000;
        if (animationData) {
            for (let i = 0; i < geometries.length; i++) {
                updateComponent(animationData.components[i], geometries[i], t, animationData.omega);
            }
        }
        controls.update();
        renderer.render(scene, camera);
    }

    function start() {
        if (running) return;
        running = true;
        rafId = requestAnimationFrame(frame);
    }

    function stop() {
        running = false;
        if (rafId !== null) cancelAnimationFrame(rafId);
    }

    fetch(binUrl)
        .then((response) => {
            if (!response.ok) {
                throw new Error(`${response.status} ${response.statusText}`);
            }
            return response.arrayBuffer();
        })
        .then((buffer) => {
            animationData = parseHarmonicMeshData(buffer);
            geometries = animationData.components.map((component, i) => {
                const { geometry, group } = buildComponentGroup(component, {
                    color: colors[i % colors.length],
                    opacity: opacities[i % opacities.length],
                    colorByHeight: colorByHeight[i % colorByHeight.length],
                    wireframe: wireframe[i % wireframe.length],
                    wireframeColor: wireframeColors[i % wireframeColors.length],
                });
                scene.add(group);
                return geometry;
            });

            const finalCameraPosition = options.cameraPosition || animationData.cameraPosition;
            const finalCameraTarget = options.cameraTarget || animationData.cameraTarget;
            camera.position.set(...finalCameraPosition);
            camera.lookAt(...finalCameraTarget);
            controls.target.set(...finalCameraTarget);
            controls.update();

            resize();
        })
        .catch((error) => {
            console.error(`harmonic_mesh_viewer: failed to load "${binUrl}"`, error);
            stop();
            if (typeof options.onError === 'function') {
                options.onError(error);
            }
        });

    window.addEventListener('resize', resize);
    resize();

    if ('IntersectionObserver' in window) {
        const observer = new IntersectionObserver(
            (entries) => entries.forEach((entry) => (entry.isIntersecting ? start() : stop())),
            { threshold: 0.05 },
        );
        observer.observe(canvas);
    } else {
        start();
    }

    return { start, stop };
}
