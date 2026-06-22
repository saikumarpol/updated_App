// utils/onnx-eye-detector.ts
// ─────────────────────────────────────────────────────────────────────────────
// Full ONNX Runtime React Native integration for eye detection.
//
// Model I/O:
//   Input  : "input"          float32[1, 3, 320, 320]  NCHW  RGB  0–1
//   Output : "cls_logits"     float32[1, 3234, 2]      raw class logits
//            "bbox_regression" float32[1, 3234, 4]     box deltas or direct coords
// ─────────────────────────────────────────────────────────────────────────────

import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { Asset } from "expo-asset";
import * as FileSystem from "expo-file-system";

// ── Model constants ────────────────────────────────────────────────────────────
export const MODEL_INPUT_SIZE = 320;
const NUM_ANCHORS = 3234;
const NUM_CLASSES = 2; // 0 = background, 1 = eye
const EYE_CLASS_IDX = 1;

// ── Detection thresholds ───────────────────────────────────────────────────────
export const CONF_THRESHOLD = 0.25;
const IOU_THRESHOLD = 0.45;
const MAX_DETECTIONS = 100; // cap candidates before NMS

// ── Anchor config ──────────────────────────────────────────────────────────────
// These produce 3234 anchors for a 320×320 input.
// Strides 8,16,32,64 → grids 40,20,10,5  (+3×3 + 1×1 extra levels)
// 3 anchor sizes per cell × (1600+400+100+25+9+1) = 3 × 2135... adjust if wrong.
//
// If your model was trained with a different anchor config, replace ANCHOR_SPECS.
// The easiest way to verify: run inference and check if decoded boxes make sense.
const ANCHOR_SPECS: Array<{
  stride: number;
  sizes: number[];
  ratios: number[];
}> = [
  { stride: 8,   sizes: [16, 32, 64],       ratios: [1.0] }, // 40×40 grid → 40*40*3 = 4800 ← too many
  // ↑ A single-ratio, 3-size-per-level config that totals 3234:
  // 40²×1 + 20²×1 + 10²×4 + 5²×4 + 3²×4 + 1²×4 = 1600+400+400+100+36+4 = 2540... still off
  //
  // Most likely config for 3234:
  // Feature maps: [40,20,10,5,3,1], anchors per cell: [3,3,3,3,3,3]
  // 3*(1600+400+100+25+9+1) = 3*2135 = 6405 — not it either.
  //
  // Exact match: 3234 = 2 * (40² + 20² + 10² + ... )? 
  // 2*(1600+400+100+25+9+1) = 2*2135 = 4270 — no.
  //
  // Or: grids [34,17,9,5,3,1], 3 anchors each:
  // 3*(34²+17²+9²+5²+3²+1²) = 3*(1156+289+81+25+9+1) = 3*1561 = 4683 — no.
  //
  // Best guess for exactly 3234:
  // SSD-style: 38²*4 + 19²*6 + 10²*6 + 5²*6 + 3²*4 + 1²*4
  // = 5776+2166+600+150+36+4 = 8732 — VGG SSD default, not it.
  //
  // RetinaNet FPN P3-P7 with 3 ratios × 3 scales = 9 anchors per cell:
  // 9*(40²+20²+10²+5²+3²) = 9*(1600+400+100+25+9) = 9*2134 = 19206 — no.
  //
  // Most probable: 3234 = 6 feature levels × (specific grid sizes) × 1 anchor:
  // All grids with 1 anchor: sum of grid²s must = 3234.
  // √3234 ≈ 56.9 — single level? unlikely.
  //
  // 🔑 RECOMMENDED: set IS_DELTA_ENCODED = false first (direct coords).
  //    If that works, no anchor config needed at all.
  //    Only switch to true + fix anchor config if direct coords give wrong boxes.
];

// ─────────────────────────────────────────────────────────────────────────────
// IMPORTANT FLAG
// Set to false first. Only flip to true if direct-coord decoding gives
// clearly wrong bounding boxes (wildly off-screen, tiny, chaotic).
// ─────────────────────────────────────────────────────────────────────────────
export const IS_DELTA_ENCODED = false;

// ── ImageNet normalization (optional) ─────────────────────────────────────────
// Try WITHOUT first (just /255). Only enable if confidence is always < 0.05.
const USE_IMAGENET_NORM = false;
const IMAGENET_MEAN = [0.485, 0.456, 0.406];
const IMAGENET_STD  = [0.229, 0.224, 0.225];

// ── Types ─────────────────────────────────────────────────────────────────────
export interface EyeDetection {
  /** Normalized 0–1 relative to MODEL_INPUT_SIZE (320) */
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  /** Sigmoid confidence, 0–1 */
  score: number;
  /** Center x, normalized */
  cx: number;
  /** Center y, normalized */
  cy: number;
  /** Width, normalized */
  w: number;
  /** Height, normalized */
  h: number;
}

// ─────────────────────────────────────────────────────────────────────────────
// PREPROCESSING
// Input:  Uint8Array  shape [H, W, 3]  HWC  RGB  uint8 0–255
// Output: Float32Array shape [1, 3, H, W]  NCHW  float32
// ─────────────────────────────────────────────────────────────────────────────
export function hwcUint8ToNchwFloat32(
  pixels: Uint8Array,
  h: number = MODEL_INPUT_SIZE,
  w: number = MODEL_INPUT_SIZE,
): Float32Array {
  const out = new Float32Array(h * w * 3);

  for (let c = 0; c < 3; c++) {
    for (let row = 0; row < h; row++) {
      for (let col = 0; col < w; col++) {
        const srcIdx = (row * w + col) * 3 + c;
        const dstIdx = c * h * w + row * w + col;
        let val = pixels[srcIdx] / 255.0;

        if (USE_IMAGENET_NORM) {
          val = (val - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }

        out[dstIdx] = val;
      }
    }
  }

  return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// ANCHOR GENERATION
// Generates [N, 4] anchors as [cx, cy, w, h] in normalized 0–1 space.
// Only used when IS_DELTA_ENCODED = true.
// ─────────────────────────────────────────────────────────────────────────────
let _anchorCache: Float32Array | null = null;

function generateAnchors(): Float32Array {
  if (_anchorCache) return _anchorCache;

  // Fallback simple anchor config that *approximates* 3234 anchors.
  // Replace with your model's exact training anchor config.
  const anchors: number[] = [];

  // Config A: FPN-style, 3 scales, 1 ratio, strides [8,16,32,64,106,320]
  // Grids: [40,20,10,5,3,1], 3 anchors/cell → 3*(1600+400+100+25+9+1)=6405 (too many)

  // Config B: 2 anchors/cell, grids [40,20,10,5,3,1]
  // → 2*2135 = 4270 (too many)

  // Config C: likely 1 anchor/cell, manually tuned grid sizes
  // Since 3234 is prime-ish (3234 = 2 × 3 × 7^2 × 11), the architecture
  // may not use square grids at all. Use direct coords (IS_DELTA_ENCODED=false).

  // This stub produces *something* so the code compiles.
  // You MUST replace with the real anchor config from your model's training script.
  const STRIDES_CFG = [8, 16, 32, 64, 128, 256];
  const SIZES_CFG   = [
    [16,  32],
    [32,  64],
    [64,  128],
    [128, 256],
    [256, 512],
    [512, 1024],
  ];

  for (let lvl = 0; lvl < STRIDES_CFG.length; lvl++) {
    const stride   = STRIDES_CFG[lvl];
    const gridSize = Math.floor(MODEL_INPUT_SIZE / stride);

    for (let gy = 0; gy < gridSize; gy++) {
      for (let gx = 0; gx < gridSize; gx++) {
        const cx = (gx + 0.5) * stride / MODEL_INPUT_SIZE;
        const cy = (gy + 0.5) * stride / MODEL_INPUT_SIZE;

        for (const sz of SIZES_CFG[lvl]) {
          const w = sz / MODEL_INPUT_SIZE;
          const h = sz / MODEL_INPUT_SIZE;
          anchors.push(cx, cy, w, h);
        }
      }
    }
  }

  // Warn if anchor count doesn't match
  const actualAnchors = anchors.length / 4;
  if (actualAnchors !== NUM_ANCHORS) {
    console.warn(
      `[ONNX] ⚠️  Anchor count mismatch: generated ${actualAnchors}, expected ${NUM_ANCHORS}. ` +
      `Set IS_DELTA_ENCODED=false to skip anchor decoding, or fix ANCHOR_SPECS.`,
    );
  }

  _anchorCache = new Float32Array(anchors);
  return _anchorCache;
}

// ─────────────────────────────────────────────────────────────────────────────
// MATH UTILS
// ─────────────────────────────────────────────────────────────────────────────
function sigmoid(x: number): number {
  return 1.0 / (1.0 + Math.exp(-x));
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, v));
}

function iou(
  a: readonly [number, number, number, number],
  b: readonly [number, number, number, number],
): number {
  const ix1 = Math.max(a[0], b[0]);
  const iy1 = Math.max(a[1], b[1]);
  const ix2 = Math.min(a[2], b[2]);
  const iy2 = Math.min(a[3], b[3]);

  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const iArea = iw * ih;
  if (iArea === 0) return 0;

  const aArea = (a[2] - a[0]) * (a[3] - a[1]);
  const bArea = (b[2] - b[0]) * (b[3] - b[1]);

  return iArea / (aArea + bArea - iArea + 1e-7);
}

// ─────────────────────────────────────────────────────────────────────────────
// NMS (greedy, descending score)
// ─────────────────────────────────────────────────────────────────────────────
interface RawDetection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
}

function nms(dets: RawDetection[], iouThresh: number): RawDetection[] {
  dets.sort((a, b) => b.score - a.score);
  const kept: RawDetection[] = [];

  for (const d of dets) {
    let suppressed = false;
    for (const k of kept) {
      if (iou([d.x1, d.y1, d.x2, d.y2], [k.x1, k.y1, k.x2, k.y2]) > iouThresh) {
        suppressed = true;
        break;
      }
    }
    if (!suppressed) kept.push(d);
  }

  return kept;
}

// ─────────────────────────────────────────────────────────────────────────────
// OUTPUT DECODER
// Converts raw model tensors → EyeDetection | null
//
// isDeltaEncoded = false  →  bbox_regression = [x1, y1, x2, y2]  normalized 0–1
// isDeltaEncoded = true   →  bbox_regression = [dx, dy, dw, dh]  delta vs anchors
// ─────────────────────────────────────────────────────────────────────────────
export function decodeOutputs(
  clsLogits: Float32Array,      // flat [NUM_ANCHORS * NUM_CLASSES]
  bboxRegression: Float32Array, // flat [NUM_ANCHORS * 4]
  isDeltaEncoded: boolean = IS_DELTA_ENCODED,
  confThreshold: number = CONF_THRESHOLD,
): EyeDetection | null {
  const anchors = isDeltaEncoded ? generateAnchors() : null;
  const candidates: RawDetection[] = [];

  for (let i = 0; i < NUM_ANCHORS; i++) {
    if (candidates.length >= MAX_DETECTIONS) break;

    // ── Score ──────────────────────────────────────────────────────────────
    const eyeLogit = clsLogits[i * NUM_CLASSES + EYE_CLASS_IDX];
    const score    = sigmoid(eyeLogit);
    if (score < confThreshold) continue;

    // ── Box ────────────────────────────────────────────────────────────────
    const r0 = bboxRegression[i * 4 + 0];
    const r1 = bboxRegression[i * 4 + 1];
    const r2 = bboxRegression[i * 4 + 2];
    const r3 = bboxRegression[i * 4 + 3];

    let x1: number, y1: number, x2: number, y2: number;

    if (!isDeltaEncoded || !anchors) {
      // ── Direct normalized [x1, y1, x2, y2] ──────────────────────────────
      x1 = clamp(Math.min(r0, r2), 0, 1);
      y1 = clamp(Math.min(r1, r3), 0, 1);
      x2 = clamp(Math.max(r0, r2), 0, 1);
      y2 = clamp(Math.max(r1, r3), 0, 1);
    } else {
      // ── Delta decoding vs anchor ─────────────────────────────────────────
      const acx = anchors[i * 4 + 0]; // anchor center x (0–1)
      const acy = anchors[i * 4 + 1]; // anchor center y (0–1)
      const aw  = anchors[i * 4 + 2]; // anchor width    (0–1)
      const ah  = anchors[i * 4 + 3]; // anchor height   (0–1)

      // FCOS-style (cx+dx, cy+dy):
      const cx = clamp(acx + r0 * aw, 0, 1);
      const cy = clamp(acy + r1 * ah, 0, 1);
      const w  = clamp(aw  * Math.exp(r2), 0, 2); // allow slight overflow before clamp
      const h  = clamp(ah  * Math.exp(r3), 0, 2);

      x1 = clamp(cx - w / 2, 0, 1);
      y1 = clamp(cy - h / 2, 0, 1);
      x2 = clamp(cx + w / 2, 0, 1);
      y2 = clamp(cy + h / 2, 0, 1);
    }

    const bw = x2 - x1;
    const bh = y2 - y1;

    // Filter degenerate boxes
    if (bw < 0.001 || bh < 0.001) continue;

    candidates.push({ x1, y1, x2, y2, score });
  }

  if (candidates.length === 0) return null;

  // ── NMS ──────────────────────────────────────────────────────────────────────
  const kept = nms(candidates, IOU_THRESHOLD);
  const best = kept[0];
  if (!best) return null;

  const cx = (best.x1 + best.x2) / 2;
  const cy = (best.y1 + best.y2) / 2;
  const w  = best.x2 - best.x1;
  const h  = best.y2 - best.y1;

  return { ...best, cx, cy, w, h };
}

// ─────────────────────────────────────────────────────────────────────────────
// ASSET RESOLUTION
// onnxruntime-react-native requires a real file path string, NOT a Metro
// require() number. expo-asset downloads/copies the bundled file to the
// device filesystem and gives us the local URI.
// ─────────────────────────────────────────────────────────────────────────────
async function resolveModelPath(moduleId: number): Promise<string> {
  const [asset] = await Asset.loadAsync(moduleId);

  // localUri is set on-device after loadAsync (preferred — no extra copy)
  if (asset.localUri) {
    // ONNX Runtime wants a bare path, not a file:// URI
    return asset.localUri.replace(/^file:\/\//, "");
  }

  // Fallback: download from bundle URI to the app's cache dir
  if (!asset.uri) {
    throw new Error("[ONNX] Asset has no URI after loadAsync");
  }

  const destDir  = FileSystem.cacheDirectory ?? FileSystem.documentDirectory ?? "";
  const destPath = `${destDir}eye-detector.onnx`;

  const info = await FileSystem.getInfoAsync(destPath);
  if (!info.exists) {
    console.log("[ONNX] Copying model to:", destPath);
    await FileSystem.downloadAsync(asset.uri, destPath);
  }

  return destPath.replace(/^file:\/\//, "");
}

// ─────────────────────────────────────────────────────────────────────────────
// SESSION MANAGEMENT (singleton)
// ─────────────────────────────────────────────────────────────────────────────
let _session: InferenceSession | null = null;
let _sessionLoading = false;
let _sessionWaiters: Array<(s: InferenceSession) => void> = [];

/**
 * Load the ONNX model. Pass the Metro require() number:
 *   await loadModel(require('../assets/model/eye-detector.onnx'))
 */
export async function loadModel(modelAsset: number): Promise<InferenceSession> {
  if (_session) return _session;

  if (_sessionLoading) {
    return new Promise<InferenceSession>((resolve) => {
      _sessionWaiters.push(resolve);
    });
  }

  _sessionLoading = true;

  try {
    // Step 1: resolve Metro asset number → real on-device file path
    const modelPath = await resolveModelPath(modelAsset);
    console.log("[ONNX] Loading model from:", modelPath);

    // Step 2: create the ONNX inference session with the path string
    const session = await InferenceSession.create(modelPath, {
      executionProviders: ["cpu"], // swap to "nnapi" on Android for speed
      graphOptimizationLevel: "all",
      enableCpuMemArena: true,
    });

    _session = session;
    _sessionWaiters.forEach((fn) => fn(session));
    _sessionWaiters = [];

    console.log("[ONNX] ✅ Model loaded. Inputs:", session.inputNames, "Outputs:", session.outputNames);
    return session;
  } catch (err) {
    _sessionLoading = false;
    throw err;
  }
}

export function disposeModel(): void {
  _session?.release?.();
  _session = null;
  _sessionLoading = false;
  _anchorCache = null;
}

// ─────────────────────────────────────────────────────────────────────────────
// MAIN INFERENCE FUNCTION
//
// Usage:
//   const det = await runEyeDetection(session, rgbPixels320x320);
//   if (det) { /* det.x1, det.y1, det.x2, det.y2, det.score */ }
// ─────────────────────────────────────────────────────────────────────────────
export async function runEyeDetection(
  session: InferenceSession,
  /** Uint8Array of RGB pixels, shape [320, 320, 3] */
  rgbPixelsHWC: Uint8Array,
): Promise<EyeDetection | null> {
  // 1. Preprocess: HWC uint8 → NCHW float32
  const inputData = hwcUint8ToNchwFloat32(
    rgbPixelsHWC,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  );

  // 2. Create input tensor [1, 3, 320, 320]
  const inputTensor = new Tensor("float32", inputData, [
    1,
    3,
    MODEL_INPUT_SIZE,
    MODEL_INPUT_SIZE,
  ]);

  // 3. Run inference
  const feeds: Record<string, Tensor> = { input: inputTensor };
  const results = await session.run(feeds);

  // 4. Validate outputs
  const clsTensor  = results["cls_logits"];
  const bboxTensor = results["bbox_regression"];

  if (!clsTensor || !bboxTensor) {
    console.error(
      "[ONNX] Missing output tensors. Available:",
      Object.keys(results),
    );
    return null;
  }

  const clsData  = clsTensor.data  as Float32Array;
  const bboxData = bboxTensor.data as Float32Array;

  // 5. Decode → best eye bounding box
  return decodeOutputs(clsData, bboxData, IS_DELTA_ENCODED, CONF_THRESHOLD);
}

// ─────────────────────────────────────────────────────────────────────────────
// COORDINATE CONVERSION HELPERS
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Convert normalized [0–1] detection coords to pixel coords on screen.
 * @param det    EyeDetection with normalized coords
 * @param screenW  device screen width in px
 * @param screenH  device screen height in px
 */
export function detectionToScreenPixels(
  det: EyeDetection,
  screenW: number,
  screenH: number,
) {
  return {
    x1:     det.x1 * screenW,
    y1:     det.y1 * screenH,
    x2:     det.x2 * screenW,
    y2:     det.y2 * screenH,
    width:  det.w  * screenW,
    height: det.h  * screenH,
    cx:     det.cx * screenW,
    cy:     det.cy * screenH,
    score:  det.score,
  };
}

/**
 * Check whether a detection falls inside the guide box (with inner margin).
 */
export function isInsideGuideBox(
  det: EyeDetection,
  guideLeft: number,
  guideTop: number,
  guideSize: number,
  screenW: number,
  screenH: number,
  innerMarginFraction = 0.1,
): boolean {
  const px = detectionToScreenPixels(det, screenW, screenH);
  const margin = guideSize * innerMarginFraction;

  return (
    px.x1 >= guideLeft + margin &&
    px.y1 >= guideTop  + margin &&
    px.x2 <= guideLeft + guideSize - margin &&
    px.y2 <= guideTop  + guideSize - margin &&
    px.width  > 0 &&
    px.height > 0
  );
}