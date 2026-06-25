/**
 * onnxDetector.ts
 *
 * ONNX inference wrapper for the SSDLite conjunctiva detector.
 *
 * ── Preprocessing (critical) ─────────────────────────────────────────────────
 * The model was trained with PyTorch's ToTensor() ONLY:
 *
 *   transforms.Compose([
 *     transforms.Resize((320, 320)),
 *     transforms.ToTensor(),          ← divides by 255, no normalise()
 *   ])
 *
 * Previous versions applied ImageNet mean/std normalisation which the model
 * was NEVER trained with. This compressed scores into 0.27–0.68 for background
 * and 0.76–0.92 for eye, making them hard to separate.
 *
 * With correct /255-only preprocessing:
 *   background (no eye) → 0.68–0.75
 *   real eye present    → 0.92–0.97
 *   safe threshold      → 0.85
 */

import { Asset } from 'expo-asset';
import * as ort from 'onnxruntime-react-native';
import { loadAnchors } from './loadAnchors';
import { postprocess, Detection } from './ssdDecoder';

// ─── Constants ───────────────────────────────────────────────────────────────

const MODEL_W     = 320;
const MODEL_H     = 320;
const NUM_ANCHORS = 3234;

// ─── Module-level state ───────────────────────────────────────────────────────

let _session: ort.InferenceSession | null = null;
let _anchors: Float32Array | null         = null;

export let detectorReady = false;

// Pre-allocated input buffer — avoids GC pressure on every frame
const _inputBuffer = new Float32Array(3 * MODEL_H * MODEL_W);

// ─── Types ────────────────────────────────────────────────────────────────────

export interface DetectionResult {
  detected:      boolean;
  topScore:      number;
  topBox:        { x1: number; y1: number; x2: number; y2: number } | null;
  allDetections: Detection[];
}

const EMPTY_RESULT: DetectionResult = {
  detected: false, topScore: 0, topBox: null, allDetections: [],
};

// ─── Initialisation ───────────────────────────────────────────────────────────

export async function initDetector(): Promise<void> {
  if (detectorReady) return;

  try {
    const [asset] = await Asset.loadAsync(
      require('../../assets/model/mobilenet.onnx'),
    );

    if (!asset.localUri) throw new Error('initDetector: model localUri is null');

    console.log('[ONNX] Loading model from', asset.localUri);

    _session = await ort.InferenceSession.create(asset.localUri, {
      executionProviders:     ['cpu'],
      graphOptimizationLevel: 'all',
      enableCpuMemArena:      true,
    });

    console.log(
      '[ONNX] Session created ✓  inputs:', _session.inputNames,
      'outputs:', _session.outputNames,
    );

    _anchors = await loadAnchors();

    detectorReady = true;
    console.log('[ONNX] Detector ready ✓');

  } catch (err) {
    console.error('[ONNX] initDetector failed:', err);
    detectorReady = false;
    throw err;
  }
}

// ─── Preprocessing ────────────────────────────────────────────────────────────

/**
 * Convert RGBA Uint8Array → CHW float32 tensor, dividing by 255 only.
 *
 * Layout expected by model: [1, 3, 320, 320] (R plane, G plane, B plane).
 * Alpha channel is discarded.
 */
function preprocessRGBA(rgba: Uint8Array): Float32Array {
  const N = MODEL_W * MODEL_H;
  for (let i = 0; i < N; i++) {
    const p = i * 4;
    _inputBuffer[i]         = rgba[p]     / 255.0;   // R
    _inputBuffer[N + i]     = rgba[p + 1] / 255.0;   // G
    _inputBuffer[N * 2 + i] = rgba[p + 2] / 255.0;   // B
    // rgba[p + 3] = alpha — ignored
  }
  return _inputBuffer;
}

// ─── Inference ────────────────────────────────────────────────────────────────

/**
 * Run one frame through the detector.
 *
 * @param rgbaPixels  320×320 RGBA pixel buffer (length = 320*320*4 = 409,600)
 * @param confThresh  Minimum confidence to accept a detection (default 0.85)
 * @param iouThresh   NMS IoU threshold (default 0.45)
 */
export async function detectConjunctiva(
  rgbaPixels: Uint8Array,
  confThresh  = 0.85,
  iouThresh   = 0.45,
): Promise<DetectionResult> {

  if (!_session || !_anchors || !detectorReady) return EMPTY_RESULT;

  try {
    const t0 = performance.now();

    const inputData   = preprocessRGBA(rgbaPixels);
    const inputTensor = new ort.Tensor('float32', inputData, [1, 3, MODEL_H, MODEL_W]);

    const t1      = performance.now();
    const results = await _session.run({ input: inputTensor });
    const t2      = performance.now();

    const clsData  = results['cls_logits']?.data     as Float32Array | undefined;
    const bboxData = results['bbox_regression']?.data as Float32Array | undefined;

    if (!clsData || !bboxData) {
      console.warn('[ONNX] Missing output tensors — check model output names');
      return EMPTY_RESULT;
    }

    const dets = postprocess(clsData, bboxData, _anchors, NUM_ANCHORS, confThresh, iouThresh);
    const t3   = performance.now();

    console.log(
      `[ONNX] pre=${(t1-t0).toFixed(1)}ms  inf=${(t2-t1).toFixed(1)}ms  ` +
      `post=${(t3-t2).toFixed(1)}ms  total=${(t3-t0).toFixed(1)}ms  dets=${dets.length}`,
    );

    if (dets.length === 0) return EMPTY_RESULT;

    const top = dets.reduce((a, b) => (a.score > b.score ? a : b));

    console.log(
      `[FINAL DETECTION] score=${top.score.toFixed(4)} ` +
      `box=[${top.x1.toFixed(3)},${top.y1.toFixed(3)},${top.x2.toFixed(3)},${top.y2.toFixed(3)}]`,
    );

    return {
      detected:      true,
      topScore:      top.score,
      topBox:        { x1: top.x1, y1: top.y1, x2: top.x2, y2: top.y2 },
      allDetections: dets,
    };

  } catch (err) {
    console.error('[ONNX] detectConjunctiva error:', err);
    return EMPTY_RESULT;
  }
}