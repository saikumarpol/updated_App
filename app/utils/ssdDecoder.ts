/**
 * ssdDecoder.ts
 *
 * SSD bounding-box decoder + NMS.
 *
 * Matches the notebook's reference decode_boxes() (cell 10) exactly:
 *
 *   anchors  → raw pixel [x1, y1, x2, y2]   (range ≈ –107 to +427)
 *   decode   → pixel-space predicted box
 *   normalise→ divide by MODEL_SIZE (320) → [0, 1]
 *
 * Previous versions wrongly pre-converted anchors to normalised
 * [cx,cy,w,h] before decoding, which is mathematically equivalent
 * but introduced a transcription error that caused the regression.
 * Keeping anchors in pixel space and matching the notebook line-for-line
 * eliminates any chance of format mismatch.
 */

export interface Detection {
  x1:      number;   // normalised [0, 1]
  y1:      number;
  x2:      number;
  y2:      number;
  score:   number;
  classId: number;
}

// TorchVision SSD variance weights
const WX = 10.0;
const WY = 10.0;
const WW = 5.0;
const WH = 5.0;

const MODEL_SIZE = 320.0;   // image side length used to normalise pixel boxes

// ─── Box decoder ─────────────────────────────────────────────────────────────

/**
 * Decode SSD regression offsets into normalised [x1,y1,x2,y2] boxes.
 *
 * @param regression  Float32Array length N×4  — [tx, ty, tw, th] per anchor
 * @param anchors     Float32Array length N×4  — PIXEL [x1, y1, x2, y2]
 * @param N           Number of anchors (3234)
 * @returns           Float32Array length N×4  — normalised [x1,y1,x2,y2] ∈ [0,1]
 */
export function decodeBoxes(
  regression: Float32Array,
  anchors:    Float32Array,
  N:          number,
): Float32Array {
  const decoded = new Float32Array(N * 4);

  for (let i = 0; i < N; i++) {
    const ri = i * 4;

    // ── Anchor geometry in pixel space (mirrors notebook exactly) ─────────
    const x1a = anchors[ri];
    const y1a = anchors[ri + 1];
    const x2a = anchors[ri + 2];
    const y2a = anchors[ri + 3];

    const W   = x2a - x1a;           // pixel anchor width
    const H   = y2a - y1a;           // pixel anchor height
    const cxA = x1a + 0.5 * W;       // pixel anchor centre-x
    const cyA = y1a + 0.5 * H;       // pixel anchor centre-y

    // ── Variance-decoded offsets ──────────────────────────────────────────
    const dx = regression[ri]     / WX;
    const dy = regression[ri + 1] / WY;
    const dw = Math.min(regression[ri + 2] / WW, 4.135);   // cap exp input
    const dh = Math.min(regression[ri + 3] / WH, 4.135);

    // ── Predicted box in pixel space ──────────────────────────────────────
    const pcx = dx * W + cxA;
    const pcy = dy * H + cyA;
    const pw  = Math.exp(dw) * W;
    const ph  = Math.exp(dh) * H;

    // ── Convert to corners, then normalise to [0, 1] ──────────────────────
    decoded[ri]     = (pcx - 0.5 * pw) / MODEL_SIZE;
    decoded[ri + 1] = (pcy - 0.5 * ph) / MODEL_SIZE;
    decoded[ri + 2] = (pcx + 0.5 * pw) / MODEL_SIZE;
    decoded[ri + 3] = (pcy + 0.5 * ph) / MODEL_SIZE;
  }

  return decoded;
}

// ─── IoU ─────────────────────────────────────────────────────────────────────

function iou(boxes: Float32Array, a: number, b: number): number {
  const ai = a * 4;
  const bi = b * 4;

  const ix1 = Math.max(boxes[ai],     boxes[bi]);
  const iy1 = Math.max(boxes[ai + 1], boxes[bi + 1]);
  const ix2 = Math.min(boxes[ai + 2], boxes[bi + 2]);
  const iy2 = Math.min(boxes[ai + 3], boxes[bi + 3]);

  const inter = Math.max(0, ix2 - ix1) * Math.max(0, iy2 - iy1);
  const area1 = (boxes[ai + 2] - boxes[ai])     * (boxes[ai + 3] - boxes[ai + 1]);
  const area2 = (boxes[bi + 2] - boxes[bi])     * (boxes[bi + 3] - boxes[bi + 1]);

  return inter / (area1 + area2 - inter + 1e-6);
}

// ─── NMS ─────────────────────────────────────────────────────────────────────

function nms(boxes: Float32Array, scores: number[], iouThreshold: number): number[] {
  const order = scores
    .map((s, i) => ({ s, i }))
    .sort((a, b) => b.s - a.s)
    .map(o => o.i);

  const keep:       number[]    = [];
  const suppressed: Uint8Array  = new Uint8Array(scores.length);

  for (let k = 0; k < order.length; k++) {
    const idx = order[k];
    if (suppressed[idx]) continue;
    keep.push(idx);
    for (let m = k + 1; m < order.length; m++) {
      const other = order[m];
      if (!suppressed[other] && iou(boxes, idx, other) >= iouThreshold) {
        suppressed[other] = 1;
      }
    }
  }

  return keep;
}

// ─── Full postprocessing pipeline ────────────────────────────────────────────

export function postprocess(
  clsLogits:  Float32Array,
  bboxReg:    Float32Array,
  anchors:    Float32Array,
  N:          number,
  confThresh: number = 0.85,
  iouThresh:  number = 0.45,
): Detection[] {

  // 1. Decode all boxes
  const decoded = decodeBoxes(bboxReg, anchors, N);

  // 2. Score filtering via softmax on class logits
  const candIdx:    number[] = [];
  const candScores: number[] = [];
  let maxScore = 0;
  let maxIdx   = -1;

  for (let i = 0; i < N; i++) {
    const l0 = clsLogits[i * 2];
    const l1 = clsLogits[i * 2 + 1];
    const m  = Math.max(l0, l1);
    const e0 = Math.exp(l0 - m);
    const e1 = Math.exp(l1 - m);
    const sc = e1 / (e0 + e1);   // P(conjunctiva)

    if (sc > maxScore) { maxScore = sc; maxIdx = i; }
    if (sc >= confThresh) { candIdx.push(i); candScores.push(sc); }
  }

  // Always log the best anchor so threshold can be tuned from logs
  if (__DEV__ && maxIdx >= 0) {
    const bi = maxIdx * 4;
    console.log(
      `[SCORE FILTER] best=#${maxIdx} score=${maxScore.toFixed(4)} ` +
      `box=[${decoded[bi].toFixed(3)},${decoded[bi+1].toFixed(3)},` +
      `${decoded[bi+2].toFixed(3)},${decoded[bi+3].toFixed(3)}] ` +
      `candidates=${candIdx.length}`,
    );
  }

  if (candIdx.length === 0) return [];

  // 3. Build candidate buffer and clamp to [0, 1]
  const candBoxes = new Float32Array(candIdx.length * 4);
  for (let k = 0; k < candIdx.length; k++) {
    const src = candIdx[k] * 4;
    const dst = k * 4;
    candBoxes[dst]     = Math.max(0, Math.min(1, decoded[src]));
    candBoxes[dst + 1] = Math.max(0, Math.min(1, decoded[src + 1]));
    candBoxes[dst + 2] = Math.max(0, Math.min(1, decoded[src + 2]));
    candBoxes[dst + 3] = Math.max(0, Math.min(1, decoded[src + 3]));
  }

  // 4. NMS
  const kept = nms(candBoxes, candScores, iouThresh);

  return kept.map(k => ({
    x1:      candBoxes[k * 4],
    y1:      candBoxes[k * 4 + 1],
    x2:      candBoxes[k * 4 + 2],
    y2:      candBoxes[k * 4 + 3],
    score:   candScores[k],
    classId: 1,
  }));
}