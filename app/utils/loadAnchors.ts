/**
 * loadAnchors.ts
 *
 * Loads ssdlite320_anchors.npy and returns the raw pixel-space
 * [x1, y1, x2, y2] anchors exactly as stored — NO conversion.
 *
 * The notebook's reference decode_boxes() (cell 10) works in pixel
 * space and normalises boxes AFTER decoding (÷ 320).  We match that.
 *
 * Do NOT convert anchors to [cx,cy,w,h] or normalise them here —
 * that was the first bug in the earlier versions.
 */

import { Asset } from 'expo-asset';

const NUM_ANCHORS = 3234;
const ANCHOR_DIM  = 4;

// ─── Minimal NPY parser (version 1 and 2) ────────────────────────────────────

function parseNpy(buffer: ArrayBuffer): Float32Array {
  const bytes = new Uint8Array(buffer);

  const MAGIC = [0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59]; // '\x93NUMPY'
  for (let i = 0; i < MAGIC.length; i++) {
    if (bytes[i] !== MAGIC[i]) throw new Error('loadAnchors: not a valid .npy file');
  }

  const major = bytes[6];
  const view  = new DataView(buffer);
  let dataOffset: number;

  if (major === 1) {
    dataOffset = 10 + view.getUint16(8, true);
  } else if (major === 2) {
    dataOffset = 12 + view.getUint32(8, true);
  } else {
    throw new Error(`loadAnchors: unsupported .npy version ${major}`);
  }

  // Sanity: check dtype is float32
  const headerStart = major === 1 ? 10 : 12;
  const headerStr   = String.fromCharCode(...bytes.slice(headerStart, dataOffset));
  if (
    !headerStr.includes("'f4'") &&
    !headerStr.includes('<f4')  &&
    !headerStr.includes('float32')
  ) {
    console.warn('[loadAnchors] dtype may not be float32 – proceeding anyway');
  }

  return new Float32Array(buffer.slice(dataOffset));
}

// ─── Module-level cache ───────────────────────────────────────────────────────

let _anchors: Float32Array | null = null;

// ─── Public API ───────────────────────────────────────────────────────────────

/**
 * Returns raw pixel-space anchors [x1,y1,x2,y2], shape [3234 × 4].
 * Values span roughly –107 to +427 (pixel coordinates for a 320×320 image).
 * Cached after first load.
 */
export async function loadAnchors(): Promise<Float32Array> {
  if (_anchors !== null) return _anchors;

  try {
    const [asset] = await Asset.loadAsync(
      require('../../assets/model/ssdlite320_anchors.npy'),
    );

    if (!asset.localUri) {
      throw new Error('loadAnchors: asset.localUri is null after loadAsync');
    }

    const response = await fetch(asset.localUri);
    if (!response.ok) {
      throw new Error(`loadAnchors: fetch failed – HTTP ${response.status}`);
    }

    const buffer = await response.arrayBuffer();
    const raw    = parseNpy(buffer);

    const expected = NUM_ANCHORS * ANCHOR_DIM;
    if (raw.length !== expected) {
      throw new Error(
        `loadAnchors: expected ${expected} floats, got ${raw.length}`,
      );
    }

    // Sanity log — first anchor should be [-24, -24, 40, 40]
    console.log(
      `[loadAnchors] first=[${raw[0].toFixed(1)},${raw[1].toFixed(1)},` +
      `${raw[2].toFixed(1)},${raw[3].toFixed(1)}]  ` +
      `last=[${raw[raw.length-4].toFixed(1)},${raw[raw.length-3].toFixed(1)},` +
      `${raw[raw.length-2].toFixed(1)},${raw[raw.length-1].toFixed(1)}]`,
    );
    console.log(`[loadAnchors] Loaded ${NUM_ANCHORS} pixel-space anchors ✓`);

    _anchors = raw;
    return _anchors;

  } catch (err) {
    console.error('[loadAnchors] Failed:', err);
    throw err;
  }
}

export function getAnchors(): Float32Array {
  if (!_anchors) throw new Error('loadAnchors: not loaded yet — call loadAnchors() first');
  return _anchors;
}