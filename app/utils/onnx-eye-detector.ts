/**
 * useOnnxModel.ts
 *
 * A React hook that loads an ONNX model from a local asset (require())
 * using onnxruntime-react-native and exposes a worklet-callable runSync.
 *
 * Usage:
 *   const { model, state } = useOnnxModel(
 *     require("../assets/model/2026-06-19__mnv3ssd-320-best-model.onnx")
 *   );
 *
 * `state`  – "loading" | "ready" | "error"
 * `model`  – null until ready, then an object with:
 *              runSync(inputs: Record<string, Float32Array>) → Record<string, Float32Array>
 */

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-react-native";

// ─── Types ────────────────────────────────────────────────────────────────────

export type OnnxModelState = "loading" | "ready" | "error";

export interface OnnxModel {
  /**
   * Synchronous inference.
   *
   * Pass a map of input-name → Float32Array (already shaped correctly).
   * Returns a map of output-name → Float32Array, or null on failure.
   *
   * Marked as a worklet so it can be called from a VisionCamera
   * useFrameProcessor without hopping to the JS thread.
   */
  runSync: (inputs: Record<string, Float32Array>) => Record<string, Float32Array> | null;
}

// ─── Hook ─────────────────────────────────────────────────────────────────────

export function useOnnxModel(asset: number | string): {
  model: OnnxModel | null;
  state: OnnxModelState;
} {
  const [state, setState] = useState<OnnxModelState>("loading");
  const sessionRef = useRef<ort.InferenceSession | null>(null);

  // Stable model wrapper so referential equality is preserved
  const modelRef = useRef<OnnxModel | null>(null);

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        setState("loading");

        // ort.InferenceSession.create accepts:
        //   - a file URI string  (file:///…)
        //   - a require() number (metro bundled asset)
        const session = await ort.InferenceSession.create(asset as any);

        if (cancelled) {
          await session.release();
          return;
        }

        sessionRef.current = session;

        // Build the model wrapper once and cache it
        modelRef.current = {
          runSync(inputs: Record<string, Float32Array>): Record<string, Float32Array> | null {
            "worklet";
            const sess = sessionRef.current;
            if (!sess) return null;

            try {
              // Build ORT tensors from the raw Float32Arrays.
              // We infer the shape from the session's input metadata when possible,
              // but fall back to a flat [length] shape if metadata is unavailable.
              const feeds: Record<string, ort.Tensor> = {};

              for (const [name, data] of Object.entries(inputs)) {
                // Try to get the expected shape from the model's input metadata
                const inputMeta = sess.inputNames.includes(name)
                  ? (sess as any).inputMetadata?.[name]
                  : undefined;

                const dims: number[] =
                  inputMeta?.dims ??
                  inferDims(name, data.length);

                feeds[name] = new ort.Tensor("float32", data, dims);
              }

              // runSync is available on onnxruntime-react-native >= 1.14
              const results = (sess as any).runSync(feeds) as Record<string, ort.Tensor>;

              // Unwrap tensors → plain Float32Arrays
              const output: Record<string, Float32Array> = {};
              for (const [k, tensor] of Object.entries(results)) {
                output[k] = tensor.data as Float32Array;
              }
              return output;
            } catch (e) {
              // Do NOT throw from a worklet; return null instead
              console.error("[OnnxModel] runSync error:", e);
              return null;
            }
          },
        };

        setState("ready");
      } catch (e) {
        if (!cancelled) {
          console.error("[useOnnxModel] Failed to load model:", e);
          setState("error");
        }
      }
    }

    load();

    return () => {
      cancelled = true;
      // Release the session when the component unmounts
      sessionRef.current?.release().catch(() => {});
      sessionRef.current = null;
      modelRef.current = null;
    };
  }, [asset]);

  return {
    model: state === "ready" ? modelRef.current : null,
    state,
  };
}

// ─── Helpers ──────────────────────────────────────────────────────────────────

/**
 * Guess the ONNX input tensor shape from its name and element count.
 *
 * For the MobileNetV3-SSD model used here:
 *   input  → [1, 3, 320, 320]   (NCHW image)
 *
 * Any unknown input falls back to a 1-D shape [length].
 */
function inferDims(name: string, length: number): number[] {
  if (name === "input") {
    // 1 × 3 × 320 × 320 = 307200
    if (length === 1 * 3 * 320 * 320) return [1, 3, 320, 320];
    // Fallback: try to detect other square NCHW sizes
    const sqrt = Math.round(Math.sqrt(length / 3));
    if (3 * sqrt * sqrt === length) return [1, 3, sqrt, sqrt];
  }
  return [length];
}