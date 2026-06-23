// app/eye-capture.tsx
//
// ─── Model Change Notes ────────────────────────────────────────────────────────
//  Previous model : YOLO (TFLite)
//    • Input  : float32[1, 3, 320, 320]  (via react-native-fast-tflite)
//    • Output : float32[300, 6]  → flat array, stride = VALS_PER_BOX (6)
//               [x1, y1, x2, y2, conf, class_id]
//
//  Current model  : MobileNet-SSD (ONNX, mnv3ssd-320)
//    • Input  : float32[1, 3, 320, 320]  (CHW, RGB, normalised 0-1)
//    • Outputs:
//        cls_logits     → float32[1, 3234, 2]   (raw logits per anchor: [bg, eye])
//        bbox_regression → float32[1, 3234, 4]  (Δ offsets relative to SSD anchors)
//    • Post-processing: sigmoid(cls_logits[:,1]) → eye confidence
//                       decode bbox_regression with SSD anchor boxes → absolute coords
//
//  Library change: react-native-fast-tflite → onnxruntime-react-native
//    • InferenceSession.create() loads the .onnx asset
//    • session.run() returns named tensors matching output names above
// ──────────────────────────────────────────────────────────────────────────────

import React, { useRef, useState, useCallback, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
} from "react-native";
import { useLocalSearchParams, useRouter } from "expo-router";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from "react-native-vision-camera";
import { InferenceSession, Tensor } from "onnxruntime-react-native";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { Worklets } from "react-native-worklets-core";
import * as ImageManipulator from "expo-image-manipulator";
import { Asset } from "expo-asset";

// ─── Constants ─────────────────────────────────────────────────────────────────
const TAG = "[EyeCapture]";

const MODEL_SIZE = 320;          // spatial input size (px)
const NUM_ANCHORS = 3234;        // total SSD anchors for 320×320 MobileNetV3-SSD
const CONF_THRESHOLD = 0.40;     // minimum sigmoid(cls_logit[eye]) to count as detection
const FRAMES_NEEDED = 4;         // consecutive in-guide frames before capture
const OUTPUT_SIZE = 512;         // final cropped image size (px)

// SSD anchor variance factors (standard SSD defaults — adjust if your training differs)
const VARIANCE_XY = 0.1;
const VARIANCE_WH = 0.2;

const { width: SW, height: SH } = Dimensions.get("window");
const GUIDE_SIZE = 220;
const GUIDE_LEFT = (SW - GUIDE_SIZE) / 2;
const GUIDE_TOP = SH * 0.35;
const GUIDE_RIGHT = GUIDE_LEFT + GUIDE_SIZE;
const GUIDE_BOTTOM = GUIDE_TOP + GUIDE_SIZE;
const GUIDE_INNER_MARGIN = GUIDE_SIZE * 0.1;

// ─── SSD Anchor Generation ────────────────────────────────────────────────────
// Generates the same anchor set that MobileNetV3-SSD 320 uses at training time.
// If your custom model used a different anchor config, replace this with the
// exact anchors from your training pipeline (e.g. exported as a JSON file).
function generateSSDAnchors(): Float32Array {
  // ── Anchor config for MobileNetV3-SSD 320 producing exactly 3234 anchors ──
  //
  // Feature maps : 20×20, 10×10, 5×5, 3×3, 2×2, 1×1
  // Anchors/cell : 6 per cell at every layer
  //   Each cell gets:
  //     1. Primary square  (minScale × minScale)
  //     2. Interpolated    (√(minScale·maxScale) × same)
  //     3. AR=2 landscape  (minScale·√2  × minScale/√2)
  //     4. AR=2 portrait   (minScale/√2  × minScale·√2)
  //     5. AR=3 landscape  (minScale·√3  × minScale/√3)
  //     6. AR=3 portrait   (minScale/√3  × minScale·√3)
  //
  // Total: (400+100+25+9+4+1) × 6 = 539 × 6 = 3234 ✅
  const specs = [
    { featureMapSize: 20, minScale: 0.1,   maxScale: 0.2,   aspectRatios: [2, 3] },
    { featureMapSize: 10, minScale: 0.2,   maxScale: 0.375, aspectRatios: [2, 3] },
    { featureMapSize: 5,  minScale: 0.375, maxScale: 0.55,  aspectRatios: [2, 3] },
    { featureMapSize: 3,  minScale: 0.55,  maxScale: 0.725, aspectRatios: [2, 3] },
    { featureMapSize: 2,  minScale: 0.725, maxScale: 0.9,   aspectRatios: [2, 3] },
    { featureMapSize: 1,  minScale: 0.9,   maxScale: 1.075, aspectRatios: [2, 3] },
  ];

  const anchors: number[] = [];

  for (const spec of specs) {
    const { featureMapSize: fms, minScale, maxScale, aspectRatios } = spec;
    const step = 1.0 / fms;

    for (let row = 0; row < fms; row++) {
      for (let col = 0; col < fms; col++) {
        const cx = (col + 0.5) * step;
        const cy = (row + 0.5) * step;

        // 1 & 2: primary square + interpolated square
        anchors.push(cy, cx, minScale, minScale);
        const interpScale = Math.sqrt(minScale * maxScale);
        anchors.push(cy, cx, interpScale, interpScale);

        // 3-6: landscape + portrait for each aspect ratio
        for (const ar of aspectRatios) {
          const sqrtAr = Math.sqrt(ar);
          anchors.push(cy, cx, minScale * sqrtAr, minScale / sqrtAr);  // landscape
          anchors.push(cy, cx, minScale / sqrtAr, minScale * sqrtAr);  // portrait
        }
      }
    }
  }

  const generated = anchors.length / 4;
  if (generated !== NUM_ANCHORS) {
    console.error(`${TAG} ❌ Anchor count mismatch: generated ${generated}, expected ${NUM_ANCHORS}`);
  } else {
    console.log(`${TAG} ✅ Generated ${generated} anchors correctly`);
  }

  const out = new Float32Array(NUM_ANCHORS * 4);
  for (let i = 0; i < anchors.length; i++) out[i] = anchors[i];
  return out;
}

// ─── Decode one SSD box ───────────────────────────────────────────────────────
// Inputs  : raw bbox_regression deltas + anchor [cy, cx, h, w] (normalised 0-1)
// Returns : { x1, y1, x2, y2 } normalised 0-1
function decodeSSDBox(
  dy: number, dx: number, dh: number, dw: number,
  acy: number, acx: number, ah: number, aw: number,
): { x1: number; y1: number; x2: number; y2: number } {
  const cy = dy * VARIANCE_XY * ah + acy;
  const cx = dx * VARIANCE_XY * aw + acx;
  const h  = Math.exp(dh * VARIANCE_WH) * ah;
  const w  = Math.exp(dw * VARIANCE_WH) * aw;

  return {
    x1: cx - w / 2,
    y1: cy - h / 2,
    x2: cx + w / 2,
    y2: cy + h / 2,
  };
}

// Pre-compute anchors once at module load
const ANCHORS = generateSSDAnchors();

// ─── Types ────────────────────────────────────────────────────────────────────
type DebugEyeBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  score: number;
  insideGuide: boolean;
};

// ─── Coordinate helpers ───────────────────────────────────────────────────────
//
// The model receives a 320×320 square crop. We crop the camera frame to the
// GUIDE_SIZE×GUIDE_SIZE region before feeding the model (via the resize step).
// Therefore model coordinates [0,1] map to the GUIDE BOX on screen, NOT to
// the full screen dimensions. Mapping to SW/SH was wrong and caused the huge
// bounding box seen in testing.
//
// Correct mapping: model [0,1] → [GUIDE_LEFT…GUIDE_RIGHT, GUIDE_TOP…GUIDE_BOTTOM]
//
function modelToScreen(nx: number, ny: number) {
  return {
    sx: GUIDE_LEFT + nx * GUIDE_SIZE,
    sy: GUIDE_TOP  + ny * GUIDE_SIZE,
  };
}

function getEyeBoxOnScreen(
  x1: number, y1: number, x2: number, y2: number,
) {
  const { sx: l1, sy: t1 } = modelToScreen(Math.min(x1, x2), Math.min(y1, y2));
  const { sx: l2, sy: t2 } = modelToScreen(Math.max(x1, x2), Math.max(y1, y2));
  return {
    left:   l1,
    top:    t1,
    right:  l2,
    bottom: t2,
    width:  l2 - l1,
    height: t2 - t1,
  };
}

function checkEyeWithinGuideBox(
  x1: number, y1: number, x2: number, y2: number,
) {
  const box = getEyeBoxOnScreen(x1, y1, x2, y2);
  // The eye box must sit inside the inner margin of the guide box
  const innerLeft   = GUIDE_LEFT   + GUIDE_INNER_MARGIN;
  const innerTop    = GUIDE_TOP    + GUIDE_INNER_MARGIN;
  const innerRight  = GUIDE_RIGHT  - GUIDE_INNER_MARGIN;
  const innerBottom = GUIDE_BOTTOM - GUIDE_INNER_MARGIN;

  return (
    box.width  > 0 &&
    box.height > 0 &&
    box.left   >= innerLeft  &&
    box.top    >= innerTop   &&
    box.right  <= innerRight &&
    box.bottom <= innerBottom
  );
}

// ─── Sigmoid (JS side, used after tensor read-back) ───────────────────────────
function sigmoid(x: number): number {
  return 1 / (1 + Math.exp(-x));
}

// ─── Component ────────────────────────────────────────────────────────────────
export default function EyeCaptureScreen() {
  const router = useRouter();
  const params = useLocalSearchParams<{
    eyeSide?: "left" | "right";
    name?: string;
    parentName?: string;
    phoneNumber?: string;
    age?: string;
    gender?: string;
    eyeSessionId?: string;
  }>();

  const { eyeSide = "right" } = params;

  const cameraRef = useRef<Camera>(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("front");
  const { resize } = useResizePlugin();

  // ── ONNX session (loaded once) ──────────────────────────────────────────────
  const sessionRef = useRef<InferenceSession | null>(null);
  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");

  // ── Heartbeat: confirms JS thread stays alive after model load ──────────────
  useEffect(() => {
    let count = 0;
    const id = setInterval(() => {
      count += 1;
      console.log(
        `${TAG} ❤️  heartbeat #${count} | modelReady=${modelReadyRef.current} | ` +
        `inFlight=${inferenceInFlightRef.current} | capturing=${capturingRef.current}`,
      );
      if (count >= 10) clearInterval(id); // stop after 10s
    }, 1000);
    return () => clearInterval(id);
  }, []);

  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        console.log(`${TAG} Loading ONNX model…`);

        // ─── WHY expo-asset? ───────────────────────────────────────────────
        // onnxruntime-react-native accepts only a plain file-system PATH string
        // or an ArrayBuffer — NOT the numeric module ID that Metro's require()
        // returns. expo-asset.loadAsync() downloads (or copies from the bundle)
        // the file and exposes its real on-device path via `localUri`.
        const [asset] = await Asset.loadAsync(
          require("../assets/model/mobilenet.onnx"),
        );

        if (!asset.localUri) {
          throw new Error(
            "Asset.localUri is null after loadAsync – verify the .onnx file is " +
            "listed under 'assets' in app.json and that metro.config.js includes " +
            "the .onnx extension in assetExts.",
          );
        }

        // ONNX Runtime expects a plain path, not a file:// URI
        const modelPath = asset.localUri.replace(/^file:\/\//, "");
        console.log(`${TAG} Asset resolved → ${modelPath}`);

        const session = await InferenceSession.create(modelPath, {
          // CoreML runs on the Neural Engine on iOS (much faster than CPU).
          // ONNX Runtime falls back to CPU automatically if CoreML is unavailable.
          executionProviders: ["coreml", "cpu"],
        });
        if (!mounted) return;
        sessionRef.current = session;
        modelReadyRef.current = true;
        setModelState("ready");
        console.log(`${TAG} ✅ ONNX model loaded. Inputs:`, session.inputNames, "Outputs:", session.outputNames);
      } catch (err: any) {
        console.error(`${TAG} ❌ Model load failed:`, err.message ?? err);
        if (mounted) setModelState("error");
      }
    })();
    return () => { mounted = false; };
  }, []);

  // ── React state ─────────────────────────────────────────────────────────────
  const [detected,   setDetected]   = useState(false);
  const [score,      setScore]      = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [statusMsg,  setStatusMsg]  = useState("Position eye in the box");
  const [capturing,  setCapturing]  = useState(false);
  const [debugEyeBox, setDebugEyeBox] = useState<DebugEyeBox | null>(null);

  const cameraReadyRef   = useRef(false);
  const capturingRef     = useRef(false);
  const consecutiveRef   = useRef(0);
  // Frame-skip counter so we don't flood slow devices
  const frameSkipRef        = useRef(0);
  const FRAME_SKIP_EVERY    = 2;   // run inference every N frames
  // Prevent queuing multiple concurrent ONNX calls; drop frames while busy
  const inferenceInFlightRef = useRef(false);
  // Mirrors modelState for the worklet (refs are readable from worklets, state is not)
  const modelReadyRef = useRef(false);

  // Worklet → JS bridges
  const setDetectedJS    = Worklets.createRunOnJS(setDetected);
  const setScoreJS       = Worklets.createRunOnJS(setScore);
  const setFrameCountJS  = Worklets.createRunOnJS(setFrameCount);
  const setStatusMsgJS   = Worklets.createRunOnJS(setStatusMsg);
  const setDebugEyeBoxJS = Worklets.createRunOnJS(setDebugEyeBox);

  // ── Capture & Navigate ──────────────────────────────────────────────────────
  const captureAndNavigate = useCallback(async () => {
    if (capturingRef.current) return;

    const camera = cameraRef.current;
    if (!cameraReadyRef.current || !camera) {
      consecutiveRef.current = 0;
      setDetected(false);
      setFrameCount(0);
      setStatusMsg("Preparing camera…");
      console.warn(`${TAG} captureAndNavigate called but camera not ready`);
      return;
    }

    capturingRef.current = true;
    setCapturing(true);
    setStatusMsg("Capturing high-quality image…");
    console.log(`${TAG} 📸 Starting capture for eye: ${eyeSide}`);

    try {
      // 1. Snapshot
      const photo = await camera.takeSnapshot({ quality: 95 });
      const photoUri = photo.path.startsWith("file://")
        ? photo.path
        : `file://${photo.path}`;
      console.log(`${TAG} Snapshot taken – size: ${photo.width}×${photo.height}, uri: ${photoUri}`);

      // 2. Calculate crop region (guide box mapped to photo coords)
      const scaleX = photo.width  / SW;
      const scaleY = photo.height / SH;
      const cropX = Math.max(0, Math.round(GUIDE_LEFT * scaleX));
      const cropY = Math.max(0, Math.round(GUIDE_TOP  * scaleY));
      const cropW = Math.min(Math.round(GUIDE_SIZE * scaleX), photo.width  - cropX);
      const cropH = Math.min(Math.round(GUIDE_SIZE * scaleY), photo.height - cropY);
      console.log(`${TAG} Crop → x:${cropX} y:${cropY} w:${cropW} h:${cropH}`);

      // 3. Crop + resize to OUTPUT_SIZE × OUTPUT_SIZE
      const manipulated = await ImageManipulator.manipulateAsync(
        photoUri,
        [
          { crop: { originX: cropX, originY: cropY, width: cropW, height: cropH } },
          { resize: { width: OUTPUT_SIZE, height: OUTPUT_SIZE } },
        ],
        { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
      );
      console.log(`${TAG} ✅ Cropped image ready: ${manipulated.uri}`);

      // 4. Navigate back with the captured image
      if (!eyeSide) {
        console.error(`${TAG} ❌ eyeSide missing – aborting navigation`);
        router.back();
        return;
      }

      const imageKey = eyeSide === "left" ? "leftEyeImage" : "rightEyeImage";
      router.replace({
        pathname: "/",
        params: {
          [imageKey]: manipulated.uri,
          name:        params.name,
          parentName:  params.parentName,
          phoneNumber: params.phoneNumber,
          age:         params.age,
          gender:      params.gender,
          eyeSessionId: params.eyeSessionId,
        },
      });
    } catch (err: any) {
      console.error(`${TAG} ❌ CAPTURE FAILED:`, err.message ?? err);
      setStatusMsg("Capture failed – try again");
    } finally {
      setCapturing(false);
      capturingRef.current = false;
    }
  }, [router, eyeSide, params]);

  const triggerCaptureJS = Worklets.createRunOnJS(captureAndNavigate);

  // ── Inference helper (runs on JS thread, called from worklet bridge) ─────────
  // This runs the actual ONNX inference and returns best detection.
  // We keep it separate so the worklet only hands off pixel data.
  const runInference = useCallback(async (
    chwArray: number[],  // CHW float32 values pre-transposed in the worklet
  ): Promise<{ bestScore: number; bestX1: number; bestY1: number; bestX2: number; bestY2: number } | null> => {
    const session = sessionRef.current;
    if (!session) {
      console.warn(`${TAG} runInference called but session not ready`);
      return null;
    }

    try {
      // ─── Wrap plain number[] into Float32Array for ONNX Runtime ──────────
      // Worklets.createRunOnJS() can only transfer primitives and plain
      // objects/arrays — no TypedArrays, no ArrayBuffers. We therefore do
      // the HWC→CHW transpose inside the worklet and send a plain number[].
      // Here we just wrap it cheaply into a Float32Array (no copy needed
      // because new Float32Array(number[]) allocates a fresh typed buffer).
      const EXPECTED = 3 * MODEL_SIZE * MODEL_SIZE;

      console.log(`${TAG} 📊 runInference — elements: ${chwArray.length} (expected ${EXPECTED})`);

      if (chwArray.length !== EXPECTED) {
        console.error(
          `${TAG} ❌ Size mismatch: got ${chwArray.length}, expected ${EXPECTED}`,
        );
        return null;
      }

      const chw = new Float32Array(chwArray);

      // Build input tensor  [1, 3, 320, 320]
      const inputTensor = new Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]);
      const feeds: Record<string, Tensor> = { input: inputTensor };

      const results = await session.run(feeds);

      // ── Unpack outputs ──────────────────────────────────────────────────────
      // cls_logits     : [1, 3234, 2]  – raw logits [background, eye]
      // bbox_regression: [1, 3234, 4]  – SSD offsets [dy, dx, dh, dw]
      const clsData  = results["cls_logits"]?.data    as Float32Array | undefined;
      const bboxData = results["bbox_regression"]?.data as Float32Array | undefined;

      if (!clsData || !bboxData) {
        console.error(`${TAG} ❌ Missing output tensors. Got:`, Object.keys(results));
        return null;
      }

      // ── Find best detection ────────────────────────────────────────────────
      let bestScore = 0;
      let bestX1 = 0, bestY1 = 0, bestX2 = 0, bestY2 = 0;

      for (let i = 0; i < NUM_ANCHORS; i++) {
        // cls_logits[i] = [logit_bg, logit_eye]  → index i*2+1 is eye class
        const eyeLogit = clsData[i * 2 + 1];
        const conf = sigmoid(eyeLogit);

        if (conf > bestScore) {
          // bbox_regression[i] = [dy, dx, dh, dw]
          const dy = bboxData[i * 4 + 0];
          const dx = bboxData[i * 4 + 1];
          const dh = bboxData[i * 4 + 2];
          const dw = bboxData[i * 4 + 3];

          // Anchor for this slot [cy, cx, h, w] stored as [cy, cx, h, w]
          const acy = ANCHORS[i * 4 + 0];
          const acx = ANCHORS[i * 4 + 1];
          const ah  = ANCHORS[i * 4 + 2];
          const aw  = ANCHORS[i * 4 + 3];

          const box = decodeSSDBox(dy, dx, dh, dw, acy, acx, ah, aw);

          bestScore = conf;
          bestX1 = box.x1;
          bestY1 = box.y1;
          bestX2 = box.x2;
          bestY2 = box.y2;
        }
      }

      console.log(
        `${TAG} 🎯 Best detection — score=${bestScore.toFixed(3)} ` +
        `box=[${bestX1.toFixed(3)},${bestY1.toFixed(3)},${bestX2.toFixed(3)},${bestY2.toFixed(3)}]`
      );
      return { bestScore, bestX1, bestY1, bestX2, bestY2 };
    } catch (err: any) {
      console.error(`${TAG} ❌ Inference error:`, err.message ?? err);
      return null;
    }
  }, []);

  // JS-thread bridge called from the worklet
  const handleInferenceResult = useCallback((
    bestScore: number,
    bestX1: number, bestY1: number,
    bestX2: number, bestY2: number,
  ) => {
    setScoreJS(bestScore);

    const eyeInsideGuide =
      bestScore >= CONF_THRESHOLD &&
      checkEyeWithinGuideBox(bestX1, bestY1, bestX2, bestY2);

    if (bestScore >= CONF_THRESHOLD) {
      const box = getEyeBoxOnScreen(bestX1, bestY1, bestX2, bestY2);
      setDebugEyeBoxJS({
        left: box.left, top: box.top,
        width: box.width, height: box.height,
        score: bestScore,
        insideGuide: eyeInsideGuide,
      });
    } else {
      setDebugEyeBoxJS(null);
    }

    if (eyeInsideGuide) {
      consecutiveRef.current += 1;
      setDetectedJS(true);
      setFrameCountJS(consecutiveRef.current);

      if (consecutiveRef.current >= FRAMES_NEEDED) {
        console.log(`${TAG} ✅ ${FRAMES_NEEDED} consecutive frames detected – triggering capture`);
        consecutiveRef.current = 0;
        triggerCaptureJS();
      } else {
        setStatusMsgJS(`Hold still… ${consecutiveRef.current}/${FRAMES_NEEDED}`);
      }
    } else {
      consecutiveRef.current = 0;
      setDetectedJS(false);
      setFrameCountJS(0);
      setStatusMsgJS(
        bestScore >= CONF_THRESHOLD
          ? "Keep full eye inside the box"
          : "Position eye in the box",
      );
    }
  }, []);

  // Async pipeline: worklet resizes → JS runs ONNX → JS updates state
  //
  // ─── WHY ArrayBuffer, not Float32Array? ──────────────────────────────────
  // Worklets.createRunOnJS() serialises arguments across the JSI bridge.
  // Float32Array (and all TypedArrays) lose their numeric data in transit —
  // the bridge reconstructs an object with the correct .length but every
  // element reads as undefined, so Float32Array.from() produces 0 values.
  // ArrayBuffer is treated as a raw binary blob and IS transferred intact.
  // We therefore send .buffer from the worklet and wrap it back into a
  // Float32Array on this (JS) thread where ONNX Runtime can use it safely.
  const runInferenceFromFrame = Worklets.createRunOnJS(
    async (chwArray: number[]) => {
      if (inferenceInFlightRef.current) return; // drop frame — previous call still running
      inferenceInFlightRef.current = true;
      console.log(`${TAG} 🎞 Frame on JS thread — elements: ${chwArray.length}`);
      try {
        const result = await runInference(chwArray);
        if (!result) return;
        handleInferenceResult(
          result.bestScore,
          result.bestX1, result.bestY1,
          result.bestX2, result.bestY2,
        );
      } finally {
        inferenceInFlightRef.current = false;
      }
    },
  );

  // ── Frame Processor ─────────────────────────────────────────────────────────
  //
  // NOTE ON ARCHITECTURE:
  //   react-native-fast-tflite ran the model synchronously inside the worklet.
  //   onnxruntime-react-native is async and cannot run inside a worklet directly.
  //   Solution: worklet only resizes the frame (fast, synchronous), then hands
  //   the Float32Array off to JS thread via Worklets.createRunOnJS().
  //   This keeps the camera pipeline unblocked while ONNX runs async on JS.
  //
  // INPUT FORMAT for MobileNet-SSD ONNX:
  //   Shape : [1, 3, 320, 320]  (NCHW)
  //   Dtype : float32
  //   Range : [0.0, 1.0]  (model was trained with normalised inputs)
  //   Color : RGB
  //   The resize plugin returns HWC by default; we convert to CHW below.
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      // Guard: don't fire until model is loaded
      if (!modelReadyRef.current) return;
      if (capturingRef.current) return;

      // Frame-rate throttle to reduce CPU/GPU load
      frameSkipRef.current += 1;
      if (frameSkipRef.current % FRAME_SKIP_EVERY !== 0) return;

      // Diagnostic: log every 30th kept frame so we know the worklet is running
      if (frameSkipRef.current % 60 === 0) {
        console.log(
          `${TAG} 🎬 Worklet frame #${frameSkipRef.current} — ` +
          `${frame.width}x${frame.height} fmt=${frame.pixelFormat}`,
        );
      }

      try {
        // ── Step 1: Resize to 320×320 RGB float32 HWC ────────────────────
        const resizedHWC = resize(frame, {
          scale:       { width: MODEL_SIZE, height: MODEL_SIZE },
          pixelFormat: "rgb",
          dataType:    "float32",
          rotation:    "270deg",
        });

        // ── Step 2: HWC [H,W,C] → CHW [C,H,W] transpose ─────────────────
        // This must happen IN the worklet. Worklets.createRunOnJS() only
        // supports plain primitives and plain JS arrays — no TypedArrays,
        // no ArrayBuffers. A plain number[] IS supported and survives intact.
        // We build the CHW array here so the JS side just wraps it cheaply.
        const H = MODEL_SIZE, W = MODEL_SIZE, C = 3;
        const TOTAL = C * H * W;
        const chw: number[] = new Array(TOTAL);
        for (let ch = 0; ch < C; ch++) {
          for (let row = 0; row < H; row++) {
            for (let col = 0; col < W; col++) {
              chw[ch * H * W + row * W + col] =
                resizedHWC[row * W * C + col * C + ch];
            }
          }
        }

        // ── Step 3: Send CHW number[] to JS thread for ONNX inference ────
        runInferenceFromFrame(chw);
      } catch (err: any) {
        console.error(`${TAG} ❌ Worklet error:`, err?.message ?? String(err));
      }
    },
    [modelReadyRef, capturingRef, frameSkipRef, resize, runInferenceFromFrame],
  );

  // ── Permission / device guards ──────────────────────────────────────────────
  if (!hasPermission) {
    return (
      <View style={s.center}>
        <Text style={s.permText} onPress={requestPermission}>
          Tap to grant camera permission
        </Text>
      </View>
    );
  }

  if (!device) {
    return (
      <View style={s.center}>
        <Text style={s.label}>No front camera found</Text>
      </View>
    );
  }

  if (modelState === "error") {
    return (
      <View style={s.center}>
        <Text style={[s.label, { color: "#ff5252", textAlign: "center", paddingHorizontal: 24 }]}>
          Failed to load detection model.{"\n"}Please restart the app.
        </Text>
      </View>
    );
  }

  const progressPct = Math.round((frameCount / FRAMES_NEEDED) * 100);

  // ── Render ──────────────────────────────────────────────────────────────────
  return (
    <View style={s.root}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={!capturing}
        frameProcessor={frameProcessor}
        onInitialized={() => {
          cameraReadyRef.current = true;
          console.log(`${TAG} 📷 Camera initialised`);
        }}
        onError={(err) => {
          cameraReadyRef.current = false;
          console.error(`${TAG} ❌ Camera error:`, err.message ?? err);
          setStatusMsg("Camera error – try again");
        }}
        photo
      />

      {/* Vignette overlay (4 rectangles around the guide box) */}
      <View style={[s.vig, { top: 0, left: 0, right: 0, height: GUIDE_TOP }]} />
      <View style={[s.vig, { top: GUIDE_TOP + GUIDE_SIZE, left: 0, right: 0, bottom: 0 }]} />
      <View style={[s.vig, { top: GUIDE_TOP, left: 0, width: GUIDE_LEFT, height: GUIDE_SIZE }]} />
      <View style={[s.vig, { top: GUIDE_TOP, left: GUIDE_LEFT + GUIDE_SIZE, right: 0, height: GUIDE_SIZE }]} />

      {/* Guide box */}
      <View
        style={[
          s.guide,
          detected && !capturing && s.guideDetected,
          capturing && s.guideCapturing,
        ]}
      />

      {/* Debug bounding box overlay */}
      {debugEyeBox && (
        <View
          pointerEvents="none"
          style={[
            s.debugEyeBox,
            debugEyeBox.insideGuide ? s.debugEyeBoxInside : s.debugEyeBoxOutside,
            {
              left:   debugEyeBox.left,
              top:    debugEyeBox.top,
              width:  debugEyeBox.width,
              height: debugEyeBox.height,
            },
          ]}
        >
          <Text
            style={[
              s.debugEyeBoxLabel,
              debugEyeBox.insideGuide ? s.debugEyeBoxLabelInside : s.debugEyeBoxLabelOutside,
            ]}
          >
            {debugEyeBox.insideGuide ? "IN" : "OUT"} {debugEyeBox.score.toFixed(2)}
          </Text>
        </View>
      )}

      {/* Top label */}
      <View style={s.topLabel}>
        <Text style={s.eyeTitle}>EYE DETECTION</Text>
        <Text style={s.eyeSub}>
          Centre your {eyeSide.toUpperCase()} eye inside the box
        </Text>
      </View>

      {/* Progress bar */}
      <View style={s.progressTrack}>
        <View style={[s.progressFill, { width: `${progressPct}%` }]} />
      </View>

      {/* HUD */}
      <View style={s.hud}>
        {capturing ? (
          <View style={s.row}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={s.hudText}> Processing image…</Text>
          </View>
        ) : (
          <>
            <Text style={[s.statusText, detected ? s.statusGreen : s.statusGray]}>
              {statusMsg}
            </Text>
            <View style={s.row}>
              <HudPill label="Side"   value={eyeSide.toUpperCase()} />
              <HudPill label="Model"  value={modelState} />
              <HudPill label="Score"  value={score.toFixed(3)} />
              <HudPill label="Frames" value={`${frameCount}/${FRAMES_NEEDED}`} />
            </View>
          </>
        )}
      </View>
    </View>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function HudPill({ label, value }: { label: string; value: string }) {
  return (
    <View style={s.pill}>
      <Text style={s.pillLabel}>{label}</Text>
      <Text style={s.pillValue}>{value}</Text>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root:    { flex: 1, backgroundColor: "#000" },
  center:  { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#000" },
  permText: { fontSize: 17, color: "#5b2d8e", textAlign: "center", padding: 24 },
  label:   { fontSize: 16, color: "#fff" },

  vig: { position: "absolute", backgroundColor: "rgba(0,0,0,0.58)" },

  guide: {
    position: "absolute",
    left: GUIDE_LEFT, top: GUIDE_TOP,
    width: GUIDE_SIZE, height: GUIDE_SIZE,
    borderWidth: 2.5, borderStyle: "dashed",
    borderColor: "#00e676", borderRadius: 6,
  },
  guideDetected:  { borderStyle: "solid", borderColor: "#00e676" },
  guideCapturing: { borderStyle: "solid", borderColor: "#ffeb3b" },

  debugEyeBox: { position: "absolute", borderWidth: 2, borderRadius: 4, zIndex: 5 },
  debugEyeBoxInside:  { borderColor: "#00e676" },
  debugEyeBoxOutside: { borderColor: "#ff5252" },
  debugEyeBoxLabel: {
    position: "absolute", top: -22, left: -2,
    paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: 4, color: "#000",
    fontSize: 10, fontWeight: "900", overflow: "hidden",
  },
  debugEyeBoxLabelInside:  { backgroundColor: "#00e676" },
  debugEyeBoxLabelOutside: { backgroundColor: "#ff5252", color: "#fff" },

  topLabel: { position: "absolute", top: 56, left: 0, right: 0, alignItems: "center" },
  eyeTitle: { fontSize: 20, fontWeight: "900", color: "#fff", letterSpacing: 3 },
  eyeSub:   { fontSize: 12, color: "rgba(255,255,255,0.55)", marginTop: 4 },

  progressTrack: {
    position: "absolute",
    top: GUIDE_TOP + GUIDE_SIZE + 10, left: GUIDE_LEFT,
    width: GUIDE_SIZE, height: 4,
    backgroundColor: "rgba(255,255,255,0.18)",
    borderRadius: 2, overflow: "hidden",
  },
  progressFill: { height: 4, borderRadius: 2 },

  hud: {
    position: "absolute", bottom: 44, left: 16, right: 16,
    backgroundColor: "rgba(0,0,0,0.78)",
    borderRadius: 14, padding: 14, gap: 8,
  },
  row:     { flexDirection: "row", alignItems: "center", justifyContent: "space-around" },
  hudText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  statusText: { fontSize: 14, fontWeight: "700", textAlign: "center" },
  statusGreen: { color: "#00e676" },
  statusGray:  { color: "rgba(255,255,255,0.55)" },

  pill:      { alignItems: "center", gap: 2 },
  pillLabel: { fontSize: 10, color: "rgba(255,255,255,0.45)", textTransform: "uppercase", letterSpacing: 0.5 },
  pillValue: { fontSize: 13, color: "#fff", fontWeight: "600" },
});