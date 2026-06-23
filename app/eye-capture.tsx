import React, { useRef, useState, useCallback, useEffect } from "react";
import { View, Text, StyleSheet, Dimensions, ActivityIndicator } from "react-native";
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

const TAG = "[EyeCapture]";

const MODEL_SIZE = 320;
const NUM_ANCHORS = 3234;
const CONF_THRESHOLD = 0.75;
const NMS_IOU_THRESHOLD = 0.35;
const FRAMES_NEEDED = 4;
const OUTPUT_SIZE = 512;

const VARIANCE_XY = 0.1;
const VARIANCE_WH = 0.2;

const { width: SW, height: SH } = Dimensions.get("window");
const GUIDE_SIZE = 220;
const GUIDE_LEFT = (SW - GUIDE_SIZE) / 2;
const GUIDE_TOP = SH * 0.35;
const GUIDE_RIGHT = GUIDE_LEFT + GUIDE_SIZE;
const GUIDE_BOTTOM = GUIDE_TOP + GUIDE_SIZE;
const GUIDE_INNER_MARGIN = GUIDE_SIZE * 0.1;

type Box = { x1: number; y1: number; x2: number; y2: number; score: number };

type DebugEyeBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  score: number;
  insideGuide: boolean;
};

// ─── Anchor & Post-processing Functions ─────────────────────────────
function generateSSDAnchors(): Float32Array {
  const specs = [
    { featureMapSize: 20, minScale: 0.1, maxScale: 0.2, aspectRatios: [2, 3] },
    { featureMapSize: 10, minScale: 0.2, maxScale: 0.375, aspectRatios: [2, 3] },
    { featureMapSize: 5, minScale: 0.375, maxScale: 0.55, aspectRatios: [2, 3] },
    { featureMapSize: 3, minScale: 0.55, maxScale: 0.725, aspectRatios: [2, 3] },
    { featureMapSize: 2, minScale: 0.725, maxScale: 0.9, aspectRatios: [2, 3] },
    { featureMapSize: 1, minScale: 0.9, maxScale: 1.075, aspectRatios: [2, 3] },
  ];

  const anchors: number[] = [];
  for (const spec of specs) {
    const { featureMapSize: fms, minScale, maxScale, aspectRatios } = spec;
    const step = 1.0 / fms;
    for (let row = 0; row < fms; row++) {
      for (let col = 0; col < fms; col++) {
        const cx = (col + 0.5) * step;
        const cy = (row + 0.5) * step;

        anchors.push(cy, cx, minScale, minScale);
        const interpScale = Math.sqrt(minScale * maxScale);
        anchors.push(cy, cx, interpScale, interpScale);

        for (const ar of aspectRatios) {
          const sqrtAr = Math.sqrt(ar);
          anchors.push(cy, cx, minScale * sqrtAr, minScale / sqrtAr);
          anchors.push(cy, cx, minScale / sqrtAr, minScale * sqrtAr);
        }
      }
    }
  }

  const out = new Float32Array(NUM_ANCHORS * 4);
  for (let i = 0; i < anchors.length; i++) out[i] = anchors[i];
  return out;
}

function decodeSSDBox(
  dy: number,
  dx: number,
  dh: number,
  dw: number,
  acy: number,
  acx: number,
  ah: number,
  aw: number,
) {
  const cy = dy * VARIANCE_XY * ah + acy;
  const cx = dx * VARIANCE_XY * aw + acx;
  const h = Math.exp(dh * VARIANCE_WH) * ah;
  const w = Math.exp(dw * VARIANCE_WH) * aw;
  return { x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2 };
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

function iou(a: Box, b: Box) {
  const ix1 = Math.max(a.x1, b.x1);
  const iy1 = Math.max(a.y1, b.y1);
  const ix2 = Math.min(a.x2, b.x2);
  const iy2 = Math.min(a.y2, b.y2);
  const iw = Math.max(0, ix2 - ix1);
  const ih = Math.max(0, iy2 - iy1);
  const inter = iw * ih;
  const areaA = Math.max(0, a.x2 - a.x1) * Math.max(0, a.y2 - a.y1);
  const areaB = Math.max(0, b.x2 - b.x1) * Math.max(0, b.y2 - b.y1);
  return inter / (areaA + areaB - inter + 1e-6);
}

function nms(boxes: Box[], threshold: number) {
  const sorted = [...boxes].sort((a, b) => b.score - a.score);
  const kept: Box[] = [];
  while (sorted.length) {
    const current = sorted.shift()!;
    kept.push(current);
    for (let i = sorted.length - 1; i >= 0; i--) {
      if (iou(current, sorted[i]) > threshold) sorted.splice(i, 1);
    }
  }
  return kept;
}

function modelToScreen(nx: number, ny: number) {
  return {
    sx: GUIDE_LEFT + nx * GUIDE_SIZE,
    sy: GUIDE_TOP + ny * GUIDE_SIZE,
  };
}

function getEyeBoxOnScreen(x1: number, y1: number, x2: number, y2: number) {
  const { sx: l1, sy: t1 } = modelToScreen(Math.min(x1, x2), Math.min(y1, y2));
  const { sx: l2, sy: t2 } = modelToScreen(Math.max(x1, x2), Math.max(y1, y2));
  return { left: l1, top: t1, right: l2, bottom: t2, width: l2 - l1, height: t2 - t1 };
}

const ANCHORS = generateSSDAnchors();

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

  const eyeSide = params.eyeSide ?? "right";

  const cameraRef = useRef<Camera>(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("front");
  const { resize } = useResizePlugin();

  const sessionRef = useRef<InferenceSession | null>(null);
  const modelReadyRef = useRef(false);
  const cameraReadyRef = useRef(false);
  const capturingRef = useRef(false);
  const inferenceInFlightRef = useRef(false);
  const consecutiveRef = useRef(0);
  const frameSkipRef = useRef(0);

  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const [detected, setDetected] = useState(false);
  const [score, setScore] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [statusMsg, setStatusMsg] = useState("Position eye in the box");
  const [capturing, setCapturing] = useState(false);
  const [debugEyeBox, setDebugEyeBox] = useState<DebugEyeBox | null>(null);

  const setDetectedJS = Worklets.createRunOnJS(setDetected);
  const setScoreJS = Worklets.createRunOnJS(setScore);
  const setFrameCountJS = Worklets.createRunOnJS(setFrameCount);
  const setStatusMsgJS = Worklets.createRunOnJS(setStatusMsg);
  const setDebugEyeBoxJS = Worklets.createRunOnJS(setDebugEyeBox);

  // Load ONNX Model
  useEffect(() => {
    let mounted = true;
    (async () => {
      try {
        const [asset] = await Asset.loadAsync(require("../assets/model/mobilenet.onnx"));
        if (!asset.localUri) throw new Error("Model localUri missing");

        const modelPath = asset.localUri.replace(/^file:\/\//, "");
        const session = await InferenceSession.create(modelPath, {
          executionProviders: ["coreml", "cpu"],
        });

        if (!mounted) return;
        sessionRef.current = session;
        modelReadyRef.current = true;
        setModelState("ready");
      } catch (err) {
        console.error(TAG, "Model loading failed", err);
        if (mounted) setModelState("error");
      }
    })();

    return () => {
      mounted = false;
    };
  }, []);

  const checkEyeInsideGuide = (box: Box) => {
    const screen = getEyeBoxOnScreen(box.x1, box.y1, box.x2, box.y2);
    const innerLeft = GUIDE_LEFT + GUIDE_INNER_MARGIN;
    const innerTop = GUIDE_TOP + GUIDE_INNER_MARGIN;
    const innerRight = GUIDE_RIGHT - GUIDE_INNER_MARGIN;
    const innerBottom = GUIDE_BOTTOM - GUIDE_INNER_MARGIN;

    const cx = (screen.left + screen.right) / 2;
    const cy = (screen.top + screen.bottom) / 2;
    const w = screen.width;
    const h = screen.height;
    const aspect = w / Math.max(h, 1e-6);

    return (
      cx >= innerLeft && cx <= innerRight &&
      cy >= innerTop && cy <= innerBottom &&
      w > 20 && h > 20 &&
      w < GUIDE_SIZE * 0.9 && h < GUIDE_SIZE * 0.9 &&
      aspect > 0.5 && aspect < 2.5
    );
  };

  // ─── Improved Capture & Navigate (Same as TFLite version) ─────────────────
  const captureAndNavigate = useCallback(async () => {
    if (capturingRef.current) return;

    const camera = cameraRef.current;
    if (!cameraReadyRef.current || !camera) return;

    capturingRef.current = true;
    setCapturing(true);
    setStatusMsg("Capturing high-quality image...");

    try {
      const photo = await camera.takeSnapshot({ quality: 95 });
      const photoUri = photo.path.startsWith("file://") ? photo.path : `file://${photo.path}`;

      const scaleX = photo.width / SW;
      const scaleY = photo.height / SH;

      const cropX = Math.max(0, Math.round(GUIDE_LEFT * scaleX));
      const cropY = Math.max(0, Math.round(GUIDE_TOP * scaleY));
      const cropW = Math.min(Math.round(GUIDE_SIZE * scaleX), photo.width - cropX);
      const cropH = Math.min(Math.round(GUIDE_SIZE * scaleY), photo.height - cropY);

      const manipulated = await ImageManipulator.manipulateAsync(
        photoUri,
        [
          { crop: { originX: cropX, originY: cropY, width: cropW, height: cropH } },
          { resize: { width: OUTPUT_SIZE, height: OUTPUT_SIZE } },
        ],
        { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG }
      );

      if (!eyeSide) {
        console.error("[EyeCapture] ❌ eyeSide is missing!");
        router.back();
        return;
      }

      const imageKey = eyeSide === "left" ? "leftEyeImage" : "rightEyeImage";

      router.replace({
        pathname: "/",
        params: {
          [imageKey]: manipulated.uri,
          name: params.name,
          parentName: params.parentName,
          phoneNumber: params.phoneNumber,
          age: params.age,
          gender: params.gender,
          eyeSessionId: params.eyeSessionId,
        },
      });
    } catch (err: any) {
      console.error("[EyeCapture] ❌ CAPTURE FAILED:", err.message);
      setStatusMsg("Capture failed – please try again");
    } finally {
      setCapturing(false);
      capturingRef.current = false;
    }
  }, [eyeSide, params, router]);

  const triggerCaptureJS = Worklets.createRunOnJS(captureAndNavigate);

  // Rest of your inference logic remains the same...
  const runInference = useCallback(async (chwArray: number[]) => {
    const session = sessionRef.current;
    if (!session) return null;

    const chw = new Float32Array(chwArray);
    const inputTensor = new Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]);
    const results = await session.run({ input: inputTensor });

    const clsData = results["cls_logits"]?.data as Float32Array | undefined;
    const bboxData = results["bbox_regression"]?.data as Float32Array | undefined;
    if (!clsData || !bboxData) return null;

    const candidates: Box[] = [];
    for (let i = 0; i < NUM_ANCHORS; i++) {
      const conf = sigmoid(clsData[i * 2 + 1]);
      if (conf < CONF_THRESHOLD) continue;

      const dy = bboxData[i * 4 + 0];
      const dx = bboxData[i * 4 + 1];
      const dh = bboxData[i * 4 + 2];
      const dw = bboxData[i * 4 + 3];

      const acy = ANCHORS[i * 4 + 0];
      const acx = ANCHORS[i * 4 + 1];
      const ah = ANCHORS[i * 4 + 2];
      const aw = ANCHORS[i * 4 + 3];

      const box = decodeSSDBox(dy, dx, dh, dw, acy, acx, ah, aw);
      candidates.push({ ...box, score: conf });
    }

    if (candidates.length === 0) return null;
    return nms(candidates, NMS_IOU_THRESHOLD)[0] ?? null;
  }, []);

  const handleInferenceResult = useCallback((box: Box | null) => {
    if (!box) {
      setScoreJS(0);
      setDetectedJS(false);
      setFrameCountJS(0);
      setStatusMsgJS("Position eye in the box");
      setDebugEyeBoxJS(null);
      consecutiveRef.current = 0;
      return;
    }

    setScoreJS(box.score);
    const inside = checkEyeInsideGuide(box);
    const screen = getEyeBoxOnScreen(box.x1, box.y1, box.x2, box.y2);

    setDebugEyeBoxJS({
      left: screen.left,
      top: screen.top,
      width: screen.width,
      height: screen.height,
      score: box.score,
      insideGuide: inside,
    });

    if (inside) {
      consecutiveRef.current += 1;
      setDetectedJS(true);
      setFrameCountJS(consecutiveRef.current);

      if (consecutiveRef.current >= FRAMES_NEEDED) {
        consecutiveRef.current = 0;
        triggerCaptureJS();
      } else {
        setStatusMsgJS(`Hold still... ${consecutiveRef.current}/${FRAMES_NEEDED}`);
      }
    } else {
      consecutiveRef.current = 0;
      setDetectedJS(false);
      setFrameCountJS(0);
      setStatusMsgJS(box.score >= CONF_THRESHOLD ? "Keep full eye inside the box" : "Position eye in the box");
    }
  }, [triggerCaptureJS]);

  const runInferenceFromFrame = Worklets.createRunOnJS(async (chwArray: number[]) => {
    if (inferenceInFlightRef.current) return;
    inferenceInFlightRef.current = true;
    try {
      const result = await runInference(chwArray);
      handleInferenceResult(result);
    } finally {
      inferenceInFlightRef.current = false;
    }
  });

  const frameProcessor = useFrameProcessor((frame) => {
    "worklet";
    if (!modelReadyRef.current) return;
    if (capturingRef.current) return;

    frameSkipRef.current += 1;
    if (frameSkipRef.current % 2 !== 0) return;

    try {
      const resizedHWC = resize(frame, {
        scale: { width: MODEL_SIZE, height: MODEL_SIZE },
        pixelFormat: "rgb",
        dataType: "float32",
        rotation: "270deg",
      });

      const H = MODEL_SIZE;
      const W = MODEL_SIZE;
      const C = 3;
      const chw: number[] = new Array(C * H * W);

      for (let ch = 0; ch < C; ch++) {
        for (let row = 0; row < H; row++) {
          for (let col = 0; col < W; col++) {
            chw[ch * H * W + row * W + col] =
              resizedHWC[row * W * C + col * C + ch];
          }
        }
      }

      runInferenceFromFrame(chw);
    } catch {}
  }, [resize, runInferenceFromFrame]);

  // ─── UI Rendering ─────────────────────────────────────────────────
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

  return (
    <View style={s.root}>
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={!capturing}
        frameProcessor={frameProcessor}
        onInitialized={() => (cameraReadyRef.current = true)}
        onError={() => {
          cameraReadyRef.current = false;
          setStatusMsg("Camera error");
        }}
        photo
      />

      {/* Vignette */}
      <View style={[s.vig, { top: 0, left: 0, right: 0, height: GUIDE_TOP }]} />
      <View style={[s.vig, { top: GUIDE_TOP + GUIDE_SIZE, left: 0, right: 0, bottom: 0 }]} />
      <View style={[s.vig, { top: GUIDE_TOP, left: 0, width: GUIDE_LEFT, height: GUIDE_SIZE }]} />
      <View style={[s.vig, { top: GUIDE_TOP, left: GUIDE_LEFT + GUIDE_SIZE, right: 0, height: GUIDE_SIZE }]} />

      <View style={[s.guide, detected && !capturing && s.guideDetected, capturing && s.guideCapturing]} />

      {debugEyeBox && (
        <View
          pointerEvents="none"
          style={[
            s.debugEyeBox,
            debugEyeBox.insideGuide ? s.debugEyeBoxInside : s.debugEyeBoxOutside,
            {
              left: debugEyeBox.left,
              top: debugEyeBox.top,
              width: debugEyeBox.width,
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

      <View style={s.topLabel}>
        <Text style={s.eyeTitle}>EYE DETECTION</Text>
        <Text style={s.eyeSub}>Centre your {eyeSide.toUpperCase()} eye inside the box</Text>
      </View>

      <View style={s.progressTrack}>
        <View style={[s.progressFill, { width: `${progressPct}%` }]} />
      </View>

      <View style={s.hud}>
        {capturing ? (
          <View style={s.row}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={s.hudText}> Processing image...</Text>
          </View>
        ) : (
          <>
            <Text style={[s.statusText, detected ? s.statusGreen : s.statusGray]}>
              {statusMsg}
            </Text>
            <View style={s.row}>
              <HudPill label="Side" value={eyeSide.toUpperCase()} />
              <HudPill label="Model" value={modelState} />
              <HudPill label="Score" value={score.toFixed(3)} />
              <HudPill label="Frames" value={`${frameCount}/${FRAMES_NEEDED}`} />
            </View>
          </>
        )}
      </View>
    </View>
  );
}

function HudPill({ label, value }: { label: string; value: string }) {
  return (
    <View style={s.pill}>
      <Text style={s.pillLabel}>{label}</Text>
      <Text style={s.pillValue}>{value}</Text>
    </View>
  );
}

const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#000" },
  permText: { fontSize: 17, color: "#5b2d8e", textAlign: "center", padding: 24 },
  label: { fontSize: 16, color: "#fff" },
  vig: { position: "absolute", backgroundColor: "rgba(0,0,0,0.58)" },
  guide: {
    position: "absolute",
    left: GUIDE_LEFT,
    top: GUIDE_TOP,
    width: GUIDE_SIZE,
    height: GUIDE_SIZE,
    borderWidth: 2.5,
    borderStyle: "dashed",
    borderColor: "#00e676",
    borderRadius: 6,
  },
  guideDetected: { borderStyle: "solid", borderColor: "#00e676" },
  guideCapturing: { borderStyle: "solid", borderColor: "#ffeb3b" },
  debugEyeBox: { position: "absolute", borderWidth: 2, borderRadius: 4, zIndex: 5 },
  debugEyeBoxInside: { borderColor: "#00e676" },
  debugEyeBoxOutside: { borderColor: "#ff5252" },
  debugEyeBoxLabel: {
    position: "absolute",
    top: -22,
    left: -2,
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 4,
    color: "#000",
    fontSize: 10,
    fontWeight: "900",
    overflow: "hidden",
  },
  debugEyeBoxLabelInside: { backgroundColor: "#00e676" },
  debugEyeBoxLabelOutside: { backgroundColor: "#ff5252", color: "#fff" },
  topLabel: { position: "absolute", top: 56, left: 0, right: 0, alignItems: "center" },
  eyeTitle: { fontSize: 20, fontWeight: "900", color: "#fff", letterSpacing: 3 },
  eyeSub: { fontSize: 12, color: "rgba(255,255,255,0.55)", marginTop: 4 },
  progressTrack: {
    position: "absolute",
    top: GUIDE_TOP + GUIDE_SIZE + 10,
    left: GUIDE_LEFT,
    width: GUIDE_SIZE,
    height: 4,
    backgroundColor: "rgba(255,255,255,0.18)",
    borderRadius: 2,
    overflow: "hidden",
  },
  progressFill: { height: 4, borderRadius: 2, backgroundColor: "#00e676" },
  hud: {
    position: "absolute",
    bottom: 44,
    left: 16,
    right: 16,
    backgroundColor: "rgba(0,0,0,0.78)",
    borderRadius: 14,
    padding: 14,
    gap: 8,
  },
  row: { flexDirection: "row", alignItems: "center", justifyContent: "space-around" },
  hudText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  statusText: { fontSize: 14, fontWeight: "700", textAlign: "center" },
  statusGreen: { color: "#00e676" },
  statusGray: { color: "rgba(255,255,255,0.55)" },
  pill: { alignItems: "center", gap: 2 },
  pillLabel: { fontSize: 10, color: "rgba(255,255,255,0.45)", textTransform: "uppercase", letterSpacing: 0.5 },
  pillValue: { fontSize: 13, color: "#fff", fontWeight: "600" },
});