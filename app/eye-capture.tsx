// app/eye-capture.tsx
import React, { useRef, useState, useCallback, useEffect } from "react";
import {
  View,
  Text,
  StyleSheet,
  Dimensions,
  ActivityIndicator,
  TouchableOpacity,
  Platform,
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
import * as Speech from "expo-speech";
import Svg, { Path, Circle } from "react-native-svg";
import { Asset } from "expo-asset";

const TAG = `[EYE:${Platform.OS}]`;
const LOG_EVERY_N_FRAMES = 15;

const MODEL_SIZE = 320;
const CONF_THRESHOLD = 0.25;
const FRAMES_NEEDED = 4;
const OUTPUT_SIZE = 512;

const { width: SW, height: SH } = Dimensions.get("window");
const GUIDE_SIZE = 220;
const GUIDE_LEFT = (SW - GUIDE_SIZE) / 2;
const GUIDE_TOP = SH * 0.35;
const GUIDE_RIGHT = GUIDE_LEFT + GUIDE_SIZE;
const GUIDE_BOTTOM = GUIDE_TOP + GUIDE_SIZE;
const GUIDE_INNER_MARGIN = GUIDE_SIZE * 0.1;

type Lang = "en" | "hi" | "te";

type DebugEyeBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  score: number;
  insideGuide: boolean;
};

// ── Coordinate helpers ────────────────────────────────────────────

function getEyeBoxOnScreen(x1: number, y1: number, x2: number, y2: number) {
  "worklet";
  const left   = (Math.min(x1, x2) / MODEL_SIZE) * SW;
  const top    = (Math.min(y1, y2) / MODEL_SIZE) * SH;
  const right  = (Math.max(x1, x2) / MODEL_SIZE) * SW;
  const bottom = (Math.max(y1, y2) / MODEL_SIZE) * SH;
  return { left, top, right, bottom, width: right - left, height: bottom - top };
}

function checkEyeWithinGuideBox(x1: number, y1: number, x2: number, y2: number) {
  "worklet";
  // Model outputs a large near-full-frame box; requiring the whole box
  // to fit inside the guide will always fail. Check center point instead.
  const cx = ((x1 + x2) / 2 / MODEL_SIZE) * SW;
  const cy = ((y1 + y2) / 2 / MODEL_SIZE) * SH;

  const innerLeft   = GUIDE_LEFT   + GUIDE_INNER_MARGIN;
  const innerTop    = GUIDE_TOP    + GUIDE_INNER_MARGIN;
  const innerRight  = GUIDE_RIGHT  - GUIDE_INNER_MARGIN;
  const innerBottom = GUIDE_BOTTOM - GUIDE_INNER_MARGIN;

  return (
    cx >= innerLeft  &&
    cx <= innerRight &&
    cy >= innerTop   &&
    cy <= innerBottom
  );
}

function getResizeRotation(
  orientation: string,
): "0deg" | "90deg" | "180deg" | "270deg" {
  "worklet";
  if (Platform.OS === "android") return "270deg";
  switch (orientation) {
    case "landscape-right":       return "90deg";
    case "landscape-left":        return "270deg";
    case "portrait-upside-down":  return "180deg";
    case "portrait":
    default:                      return "90deg";
  }
}

// ── Voice ─────────────────────────────────────────────────────────

const VOICE: Record<string, Record<Lang, { text: string; bcp47: string }>> = {
  alignFace: {
    en: { text: "Pull down your lower eyelid and fill the box with the pink area.", bcp47: "en-IN" },
    hi: { text: "निचली पलक को नीचे खींचें और गुलाबी हिस्से से बॉक्स भरें।", bcp47: "hi-IN" },
    te: { text: "కింది రెప్పను కిందకు లాగి, గులాబీ భాగంతో బాక్స్ నింపండి.", bcp47: "te-IN" },
  },
  allGood: {
    en: { text: "Good! Capturing now.", bcp47: "en-IN" },
    hi: { text: "बढ़िया! अभी कैप्चर हो रहा है।", bcp47: "hi-IN" },
    te: { text: "బాగుంది! ఇప్పుడు క్యాప్చర్ అవుతోంది.", bcp47: "te-IN" },
  },
};

// ── Icons ─────────────────────────────────────────────────────────

const IconEye = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path
      d="M1 12C1 12 5 5 12 5C19 5 23 12 23 12C23 12 19 19 12 19C5 19 1 12 1 12Z"
      stroke={color} strokeWidth="2" strokeLinejoin="round"
    />
    <Circle cx="12" cy="12" r="3.5" stroke={color} strokeWidth="2" />
  </Svg>
);

const IconSpeaker = ({ size = 22, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path
      d="M11 5L6 9H3C2.45 9 2 9.45 2 10V14C2 14.55 2.45 15 3 15H6L11 19V5Z"
      stroke={color} strokeWidth="2" strokeLinejoin="round"
    />
    <Path
      d="M15.54 8.46C16.48 9.4 17 10.67 17 12C17 13.33 16.48 14.6 15.54 15.54"
      stroke={color} strokeWidth="2" strokeLinecap="round"
    />
    <Path
      d="M19.07 4.93C20.96 6.82 22 9.35 22 12C22 14.65 20.96 17.18 19.07 19.07"
      stroke={color} strokeWidth="2" strokeLinecap="round"
    />
  </Svg>
);

// ─────────────────────────────────────────────────────────────────
//  COMPONENT
// ─────────────────────────────────────────────────────────────────
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
    leftEyeImage?: string;
    rightEyeImage?: string;
  }>();

  const { eyeSide = "right" } = params;

  const cameraRef = useRef<Camera>(null);
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("front");

  const [modelState, setModelState] = useState<"loading" | "ready" | "error">("loading");
  const sessionRef = useRef<InferenceSession | null>(null);

  // ── Load ONNX model ───────────────────────────────────────────
  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [asset] = await Asset.loadAsync(
          require("../assets/model/2026-06-24__mnv3ssd-320-best-model-with-postprocess.onnx"),
        );
        if (!asset.localUri) throw new Error("Asset failed to resolve to a local URI");
        const modelPath = asset.localUri.replace("file://", "");
        console.log(`${TAG} 📦 model path: ${modelPath}`);
        const s = await InferenceSession.create(modelPath);
        if (!cancelled) {
          sessionRef.current = s;
          setModelState("ready");
          console.log(`${TAG} ✅ ONNX loaded. inputs=${s.inputNames} outputs=${s.outputNames}`);
        }
      } catch (err: any) {
        if (!cancelled) {
          console.error(`${TAG} ❌ ONNX load failed:`, err?.message ?? String(err));
          setModelState("error");
        }
      }
    })();
    return () => { cancelled = true; };
  }, []);

  const { resize } = useResizePlugin();

  // ── UI state ──────────────────────────────────────────────────
  const [detected,    setDetected]    = useState(false);
  const [score,       setScore]       = useState(0);
  const [frameCount,  setFrameCount]  = useState(0);
  const [statusMsg,   setStatusMsg]   = useState("Position eye in the box");
  const [capturing,   setCapturing]   = useState(false);
  const [debugEyeBox, setDebugEyeBox] = useState<DebugEyeBox | null>(null);
  const [inferMs,     setInferMs]     = useState(0);   // ← inference timing

  const [lang,        setLang]        = useState<Lang>("en");
  const [isSpeaking,  setIsSpeaking]  = useState(false);

  const cameraReadyRef    = useRef(false);
  const capturingRef      = useRef(false);
  const consecutiveRef    = useRef(0);
  const frameLogCounterRef = useRef(0);
  const lastSpokenKey     = useRef<string | null>(null);
  const lastSpokenTime    = useRef<number>(0);

  // Worklet → JS bridges
  const setDetectedJS   = Worklets.createRunOnJS(setDetected);
  const setScoreJS      = Worklets.createRunOnJS(setScore);
  const setFrameCountJS = Worklets.createRunOnJS(setFrameCount);
  const setStatusMsgJS  = Worklets.createRunOnJS(setStatusMsg);
  const setDebugEyeBoxJS = Worklets.createRunOnJS(setDebugEyeBox);
  const setInferMsJS    = Worklets.createRunOnJS(setInferMs);

  useEffect(() => () => Speech.stop(), []);

  // ── Capture & navigate ────────────────────────────────────────
  const captureAndNavigate = useCallback(async () => {
    if (capturingRef.current) return;
    const camera = cameraRef.current;
    if (!cameraReadyRef.current || !camera) {
      console.warn(`${TAG} ⚠️ capture blocked — camera not ready`);
      return;
    }

    capturingRef.current = true;
    setCapturing(true);
    setStatusMsg("Capturing…");

    try {
      const photo = await camera.takeSnapshot({ quality: 95 });
      const photoUri = photo.path.startsWith("file://") ? photo.path : `file://${photo.path}`;

      const scaleX = photo.width  / SW;
      const scaleY = photo.height / SH;

      const cropX = Math.max(0, Math.round(GUIDE_LEFT * scaleX));
      const cropY = Math.max(0, Math.round(GUIDE_TOP  * scaleY));
      const cropW = Math.min(Math.round(GUIDE_SIZE * scaleX), photo.width  - cropX);
      const cropH = Math.min(Math.round(GUIDE_SIZE * scaleY), photo.height - cropY);

      const manipulated = await ImageManipulator.manipulateAsync(
        photoUri,
        [
          { crop: { originX: cropX, originY: cropY, width: cropW, height: cropH } },
          { resize: { width: OUTPUT_SIZE, height: OUTPUT_SIZE } },
        ],
        { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
      );

      console.log(`${TAG} ✅ Captured ${eyeSide} eye: ${manipulated.uri}`);

      router.replace({
        pathname: "/",
        params: {
          eyeImage:      manipulated.uri,
          eyeSide:       eyeSide,
          leftEyeImage:  eyeSide === "left"  ? manipulated.uri : (params.leftEyeImage  ?? ""),
          rightEyeImage: eyeSide === "right" ? manipulated.uri : (params.rightEyeImage ?? ""),
          name:          params.name,
          parentName:    params.parentName,
          phoneNumber:   params.phoneNumber,
          age:           params.age,
          gender:        params.gender,
          eyeSessionId:  params.eyeSessionId,
        },
      });
    } catch (err: any) {
      console.error(`${TAG} ❌ CAPTURE FAILED:`, err);
      setStatusMsg("Capture failed — try again");
    } finally {
      setCapturing(false);
      capturingRef.current = false;
    }
  }, [router, eyeSide, params]);

  const triggerCaptureJS = Worklets.createRunOnJS(captureAndNavigate);

  // ── ONNX inference (JS thread) ────────────────────────────────
  const runOnnxDetection = Worklets.createRunOnJS(
    async (hwcBuffer: Float32Array) => {
      const s = sessionRef.current;
      if (!s || capturingRef.current) return;

      try {
        // HWC → CHW transpose
        const chw = new Float32Array(3 * MODEL_SIZE * MODEL_SIZE);
        const px  = MODEL_SIZE * MODEL_SIZE;
        for (let i = 0; i < px; i++) {
          chw[i]          = hwcBuffer[i * 3];
          chw[i + px]     = hwcBuffer[i * 3 + 1];
          chw[i + px * 2] = hwcBuffer[i * 3 + 2];
        }

        const inputTensor = new Tensor("float32", chw, [1, 3, MODEL_SIZE, MODEL_SIZE]);

        // ── Inference timing ──────────────────────────────────
        const t0 = performance.now();
        const results = await s.run({ input: inputTensor });
        const inferenceMs = performance.now() - t0;
        setInferMsJS(Math.round(inferenceMs));
        console.log(`${TAG} ⏱ inference: ${inferenceMs.toFixed(1)} ms`);
        // ──────────────────────────────────────────────────────

        const detTensor = results["detections"];

        if (!detTensor) {
          console.warn(`${TAG} ⚠️ no "detections" output`);
          setDetectedJS(false);
          setScoreJS(0);
          setDebugEyeBoxJS(null);
          setStatusMsgJS("No eye detected");
          consecutiveRef.current = 0;
          setFrameCountJS(0);
          return;
        }

        const raw           = detTensor.data as Float32Array;
        const numDetections = raw.length / 6;

        let bestScore = 0, bestX1 = 0, bestY1 = 0, bestX2 = 0, bestY2 = 0;
        for (let i = 0; i < numDetections; i++) {
          const o = i * 6;
          if (raw[o + 4] > bestScore) {
            bestScore = raw[o + 4];
            bestX1 = raw[o];  bestY1 = raw[o + 1];
            bestX2 = raw[o + 2]; bestY2 = raw[o + 3];
          }
        }

        console.log(
          `${TAG} detection score=${bestScore.toFixed(3)} box=[${bestX1.toFixed(1)},${bestY1.toFixed(1)},${bestX2.toFixed(1)},${bestY2.toFixed(1)}]`,
        );

        setScoreJS(bestScore);

        if (bestScore < CONF_THRESHOLD) {
          setDetectedJS(false);
          setDebugEyeBoxJS(null);
          setStatusMsgJS("Position eye in the box");
          consecutiveRef.current = 0;
          setFrameCountJS(0);
          return;
        }

        const box          = getEyeBoxOnScreen(bestX1, bestY1, bestX2, bestY2);
        const insideGuide  = checkEyeWithinGuideBox(bestX1, bestY1, bestX2, bestY2);

        setDebugEyeBoxJS({
          left: box.left, top: box.top,
          width: box.width, height: box.height,
          score: bestScore, insideGuide,
        });

        if (insideGuide) {
          consecutiveRef.current += 1;
          setDetectedJS(true);
          setFrameCountJS(consecutiveRef.current);

          if (consecutiveRef.current >= FRAMES_NEEDED) {
            console.log(`${TAG} 🎯 triggering capture (${FRAMES_NEEDED} frames held)`);
            consecutiveRef.current = 0;
            triggerCaptureJS();
          } else {
            setStatusMsgJS(`Hold still… ${consecutiveRef.current}/${FRAMES_NEEDED}`);
          }
        } else {
          consecutiveRef.current = 0;
          setDetectedJS(false);
          setFrameCountJS(0);
          setStatusMsgJS("Move eye into the box");
        }
      } catch (err: any) {
        console.error(`${TAG} ❌ ONNX inference failed:`, err?.message ?? String(err));
      }
    },
  );

  // ── Frame processor ───────────────────────────────────────────
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      frameLogCounterRef.current += 1;
      const shouldLog = frameLogCounterRef.current % LOG_EVERY_N_FRAMES === 0;

      if (!sessionRef.current) {
        if (shouldLog) console.log(`${TAG} ⏳ session not ready — frame ${frameLogCounterRef.current}`);
        return;
      }
      if (capturingRef.current) return;

      const rotation = getResizeRotation(frame.orientation);

      if (shouldLog) {
        console.log(
          `${TAG} frame w=${frame.width} h=${frame.height} orientation=${frame.orientation} rotation=${rotation}`,
        );
      }

      let resized: Float32Array;
      try {
        resized = resize(frame, {
          scale: { width: MODEL_SIZE, height: MODEL_SIZE },
          pixelFormat: "rgb",
          dataType: "float32",
          rotation,
        }) as Float32Array;
      } catch (err: any) {
        console.error(`${TAG} ❌ resize failed:`, err?.message ?? String(err));
        return;
      }

      runOnnxDetection(resized);
    },
    [sessionRef],
  );

  // ── Voice guidance ────────────────────────────────────────────
  const speakNow = useCallback(
    (key: string, l: Lang = lang) => {
      const entry = VOICE[key]?.[l];
      if (!entry) return;
      Speech.stop();
      setIsSpeaking(true);
      Speech.speak(entry.text, {
        language: entry.bcp47,
        rate: 1.0,
        pitch: 1.0,
        onDone:  () => setIsSpeaking(false),
        onError: () => setIsSpeaking(false),
      });
    },
    [lang],
  );

  const problemKey = detected ? "allGood" : "alignFace";
  const instrText  = VOICE[problemKey]?.[lang]?.text ?? "";

  useEffect(() => {
    const now = Date.now();
    if (
      problemKey !== lastSpokenKey.current ||
      now - lastSpokenTime.current > 8000
    ) {
      lastSpokenKey.current  = problemKey;
      lastSpokenTime.current = now;
      speakNow(problemKey, lang);
    }
  }, [problemKey, lang, speakNow]);

  // ── Guards ────────────────────────────────────────────────────
  if (!hasPermission) {
    return (
      <View style={s.center}>
        <Text style={s.permText} onPress={async () => { await requestPermission(); }}>
          Tap to grant camera permission
        </Text>
      </View>
    );
  }
  if (!device) {
    return <View style={s.center}><Text style={s.label}>No front camera found</Text></View>;
  }
  if (modelState === "loading") {
    return (
      <View style={s.center}>
        <ActivityIndicator size="large" color="#00c853" />
        <Text style={[s.permText, { marginTop: 14 }]}>Loading model…</Text>
      </View>
    );
  }
  if (modelState === "error") {
    return (
      <View style={s.center}>
        <Text style={[s.permText, { color: "#ff5555" }]}>❌ Model failed to load</Text>
      </View>
    );
  }

  // ── Derived UI ────────────────────────────────────────────────
  const progressPct = Math.min(100, Math.round((frameCount / FRAMES_NEEDED) * 100));
  const guideColor  = capturing ? "#ffeb3b" : detected ? "#00ff66" : "#4c9fff";

  let captureBtnLabel = "Align eye…";
  if (capturing)  captureBtnLabel = "Capturing…";
  else if (detected) captureBtnLabel = `📸 Auto-capturing… ${frameCount}/${FRAMES_NEEDED}`;

  return (
    <View style={s.root}>

      {/* Camera */}
      <Camera
        ref={cameraRef}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={!capturing}
        frameProcessor={frameProcessor}
        onInitialized={() => {
          cameraReadyRef.current = true;
          console.log(`${TAG} ✅ camera initialized`);
        }}
        onError={(error) => {
          cameraReadyRef.current = false;
          console.error(`${TAG} ❌ camera error:`, error?.message ?? String(error));
          setStatusMsg("Camera error — try again");
        }}
        photo
      />

      {/* Vignette */}
      <View style={[s.vig, { top: 0, left: 0, right: 0, height: GUIDE_TOP }]} />
      <View style={[s.vig, { top: GUIDE_TOP + GUIDE_SIZE, left: 0, right: 0, bottom: 0 }]} />
      <View style={[s.vig, { top: GUIDE_TOP, left: 0, width: GUIDE_LEFT, height: GUIDE_SIZE }]} />
      <View style={[s.vig, { top: GUIDE_TOP, left: GUIDE_LEFT + GUIDE_SIZE, right: 0, height: GUIDE_SIZE }]} />

      {/* Guide box */}
      <View style={[s.guide, { borderColor: guideColor, borderStyle: detected ? "solid" : "dashed" }]}>
        {(["TL", "TR", "BL", "BR"] as const).map((c) => (
          <View key={c} style={[s.corner, (s as any)[`corner${c}`], { borderColor: guideColor }]} />
        ))}
        <View style={s.verticalLine} />
        <View style={s.horizontalLine} />
        <View style={[s.centerDot, { borderColor: guideColor }]} />
      </View>

      {/* Debug eye box overlay */}
      {debugEyeBox && (
        <View
          pointerEvents="none"
          style={[
            s.debugEyeBox,
            debugEyeBox.insideGuide ? s.debugEyeBoxInside : s.debugEyeBoxOutside,
            { left: debugEyeBox.left, top: debugEyeBox.top, width: debugEyeBox.width, height: debugEyeBox.height },
          ]}
        >
          <Text style={[s.debugEyeBoxLabel, debugEyeBox.insideGuide ? s.debugLabelInside : s.debugLabelOutside]}>
            {debugEyeBox.insideGuide ? "IN" : "OUT"} {(debugEyeBox.score * 100).toFixed(0)}%
          </Text>
        </View>
      )}

      {/* Progress bar */}
      <View style={s.progressTrack}>
        <View style={[s.progressFill, { width: `${progressPct}%`, backgroundColor: guideColor }]} />
      </View>

      {/* Top panel */}
      <View style={s.topPanel}>
        <View style={s.controlRow}>
          <TouchableOpacity style={s.backBtn} onPress={() => router.back()}>
            <Text style={s.backBtnText}>← Back</Text>
          </TouchableOpacity>

          <View style={s.sideLabel}>
            <Text style={s.sideLabelText}>{eyeSide.toUpperCase()} EYE</Text>
          </View>

          <View style={s.radioGroup}>
            {(["en", "hi", "te"] as Lang[]).map((l, idx) => (
              <TouchableOpacity
                key={l}
                style={[
                  s.radioBtn,
                  idx === 0 && s.radioBtnFirst,
                  idx === 2 && s.radioBtnLast,
                  lang === l && s.radioBtnActive,
                ]}
                onPress={() => setLang(l)}
              >
                <Text style={[s.radioLabel, lang === l && s.radioLabelActive]}>
                  {l === "en" ? "EN" : l === "hi" ? "हि" : "తె"}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        {/* Status pill */}
        <View style={[s.statusPill, { borderColor: guideColor }]}>
          <IconEye size={16} color={guideColor} />
          <Text style={[s.statusText, { color: guideColor }]}>{statusMsg}</Text>
        </View>

        {/* Voice instruction */}
        <View style={[s.instrBox, { borderColor: detected ? "#00e676" : "rgba(255,255,255,0.15)" }]}>
          {isSpeaking && (
            <View style={s.speakingDot}>
              <View style={[s.speakingPulse, { backgroundColor: detected ? "#00e676" : "#f4a97f" }]} />
            </View>
          )}
          <Text style={[s.instrText, { color: detected ? "#00e676" : "#fff" }]}>{instrText}</Text>
          <TouchableOpacity style={s.speakBtn} onPress={() => speakNow(problemKey)}>
            <IconSpeaker size={20} color={detected ? "#00e676" : "#fff"} />
          </TouchableOpacity>
        </View>

        <View style={s.hintBox}>
          <Text style={s.hintText}>📱 Hold 10–15 cm • Pull lower eyelid down • Fill the guide box</Text>
        </View>
      </View>

      {/* Bottom debug + capture */}
      <View style={s.captureArea}>
        <Text style={s.debugText}>
          score: {score.toFixed(2)}  |  frames: {frameCount}/{FRAMES_NEEDED}  |  infer: {inferMs} ms
        </Text>
        <Text style={s.debugText}>
          model: {modelState}  |  side: {eyeSide}
        </Text>

        <TouchableOpacity
          style={[
            s.captureBtn,
            { backgroundColor: detected ? "#00c853" : "#2a2a2a", borderColor: detected ? "#00ff66" : "#444" },
            capturing && { opacity: 0.6 },
          ]}
          onPress={captureAndNavigate}
          disabled={capturing}
          activeOpacity={0.75}
        >
          {capturing
            ? <ActivityIndicator color="#fff" />
            : <Text style={[s.captureBtnText, { opacity: detected ? 1 : 0.4 }]}>{captureBtnLabel}</Text>
          }
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────
//  STYLES
// ─────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root:    { flex: 1, backgroundColor: "#000" },
  center:  { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#000", padding: 24 },
  permText:{ fontSize: 16, color: "#fff", textAlign: "center" },
  label:   { fontSize: 16, color: "#fff" },

  vig: { position: "absolute", backgroundColor: "rgba(0,0,0,0.58)" },

  guide: {
    position: "absolute",
    left: GUIDE_LEFT, top: GUIDE_TOP,
    width: GUIDE_SIZE, height: GUIDE_SIZE,
    borderWidth: 2.5, borderRadius: 6,
    justifyContent: "center", alignItems: "center",
  },
  corner: { position: "absolute", width: 22, height: 22, borderWidth: 3 },
  cornerTL: { top: -2, left: -2,   borderBottomWidth: 0, borderRightWidth: 0,  borderTopLeftRadius: 8 },
  cornerTR: { top: -2, right: -2,  borderBottomWidth: 0, borderLeftWidth: 0,   borderTopRightRadius: 8 },
  cornerBL: { bottom: -2, left: -2,  borderTopWidth: 0, borderRightWidth: 0,   borderBottomLeftRadius: 8 },
  cornerBR: { bottom: -2, right: -2, borderTopWidth: 0, borderLeftWidth: 0,    borderBottomRightRadius: 8 },
  verticalLine:   { position: "absolute", width: 1,  height: "80%", backgroundColor: "rgba(255,255,255,0.2)" },
  horizontalLine: { position: "absolute", height: 1, width:  "80%", backgroundColor: "rgba(255,255,255,0.2)" },
  centerDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 2 },

  debugEyeBox: { position: "absolute", borderWidth: 2, borderRadius: 4, zIndex: 5 },
  debugEyeBoxInside:  { borderColor: "#00e676" },
  debugEyeBoxOutside: { borderColor: "#ff5252" },
  debugEyeBoxLabel: {
    position: "absolute", top: -22, left: -2,
    paddingHorizontal: 6, paddingVertical: 2,
    borderRadius: 4, fontSize: 10, fontWeight: "900", overflow: "hidden",
  },
  debugLabelInside:  { backgroundColor: "#00e676", color: "#000" },
  debugLabelOutside: { backgroundColor: "#ff5252", color: "#fff" },

  progressTrack: {
    position: "absolute",
    top: GUIDE_TOP + GUIDE_SIZE + 10,
    left: GUIDE_LEFT, width: GUIDE_SIZE, height: 4,
    backgroundColor: "rgba(255,255,255,0.18)", borderRadius: 2, overflow: "hidden",
  },
  progressFill: { height: 4, borderRadius: 2 },

  topPanel: {
    position: "absolute", top: 0, left: 0, right: 0,
    paddingTop: 46, paddingHorizontal: 12, paddingBottom: 8,
    backgroundColor: "rgba(0,0,0,0.80)", gap: 6, zIndex: 10,
  },
  controlRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },
  backBtn: {
    paddingVertical: 6, paddingHorizontal: 12, borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.1)", borderWidth: 1, borderColor: "rgba(255,255,255,0.2)",
  },
  backBtnText: { color: "#fff", fontSize: 12, fontWeight: "600" },
  sideLabel: { backgroundColor: "rgba(91,45,142,0.8)", paddingVertical: 4, paddingHorizontal: 12, borderRadius: 20 },
  sideLabelText: { color: "#fff", fontWeight: "700", fontSize: 12, letterSpacing: 1 },

  radioGroup: { flexDirection: "row", borderRadius: 10, overflow: "hidden", borderWidth: 1, borderColor: "rgba(255,255,255,0.2)" },
  radioBtn: { paddingVertical: 6, paddingHorizontal: 10, backgroundColor: "rgba(255,255,255,0.07)", borderRightWidth: 1, borderRightColor: "rgba(255,255,255,0.15)" },
  radioBtnFirst: { borderTopLeftRadius: 10, borderBottomLeftRadius: 10 },
  radioBtnLast:  { borderTopRightRadius: 10, borderBottomRightRadius: 10, borderRightWidth: 0 },
  radioBtnActive: { backgroundColor: "rgba(255,215,0,0.18)" },
  radioLabel: { color: "rgba(255,255,255,0.6)", fontSize: 11, fontWeight: "700" },
  radioLabelActive: { color: "#FFD700" },

  statusPill: {
    flexDirection: "row", alignItems: "center", gap: 8,
    borderWidth: 1, borderRadius: 12,
    paddingVertical: 8, paddingHorizontal: 12,
    backgroundColor: "rgba(255,255,255,0.06)",
  },
  statusText: { fontSize: 13, fontWeight: "700", flex: 1 },

  instrBox: {
    borderWidth: 1, borderRadius: 12,
    paddingVertical: 9, paddingHorizontal: 12,
    flexDirection: "row", alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.06)",
  },
  speakingDot:  { marginRight: 8, justifyContent: "center", alignItems: "center" },
  speakingPulse:{ width: 9, height: 9, borderRadius: 5, opacity: 0.9 },
  instrText:    { flex: 1, fontSize: 12.5, fontWeight: "700" },
  speakBtn:     { paddingLeft: 8 },

  hintBox: { backgroundColor: "rgba(255,255,255,0.06)", borderRadius: 10, paddingVertical: 5, paddingHorizontal: 10 },
  hintText: { color: "rgba(255,255,255,0.55)", fontSize: 10, textAlign: "center" },

  captureArea: {
    position: "absolute", left: 0, right: 0, bottom: 0,
    paddingVertical: 14, paddingHorizontal: 16,
    alignItems: "center", gap: 6,
    backgroundColor: "rgba(0,0,0,0.65)",
  },
  debugText: { color: "rgba(255,255,255,0.55)", fontSize: 11, fontFamily: "monospace", textAlign: "center" },
  captureBtn: {
    marginTop: 4, paddingVertical: 16, paddingHorizontal: 48,
    borderRadius: 80, borderWidth: 2,
    alignItems: "center", justifyContent: "center",
    minWidth: 200, minHeight: 56,
  },
  captureBtnText: { color: "#fff", fontSize: 18, fontWeight: "bold" },
});