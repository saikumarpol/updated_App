import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  StyleSheet,
  Text,
  View,
  TouchableOpacity,
  ActivityIndicator,
  Platform,
} from "react-native";
import * as ImageManipulator from "expo-image-manipulator";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from "react-native-vision-camera";
import { useTensorflowModel } from "react-native-fast-tflite";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { Worklets } from "react-native-worklets-core";
import { useSharedValue } from "react-native-reanimated";
import Svg, { Path, Circle, Line } from "react-native-svg";
import * as Speech from "expo-speech";

import { useRouter, useLocalSearchParams } from "expo-router";

// ─────────────────────────────────────────────────────────────────
//  MODEL CONFIG
// ─────────────────────────────────────────────────────────────────
const MODEL_INPUT_SIZE = 320;
const GUIDE_BOX_SIZE = 512;
const GUIDE_SCREEN_SIZE = 220;
const NUM_DETECTIONS = 300;
const VALS_PER_BOX = 6;
const CONF_THRESHOLD = 0.25;

const BBOX_MARGIN = 20;
const BBOX_MIN_AREA_RATIO = 0.01;
const BBOX_MAX_AREA_RATIO = 0.25;
const BBOX_DEBUG_LOG = true;
const BEST_OF_FRAMES = 4;

// ─────────────────────────────────────────────────────────────────
//  ROTATION / MIRROR CONFIG
// ─────────────────────────────────────────────────────────────────
type RotationDeg = "0deg" | "90deg" | "180deg" | "270deg";

const ROTATION_CONFIG: Record<
  "ios" | "android",
  { front: RotationDeg; back: RotationDeg }
> = {
  ios: { front: "90deg", back: "90deg" },
  android: { front: "270deg", back: "90deg" },
};

const MIRROR_CONFIG: Record<
  "ios" | "android",
  { front: boolean; back: boolean }
> = {
  ios: { front: true, back: false },
  android: { front: false, back: false },
};

const PLATFORM_KEY: "ios" | "android" =
  Platform.OS === "android" ? "android" : "ios";

// ─────────────────────────────────────────────────────────────────
//  QUALITY THRESHOLDS
// ─────────────────────────────────────────────────────────────────
const BLUR_THRESHOLD = 30;
const BRIGHTNESS_LOW = 65;
const BRIGHTNESS_HIGH = 205;
const CCT_WARM_THRESHOLD = 3900;
const CCT_WARM_BIAS = 1.0;
const Q_STEP = 4;

// ─────────────────────────────────────────────────────────────────
//  TYPES
// ─────────────────────────────────────────────────────────────────
type Lang = "en" | "hi" | "te";
type LightStatus = "good" | "low" | "high";
type BlurStatus = "good" | "blurry";
type TempStatus = "good" | "warm";

interface QualityState {
  blurScore: number;
  blurStatus: BlurStatus;
  brightness: number;
  lightStatus: LightStatus;
  cct: number;
  tempStatus: TempStatus;
}

interface DetectedBox {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  type: "full" | "partial";
  valid: boolean;
  areaRatio: number;
}

// ─────────────────────────────────────────────────────────────────
//  FRAME DIMS REF
// ─────────────────────────────────────────────────────────────────
interface FrameDims {
  rawW: number;
  rawH: number;
  rotation: RotationDeg;
  mirror: boolean;
}

// ─────────────────────────────────────────────────────────────────
//  PERF TRACKER
// ─────────────────────────────────────────────────────────────────
const perfHistory = {
  resize: [] as number[],
  quality: [] as number[],
  infer: [] as number[],
  detect: [] as number[],
  total: [] as number[],
  maxLen: 60,
};

function trackPerf(key: keyof Omit<typeof perfHistory, "maxLen">, ms: number) {
  const arr = perfHistory[key];
  arr.push(ms);
  if (arr.length > perfHistory.maxLen) arr.shift();
}

function avgPerf(key: keyof Omit<typeof perfHistory, "maxLen">) {
  const arr = perfHistory[key];
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}

function p95Perf(key: keyof Omit<typeof perfHistory, "maxLen">) {
  const arr = [...perfHistory[key]].sort((a, b) => a - b);
  if (!arr.length) return 0;
  return arr[Math.floor(arr.length * 0.95)] ?? arr[arr.length - 1];
}

// ─────────────────────────────────────────────────────────────────
//  STRUCTURED LOGGER
// ─────────────────────────────────────────────────────────────────
type LogScope =
  | "app-init"
  | "camera-init"
  | "camera-switch"
  | "camera-initialized"
  | "camera-error"
  | "camera-flip"
  | "model-state"
  | "permission"
  | "frame-perf"
  | "frame-perf-summary"
  | "resize"
  | "resize-diagnostic"
  | "quality-metrics"
  | "quality-change"
  | "bbox"
  | "detection-summary"
  | "eye-confirmed"
  | "voice-guidance"
  | "auto-capture-trigger"
  | "capture-start"
  | "capture-photo-done"
  | "capture-crop-start"
  | "crop-512-input"
  | "crop-512-clamped"
  | "crop-512-done"
  | "crop-512-error"
  | "capture-complete"
  | "capture-error"
  | "capture-aborted"
  | "lang-change"
  | "session-info";

function clog(scope: LogScope, data: Record<string, unknown>) {
  console.log(JSON.stringify({ ts: Date.now(), scope, ...data }));
}

// ─────────────────────────────────────────────────────────────────
//  ICONS
// ─────────────────────────────────────────────────────────────────
const IconEye = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path
      d="M1 12C1 12 5 5 12 5C19 5 23 12 23 12C23 12 19 19 12 19C5 19 1 12 1 12Z"
      stroke={color}
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <Circle cx="12" cy="12" r="3.5" stroke={color} strokeWidth="2" />
  </Svg>
);

const IconLight = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Circle cx="12" cy="12" r="4" stroke={color} strokeWidth="2" />
    <Line
      x1="12"
      y1="2"
      x2="12"
      y2="5"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="12"
      y1="19"
      x2="12"
      y2="22"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="2"
      y1="12"
      x2="5"
      y2="12"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="19"
      y1="12"
      x2="22"
      y2="12"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="4.93"
      y1="4.93"
      x2="7.05"
      y2="7.05"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="16.95"
      y1="16.95"
      x2="19.07"
      y2="19.07"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="19.07"
      y1="4.93"
      x2="16.95"
      y2="7.05"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Line
      x1="7.05"
      y1="16.95"
      x2="4.93"
      y2="19.07"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
  </Svg>
);

const IconBlur = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Circle cx="12" cy="12" r="10" stroke={color} strokeWidth="2" />
    <Path
      d="M12 2C12 2 15 7 14 12C13 17 12 22 12 22"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
    />
    <Path
      d="M2 12C2 12 7 9 12 10C17 11 22 12 22 12"
      stroke={color}
      strokeWidth="1.5"
      strokeLinecap="round"
    />
  </Svg>
);

const IconSpeaker = ({ size = 22, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path
      d="M11 5L6 9H3C2.45 9 2 9.45 2 10V14C2 14.55 2.45 15 3 15H6L11 19V5Z"
      stroke={color}
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <Path
      d="M15.54 8.46C16.48 9.4 17 10.67 17 12C17 13.33 16.48 14.6 15.54 15.54"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
    <Path
      d="M19.07 4.93C20.96 6.82 22 9.35 22 12C22 14.65 20.96 17.18 19.07 19.07"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
    />
  </Svg>
);

const IconFlip = ({ size = 18, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path
      d="M20 7H4C2.9 7 2 7.9 2 9V18C2 19.1 2.9 20 4 20H20C21.1 20 22 19.1 22 18V9C22 7.9 21.1 7 20 7Z"
      stroke={color}
      strokeWidth="2"
      strokeLinejoin="round"
    />
    <Circle cx="12" cy="13.5" r="3.5" stroke={color} strokeWidth="2" />
    <Path
      d="M9 4L12 1L15 4"
      stroke={color}
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </Svg>
);

// ─────────────────────────────────────────────────────────────────
//  VOICE GUIDANCE
// ─────────────────────────────────────────────────────────────────
const VOICE: Record<string, Record<Lang, { text: string; bcp47: string }>> = {
  alignFace: {
    en: {
      text: "Pull down your lower eyelid and fill the box with the pink area.",
      bcp47: "en-IN",
    },
    hi: {
      text: "निचली पलक को नीचे खींचें और गुलाबी हिस्से से बॉक्स भरें।",
      bcp47: "hi-IN",
    },
    te: {
      text: "కింది రెప్పను కిందకు లాగి, గులాబీ భాగంతో బాక్స్ నింపండి.",
      bcp47: "te-IN",
    },
  },
  lowLight: {
    en: { text: "Move to a brighter place.", bcp47: "en-IN" },
    hi: { text: "रोशनी वाली जगह पर जाएं।", bcp47: "hi-IN" },
    te: { text: "వెలుతురు ఉన్న చోటికి వెళ్ళండి.", bcp47: "te-IN" },
  },
  highLight: {
    en: { text: "Too much light. Move away from window.", bcp47: "en-IN" },
    hi: { text: "बहुत रोशनी है। खिड़की से दूर जाएं।", bcp47: "hi-IN" },
    te: {
      text: "చాలా వెలుతురు. కిటికీ నుండి దూరంగా వెళ్ళండి.",
      bcp47: "te-IN",
    },
  },
  holdStill: {
    en: { text: "Hold your phone steady.", bcp47: "en-IN" },
    hi: { text: "फोन को स्थिर रखें।", bcp47: "hi-IN" },
    te: { text: "ఫోన్‌ను స్థిరంగా పట్టుకోండి.", bcp47: "te-IN" },
  },
  warmLight: {
    en: {
      text: "Avoid yellow light. Use neutral white light for better result.",
      bcp47: "en-IN",
    },
    hi: {
      text: "पीली रोशनी से बचें। बेहतर परिणाम के लिए सफेद रोशनी का उपयोग करें।",
      bcp47: "hi-IN",
    },
    te: {
      text: "పసుపు వెలుతురు నుండి దూరంగా ఉండండి. మంచి ఫలితం కోసం తెల్లటి వెలుతురు ఉపయోగించండి.",
      bcp47: "te-IN",
    },
  },
  allGood: {
    en: { text: "Good! Capturing now.", bcp47: "en-IN" },
    hi: { text: "बढ़िया! अभी कैप्चर हो रहा है।", bcp47: "hi-IN" },
    te: { text: "బాగుంది! ఇప్పుడు క్యాప్చర్ అవుతోంది.", bcp47: "te-IN" },
  },
};

function getProblemKey(
  eyeConfirmed: boolean,
  lightStatus: string,
  blurStatus: string,
  tempStatus: string,
): string {
  if (!eyeConfirmed) return "alignFace";
  if (lightStatus === "low") return "lowLight";
  if (lightStatus === "high") return "highLight";
  if (tempStatus === "warm") return "warmLight";
  if (blurStatus === "blurry") return "holdStill";
  return "allGood";
}

// ─────────────────────────────────────────────────────────────────
//  BBOX VALIDATION
// ─────────────────────────────────────────────────────────────────
function validateBBox(
  nx1: number,
  ny1: number,
  nx2: number,
  ny2: number,
  imgW: number,
  imgH: number,
) {
  "worklet";

  const x1 = Math.round(nx1 * imgW);
  const y1 = Math.round(ny1 * imgH);
  const x2 = Math.round(nx2 * imgW);
  const y2 = Math.round(ny2 * imgH);

  const touchesLeft = x1 <= BBOX_MARGIN;
  const touchesTop = y1 <= BBOX_MARGIN;
  const touchesRight = x2 >= imgW - BBOX_MARGIN;
  const touchesBottom = y2 >= imgH - BBOX_MARGIN;

  const isPartial = touchesLeft || touchesTop || touchesRight || touchesBottom;

  if (isPartial) {
    return {
      valid: 0 as const,
      type: "partial" as const,
      area: 0,
      areaRatio: 0,
      x1,
      y1,
      x2,
      y2,
      reason: "touching border",
    };
  }

  const width = x2 - x1;
  const height = y2 - y1;
  const area = width * height;
  const areaRatio = area / (imgW * imgH);
  const areaOk =
    areaRatio >= BBOX_MIN_AREA_RATIO && areaRatio <= BBOX_MAX_AREA_RATIO;

  return {
    valid: (areaOk ? 1 : 0) as 0 | 1,
    type: "full" as const,
    area,
    areaRatio,
    x1,
    y1,
    x2,
    y2,
    reason: areaOk ? "ok" : "area out of range",
  };
}

// ─────────────────────────────────────────────────────────────────
//  CROP UTILITY — FIXED iOS + ANDROID
//
//  iOS:
//    takePhoto() returns a photo already in LOGICAL orientation
//    (EXIF applied by OS). Scale the center crop from logical
//    frame coords → photo pixel coords directly.
//
//  Android:
//    takePhoto() returns the photo in RAW SENSOR orientation
//    (NOT rotated). However, for ANY rotation (0/90/180/270°),
//    the guide box center-crop maps back to a CENTERED square
//    in sensor space (rotation is a rigid transform preserving
//    center and size). So we always crop the center of the photo
//    at GUIDE_BOX_SIZE scaled to actual photo resolution.
// ─────────────────────────────────────────────────────────────────
async function cropGuideBoxTo512(
  uri: string,
  dims: FrameDims,
): Promise<string> {
  const cropStart = Date.now();
  const { rawW, rawH, rotation } = dims;

  clog("capture-crop-start", {
    uri,
    rawW,
    rawH,
    rotation,
    platform: Platform.OS,
  });

  try {
    const { width: imgW, height: imgH } = await new Promise<{
      width: number;
      height: number;
    }>((resolve, reject) =>
      require("react-native").Image.getSize(
        uri,
        (w: number, h: number) => resolve({ width: w, height: h }),
        reject,
      ),
    );

    if (Platform.OS === "android") {
      const is90or270 = rotation === "90deg" || rotation === "270deg";
      const logicalW = is90or270 ? rawH : rawW;
      const logicalH = is90or270 ? rawW : rawH;

      // Scaling factor: how many photo pixels correspond to one logical preview pixel.
      // We use the minimum dimension as a stable reference for portrait-first UI.
      const scale = Math.min(imgW, imgH) / Math.min(logicalW, logicalH);

      // The crop size in photo pixels should match the GUIDE_SCREEN_SIZE (220) seen by the user.
      const cropSize = Math.round(GUIDE_SCREEN_SIZE * scale);

      const originX = Math.round(imgW / 2 - cropSize / 2);
      const originY = Math.round(imgH / 2 - cropSize / 2);

      const safeOriginX = Math.max(0, Math.min(imgW - 1, originX));
      const safeOriginY = Math.max(0, Math.min(imgH - 1, originY));
      const safeW = Math.min(imgW - safeOriginX, Math.max(1, cropSize));
      const safeH = Math.min(imgH - safeOriginY, Math.max(1, cropSize));

      clog("crop-512-input", {
        platform: Platform.OS,
        rawW,
        rawH,
        rotation,
        logicalW,
        logicalH,
        imgW,
        imgH,
        scale: +scale.toFixed(3),
        originX,
        originY,
        cropSize,
      });

      const manipStart = Date.now();
      const result = await ImageManipulator.manipulateAsync(
        uri,
        [
          {
            crop: {
              originX: safeOriginX,
              originY: safeOriginY,
              width: safeW,
              height: safeH,
            },
          },
          { resize: { width: GUIDE_BOX_SIZE, height: GUIDE_BOX_SIZE } },
        ],
        { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
      );
      const manipMs = Date.now() - manipStart;
      const totalMs = Date.now() - cropStart;

      clog("crop-512-done", {
        platform: Platform.OS,
        outputUri: result.uri,
        outputSize: `${GUIDE_BOX_SIZE}x${GUIDE_BOX_SIZE}`,
        manipMs,
        totalCropMs: totalMs,
      });

      return result.uri ?? uri;
    } else {
      let originX: number;
      let originY: number;
      let cropW: number;
      let cropH: number;

      // Photo pixel dimensions match the logical (rotated) frame.
      // Logical frame dims (what the model sees as upright):
      const is90or270 = rotation === "90deg" || rotation === "270deg";
      const logicalW = is90or270 ? rawH : rawW;
      const logicalH = is90or270 ? rawW : rawH;

      // Scale from logical frame → photo pixels
      const scaleX = imgW / logicalW;
      const scaleY = imgH / logicalH;
      const half = GUIDE_BOX_SIZE / 2;

      originX = Math.round((logicalW / 2 - half) * scaleX);
      originY = Math.round((logicalH / 2 - half) * scaleY);
      cropW = Math.round(GUIDE_BOX_SIZE * scaleX);
      cropH = Math.round(GUIDE_BOX_SIZE * scaleY);

      // Clamp to photo bounds
      const safeOriginX = Math.max(0, Math.min(imgW - 1, originX));
      const safeOriginY = Math.max(0, Math.min(imgH - 1, originY));
      const safeW = Math.min(imgW - safeOriginX, Math.max(1, cropW));
      const safeH = Math.min(imgH - safeOriginY, Math.max(1, cropH));

      const manipStart = Date.now();
      const result = await ImageManipulator.manipulateAsync(
        uri,
        [
          {
            crop: {
              originX: safeOriginX,
              originY: safeOriginY,
              width: safeW,
              height: safeH,
            },
          },
          { resize: { width: GUIDE_BOX_SIZE, height: GUIDE_BOX_SIZE } },
        ],
        { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
      );
      const manipMs = Date.now() - manipStart;
      const totalMs = Date.now() - cropStart;

      return result.uri ?? uri;
    }
  } catch (err: any) {
    const totalMs = Date.now() - cropStart;
    clog("crop-512-error", { error: err?.message ?? String(err), totalMs });
    return uri;
  }
}

// ─────────────────────────────────────────────────────────────────
//  COMPONENT
// ─────────────────────────────────────────────────────────────────
export default function EyeCaptureScreen() {
  const router = useRouter();
  const sessionStartRef = useRef(Date.now());

  const params = useLocalSearchParams<{
    name?: string;
    parentName?: string;
    phoneNumber?: string;
    age?: string;
    gender?: string;
    eyeSessionId?: string;
    leftEyeImage?: string;
    rightEyeImage?: string;
    eyeSide?: string;
  }>();

  const eyeSide = (params.eyeSide ?? "left") as "left" | "right";

  const { hasPermission, requestPermission } = useCameraPermission();

  const [facing, setFacing] = useState<"front" | "back">("front");

  const [eyeConfirmed, setEyeConfirmed] = useState(false);
  const [debugInfo, setDebugInfo] = useState("Waiting...");
  const [camReady, setCamReady] = useState(false);
  const [lang, setLang] = useState<Lang>("en");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [capturing, setCapturing] = useState(false);
  const [quality, setQuality] = useState<QualityState>({
    blurScore: 0,
    blurStatus: "blurry",
    brightness: 0,
    lightStatus: "low",
    cct: 5500,
    tempStatus: "good",
  });
  const [detectedBox, setDetectedBox] = useState<DetectedBox | null>(null);

  const prevQualityRef = useRef<QualityState | null>(null);
  const prevProblemKey = useRef<string | null>(null);

  const frameDimsRef = useRef<FrameDims>({
    rawW: 1080,
    rawH: 1920,
    rotation: ROTATION_CONFIG[PLATFORM_KEY]["front"],
    mirror: MIRROR_CONFIG[PLATFORM_KEY]["front"],
  });

  const lastSpokenKey = useRef<string | null>(null);
  const lastSpokenTime = useRef<number>(0);
  const cameraRef = useRef<any>(null);
  const hasFiredCapture = useRef(false);

  const jsFrameCountRef = useRef(0);
  const confirmedFrameRef = useRef<number | null>(null);

  const backDevice = useCameraDevice("back");
  const frontDevice = useCameraDevice("front");
  const device = facing === "back" ? backDevice : frontDevice;

  const { model, state } = useTensorflowModel(
    require("../assets/model/11-06-2026-yolo-26-n-best_float32.tflite"),
  );
  const { resize } = useResizePlugin();

  const frameWindow = useSharedValue<number[]>([]);
  const frameCounter = useSharedValue(0);
  const isFrontCam = useSharedValue(0);
  const eyeConfirmedSV = useSharedValue(0);

  // ── Startup log ───────────────────────────────────────────────
  useEffect(() => {
    clog("app-init", {
      eyeSide,
      platform: Platform.OS,
      defaultCam: facing,
      sessionId: params.eyeSessionId ?? "none",
      patient: params.name ?? "unknown",
      age: params.age ?? "unknown",
      gender: params.gender ?? "unknown",
      modelInput: MODEL_INPUT_SIZE,
      guideBox: GUIDE_BOX_SIZE,
      confThresh: CONF_THRESHOLD,
      bestOf: BEST_OF_FRAMES,
      blurThresh: BLUR_THRESHOLD,
      briLow: BRIGHTNESS_LOW,
      briHigh: BRIGHTNESS_HIGH,
      cctWarm: CCT_WARM_THRESHOLD,
      rotationConfig: ROTATION_CONFIG[PLATFORM_KEY],
      mirrorConfig: MIRROR_CONFIG[PLATFORM_KEY],
    });
  }, []);

  useEffect(() => {
    clog("model-state", { state, eyeSide });
  }, [state]);

  useEffect(() => {
    if (!hasPermission) {
      clog("permission", { status: "requesting" });
      requestPermission().then((granted) => {
        clog("permission", { status: granted ? "granted" : "denied" });
      });
    } else {
      clog("permission", { status: "already-granted" });
    }
  }, [hasPermission]);

  useEffect(
    () => () => {
      const sessionMs = Date.now() - sessionStartRef.current;
      clog("session-info", {
        eyeSide,
        totalSessionMs: sessionMs,
        totalFrames: jsFrameCountRef.current,
        confirmedAtFrame: confirmedFrameRef.current,
        avgFps:
          jsFrameCountRef.current > 0
            ? +(jsFrameCountRef.current / (sessionMs / 1000)).toFixed(1)
            : 0,
      });
      Speech.stop();
    },
    [],
  );

  useEffect(() => {
    const camKey = facing === "front" ? "front" : "back";
    isFrontCam.value = facing === "front" ? 1 : 0;
    frameWindow.value = [];
    frameCounter.value = 0;
    eyeConfirmedSV.value = 0;
    hasFiredCapture.current = false;
    jsFrameCountRef.current = 0;
    confirmedFrameRef.current = null;
    setEyeConfirmed(false);
    setCamReady(false);
    setDetectedBox(null);
    setDebugInfo(`Switching to ${facing}...`);

    frameDimsRef.current = {
      rawW: 1080,
      rawH: 1920,
      rotation: ROTATION_CONFIG[PLATFORM_KEY][camKey],
      mirror: MIRROR_CONFIG[PLATFORM_KEY][camKey],
    };

    clog("camera-switch", {
      to: facing,
      eyeSide,
      device: facing === "back" ? backDevice?.id : frontDevice?.id,
    });

    const delay = facing === "back" ? 1400 : 800;
    const t = setTimeout(() => setCamReady(true), delay);
    return () => clearTimeout(t);
  }, [facing]);

  // ── Worklet → JS bridges ──────────────────────────────────────
  const setEyeConfirmedJS = Worklets.createRunOnJS(setEyeConfirmed);
  const setDebugInfoJS = Worklets.createRunOnJS(setDebugInfo);
  const setQualityJS = Worklets.createRunOnJS(setQuality);
  const setDetectedBoxJS = Worklets.createRunOnJS(setDetectedBox);

  const setFrameDimsJS = Worklets.createRunOnJS(
    (rawW: number, rawH: number, rotation: RotationDeg, mirror: boolean) => {
      frameDimsRef.current = { rawW, rawH, rotation, mirror };
    },
  );

  const trackPerfJS = Worklets.createRunOnJS(
    (
      resize: number,
      quality: number,
      infer: number,
      detect: number,
      total: number,
    ) => {
      trackPerf("resize", resize);
      trackPerf("quality", quality);
      trackPerf("infer", infer);
      trackPerf("detect", detect);
      trackPerf("total", total);
      jsFrameCountRef.current += 1;
    },
  );

  const logPerfSummaryJS = Worklets.createRunOnJS(
    (frame: number, cam: string) => {
      clog("frame-perf-summary", {
        cam,
        frame,
        totalFrames: jsFrameCountRef.current,
        avg: {
          resizeMs: +avgPerf("resize").toFixed(1),
          qualityMs: +avgPerf("quality").toFixed(1),
          inferMs: +avgPerf("infer").toFixed(1),
          detectMs: +avgPerf("detect").toFixed(1),
          totalMs: +avgPerf("total").toFixed(1),
        },
        p95: {
          resizeMs: +p95Perf("resize").toFixed(1),
          qualityMs: +p95Perf("quality").toFixed(1),
          inferMs: +p95Perf("infer").toFixed(1),
          detectMs: +p95Perf("detect").toFixed(1),
          totalMs: +p95Perf("total").toFixed(1),
        },
        estFps:
          jsFrameCountRef.current > 0
            ? +(1000 / (avgPerf("total") || 1)).toFixed(1)
            : 0,
      });
    },
  );

  const logQualityChangeJS = Worklets.createRunOnJS(
    (
      prev: QualityState | null,
      next: QualityState,
      frame: number,
      cam: string,
    ) => {
      const changed =
        !prev ||
        prev.lightStatus !== next.lightStatus ||
        prev.blurStatus !== next.blurStatus ||
        prev.tempStatus !== next.tempStatus;

      if (changed) {
        clog("quality-change", {
          cam,
          frame,
          from: prev
            ? {
                light: prev.lightStatus,
                blur: prev.blurStatus,
                temp: prev.tempStatus,
              }
            : null,
          to: {
            light: next.lightStatus,
            blur: next.blurStatus,
            temp: next.tempStatus,
          },
          metrics: {
            brightness: +next.brightness.toFixed(1),
            blurScore: +next.blurScore.toFixed(1),
            cct: +next.cct.toFixed(0),
          },
        });
        prevQualityRef.current = next;
      }
    },
  );

  const speakNow = useCallback(
    (key: string, l: Lang = lang) => {
      const entry = VOICE[key]?.[l];
      if (!entry) return;

      const now = Date.now();
      clog("voice-guidance", {
        key,
        lang: l,
        text: entry.text,
        bcp47: entry.bcp47,
        triggerTs: now,
      });

      Speech.stop();
      setIsSpeaking(true);
      Speech.speak(entry.text, {
        language: entry.bcp47,
        rate: 1.0,
        pitch: 1.0,
        onDone: () => setIsSpeaking(false),
        onError: () => setIsSpeaking(false),
      });
    },
    [lang],
  );

  const qualityGood =
    quality.blurStatus === "good" &&
    quality.lightStatus === "good" &&
    quality.tempStatus === "good";

  const allGood = eyeConfirmed && qualityGood;
  const problemKey = getProblemKey(
    eyeConfirmed,
    quality.lightStatus,
    quality.blurStatus,
    quality.tempStatus,
  );

  useEffect(() => {
    if (problemKey !== prevProblemKey.current) {
      clog("voice-guidance", {
        event: "key-change",
        from: prevProblemKey.current,
        to: problemKey,
        frame: jsFrameCountRef.current,
        eyeSide,
      });
      prevProblemKey.current = problemKey;
    }
  }, [problemKey]);

  useEffect(() => {
    const now = Date.now();
    if (
      problemKey !== lastSpokenKey.current ||
      now - lastSpokenTime.current > 8000
    ) {
      lastSpokenKey.current = problemKey;
      lastSpokenTime.current = now;
      speakNow(problemKey, lang);
    }
  }, [problemKey, lang, speakNow]);

  const handleLangChange = useCallback(
    (l: Lang) => {
      clog("lang-change", {
        from: lang,
        to: l,
        frame: jsFrameCountRef.current,
      });
      setLang(l);
    },
    [lang],
  );

  // ── AUTO-CAPTURE ──────────────────────────────────────────────
  useEffect(() => {
    if (eyeConfirmed && qualityGood && !capturing && !hasFiredCapture.current) {
      hasFiredCapture.current = true;
      const waitMs = Date.now() - sessionStartRef.current;
      clog("auto-capture-trigger", {
        eyeSide,
        qualityGood,
        totalFrames: jsFrameCountRef.current,
        waitMs,
        confirmedAtFrame: confirmedFrameRef.current,
        quality: {
          blurScore: +quality.blurScore.toFixed(1),
          blurStatus: quality.blurStatus,
          brightness: +quality.brightness.toFixed(1),
          lightStatus: quality.lightStatus,
          cct: +quality.cct.toFixed(0),
          tempStatus: quality.tempStatus,
        },
      });
      handleCapture();
    }
  }, [eyeConfirmed, qualityGood, capturing]);

  // ── Capture handler ───────────────────────────────────────────
  const handleCapture = useCallback(async () => {
    if (capturing || !cameraRef.current) {
      clog("capture-aborted", {
        reason: capturing ? "already-capturing" : "no-camera-ref",
        eyeSide,
      });
      return;
    }
    setCapturing(true);
    Speech.stop();

    const captureStart = Date.now();
    const currentDims = frameDimsRef.current;

    clog("capture-start", {
      eyeSide,
      facing,
      frame: jsFrameCountRef.current,
      rawW: currentDims.rawW,
      rawH: currentDims.rawH,
      rotation: currentDims.rotation,
      mirror: currentDims.mirror,
      sessionMs: captureStart - sessionStartRef.current,
      qualityAtCapture: {
        blurScore: +quality.blurScore.toFixed(1),
        blurStatus: quality.blurStatus,
        brightness: +quality.brightness.toFixed(1),
        lightStatus: quality.lightStatus,
        cct: +quality.cct.toFixed(0),
        tempStatus: quality.tempStatus,
      },
    });

    try {
      const photoStart = Date.now();
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: "balanced",
        flash: "off",
        enableAutoRedEyeReduction: false,
      });
      const photoMs = Date.now() - photoStart;

      const uri =
        Platform.OS === "android" ? `file://${photo.path}` : photo.path;

      clog("capture-photo-done", {
        eyeSide,
        photoMs,
        rawW: currentDims.rawW,
        rawH: currentDims.rawH,
        rotation: currentDims.rotation,
        photoWidth: photo.width,
        photoHeight: photo.height,
        uri,
      });

      const croppedUri = await cropGuideBoxTo512(uri, currentDims);

      const totalMs = Date.now() - captureStart;
      clog("capture-complete", {
        eyeSide,
        totalMs,
        photoMs,
        croppedUri,
        sessionMs: Date.now() - sessionStartRef.current,
        totalFrames: jsFrameCountRef.current,
        avgInferMs: +avgPerf("infer").toFixed(1),
        avgTotalMs: +avgPerf("total").toFixed(1),
        p95InferMs: +p95Perf("infer").toFixed(1),
        p95TotalMs: +p95Perf("total").toFixed(1),
      });

      router.replace({
        pathname: "/",
        params: {
          name: params.name ?? "",
          parentName: params.parentName ?? "",
          phoneNumber: params.phoneNumber ?? "",
          age: params.age ?? "",
          gender: params.gender ?? "",
          eyeSessionId: params.eyeSessionId ?? "",
          leftEyeImage:
            eyeSide === "left" ? croppedUri : (params.leftEyeImage ?? ""),
          rightEyeImage:
            eyeSide === "right" ? croppedUri : (params.rightEyeImage ?? ""),
        },
      });
    } catch (err: any) {
      const totalMs = Date.now() - captureStart;
      clog("capture-error", {
        eyeSide,
        error: err?.message ?? String(err),
        totalMs,
      });
      hasFiredCapture.current = false;
    } finally {
      setCapturing(false);
    }
  }, [capturing, eyeSide, params, router, quality, facing]);

  // ── Frame processor ───────────────────────────────────────────
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      const startTotal = Date.now();
      frameCounter.value += 1;
      const f = frameCounter.value;
      const isFront = isFrontCam.value === 1;
      const tag = isFront ? "FRONT" : "BACK";

      if (model == null || camReady === false) return;

      const platformKey = Platform.OS === "android" ? "android" : "ios";
      const camKey = isFront ? "front" : "back";
      const rotation: "0deg" | "90deg" | "180deg" | "270deg" =
        ROTATION_CONFIG[platformKey][camKey];
      const mirror: boolean = MIRROR_CONFIG[platformKey][camKey];

      const rawFw = frame.width;
      const rawFh = frame.height;

      if (f === 1 || f % 60 === 0) {
        setFrameDimsJS(rawFw, rawFh, rotation, mirror);
      }

      const is90or270 = rotation === "90deg" || rotation === "270deg";
      const fw = is90or270 ? rawFh : rawFw;
      const fh = is90or270 ? rawFw : rawFh;

      if (f === 1) {
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "resize-diagnostic",
            cam: tag,
            platform: platformKey,
            rawW: rawFw,
            rawH: rawFh,
            logicalW: fw,
            logicalH: fh,
            rotation,
            mirror,
            note:
              "If detection fails on Android, cycle ROTATION_CONFIG[android][" +
              camKey +
              "] through 0/90/180/270deg and toggle MIRROR_CONFIG, then rebuild.",
          }),
        );
      }

      const resizeStart = Date.now();
      let resized: any;
      let usedFallbackResize = false;

      resized = resize(frame, {
        scale: { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
        pixelFormat: "rgb",
        dataType: "float32",
        rotation,
        mirror,
      });

      const resizeMs = Date.now() - resizeStart;

      if (f === 1) {
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "resize",
            cam: tag,
            frame: f,
            frameRaw: `${rawFw}x${rawFh}`,
            frameLogical: `${fw}x${fh}`,
            rotation,
            mirror,
            resizeMs,
            usedFallback: false,
          }),
        );
      } else if (usedFallbackResize) {
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "resize",
            cam: tag,
            frame: f,
            event: "fallback-used",
            resizeMs,
          }),
        );
      }

      // ── Step 2: Quality checks ───────────────────────────────────
      const qualityStart = Date.now();
      const qd = resized as Float32Array;
      const qW = MODEL_INPUT_SIZE;
      const qH = MODEL_INPUT_SIZE;

      let lumaSum = 0,
        lumaCount = 0;
      for (let qy = 0; qy < qH; qy += Q_STEP) {
        for (let qx = 0; qx < qW; qx += Q_STEP) {
          const b = (qy * qW + qx) * 3;
          const r = qd[b] * 255,
            g = qd[b + 1] * 255,
            bv = qd[b + 2] * 255;
          lumaSum += 0.299 * r + 0.587 * g + 0.114 * bv;
          lumaCount++;
        }
      }
      const brightness = lumaCount > 0 ? lumaSum / lumaCount : 0;
      const lightStatus: LightStatus =
        brightness < BRIGHTNESS_LOW
          ? "low"
          : brightness > BRIGHTNESS_HIGH
            ? "high"
            : "good";

      const gsW = Math.floor(qW / Q_STEP);
      const gsH = Math.floor(qH / Q_STEP);
      const gray = new Float32Array(gsW * gsH);
      for (let sy = 0; sy < gsH; sy++) {
        for (let sx = 0; sx < gsW; sx++) {
          const b = (sy * Q_STEP * qW + sx * Q_STEP) * 3;
          gray[sy * gsW + sx] =
            0.299 * qd[b] * 255 +
            0.587 * qd[b + 1] * 255 +
            0.114 * qd[b + 2] * 255;
        }
      }
      let lapSum = 0,
        lapSumSq = 0,
        lapCount = 0;
      for (let ly = 1; ly < gsH - 1; ly++) {
        for (let lx = 1; lx < gsW - 1; lx += 2) {
          const idx = ly * gsW + lx;
          const lap =
            gray[(ly - 1) * gsW + lx] +
            gray[(ly + 1) * gsW + lx] +
            gray[ly * gsW + (lx - 1)] +
            gray[ly * gsW + (lx + 1)] -
            4 * gray[idx];
          lapSum += lap;
          lapSumSq += lap * lap;
          lapCount++;
        }
      }
      const blurMean = lapCount > 0 ? lapSum / lapCount : 0;
      const blurScore =
        lapCount > 0 ? lapSumSq / lapCount - blurMean * blurMean : 0;
      const blurStatus: BlurStatus =
        blurScore >= BLUR_THRESHOLD ? "good" : "blurry";

      let cctSumX = 0,
        cctSumY = 0,
        cctSumZ = 0,
        cctValid = 0;
      for (let cy = 0; cy < qH; cy += Q_STEP) {
        for (let cx = 0; cx < qW; cx += Q_STEP) {
          const b = (cy * qW + cx) * 3;
          const rv = qd[b],
            gv = qd[b + 1],
            bv2 = qd[b + 2];
          const r8 = rv * 255,
            g8 = gv * 255,
            b8 = bv2 * 255;
          const maxV = r8 > g8 ? (r8 > b8 ? r8 : b8) : g8 > b8 ? g8 : b8;
          const minV = r8 < g8 ? (r8 < b8 ? r8 : b8) : g8 < b8 ? g8 : b8;
          const sat = maxV === 0 ? 0 : (maxV - minV) / maxV;
          if (maxV < 35 || sat > 0.82 || (r8 > 210 && g8 < 100 && b8 < 100))
            continue;
          const rl =
            rv <= 0.04045 ? rv / 12.92 : Math.pow((rv + 0.055) / 1.055, 2.4);
          const gl2 =
            gv <= 0.04045 ? gv / 12.92 : Math.pow((gv + 0.055) / 1.055, 2.4);
          const bl2 =
            bv2 <= 0.04045 ? bv2 / 12.92 : Math.pow((bv2 + 0.055) / 1.055, 2.4);
          cctSumX += 0.4124564 * rl + 0.3575761 * gl2 + 0.1804375 * bl2;
          cctSumY += 0.2126729 * rl + 0.7151522 * gl2 + 0.072175 * bl2;
          cctSumZ += 0.0193339 * rl + 0.119192 * gl2 + 0.9503041 * bl2;
          cctValid++;
        }
      }
      let cct = 5500;
      if (cctValid >= 20) {
        const Xv = cctSumX / cctValid,
          Yv = cctSumY / cctValid,
          Zv = cctSumZ / cctValid;
        const tot = Xv + Yv + Zv;
        if (tot > 0) {
          const xc = Xv / tot,
            yc = Yv / tot;
          const nc = (xc - 0.332) / (yc - 0.1858);
          const raw =
            -449 * nc * nc * nc + 3525 * nc * nc - 6823.3 * nc + 5520.33;
          cct = (raw < 2000 ? 2000 : raw > 10000 ? 10000 : raw) * CCT_WARM_BIAS;
        }
      }
      const tempStatus: TempStatus = cct < CCT_WARM_THRESHOLD ? "warm" : "good";
      const qualityMs = Date.now() - qualityStart;

      const nextQuality = {
        blurScore,
        blurStatus,
        brightness,
        lightStatus,
        cct,
        tempStatus,
      };
      setQualityJS(nextQuality);
      logQualityChangeJS(prevQualityRef.current, nextQuality, f, tag);

      if (f % 15 === 0) {
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "quality-metrics",
            cam: tag,
            frame: f,
            brightness: +brightness.toFixed(1),
            lightStatus,
            blurScore: +blurScore.toFixed(2),
            blurMean: +blurMean.toFixed(2),
            blurStatus,
            cct: +cct.toFixed(0),
            cctValidPx: cctValid,
            tempStatus,
            qualityMs,
            allGood:
              lightStatus === "good" &&
              blurStatus === "good" &&
              tempStatus === "good",
          }),
        );
      }

      // ── Step 3: Model inference ───────────────────────────────────
      const inferStart = Date.now();
      let outputs: any;
      try {
        outputs = model.runSync([resized]);
      } catch (e: any) {
        setDebugInfoJS(`[${tag}] infer fail: ${e?.message ?? "?"}`);
        return;
      }
      const inferMs = Date.now() - inferStart;

      if (!outputs || outputs.length === 0) return;
      const raw = outputs[0] as Float32Array;

      // ── Step 4: Parse + validate detections ──────────────────────
      const detectStart = Date.now();
      let bestBoxScore = 0;
      let validCount = 0;
      let partialCount = 0;
      let areaFailCount = 0;
      let totalAboveConf = 0;

      let overlayBox: DetectedBox | null = null;
      let overlayScore = -1;

      for (let i = 0; i < NUM_DETECTIONS; i++) {
        const base = i * VALS_PER_BOX;
        const score = raw[base + 4];
        if (score < 0.05) continue;

        const nx1 = raw[base + 0];
        const ny1 = raw[base + 1];
        const nx2 = raw[base + 2];
        const ny2 = raw[base + 3];
        const bw = nx2 - nx1;
        const bh = ny2 - ny1;
        if (bw <= 0 || bh <= 0 || bw > 1.5 || bh > 1.5) continue;

        const bbox = validateBBox(nx1, ny1, nx2, ny2, fw, fh);

        if (score >= CONF_THRESHOLD) {
          totalAboveConf++;
          if (bbox.type === "partial") partialCount++;
          else if (bbox.valid !== 1) areaFailCount++;
        }

        if (BBOX_DEBUG_LOG && score >= CONF_THRESHOLD) {
          if (bbox.type === "full") {
            console.log(
              JSON.stringify({
                ts: Date.now(),
                scope: "bbox",
                cam: tag,
                frame: f,
                x1: bbox.x1,
                y1: bbox.y1,
                x2: bbox.x2,
                y2: bbox.y2,
                w: bbox.x2 - bbox.x1,
                h: bbox.y2 - bbox.y1,
                type: "full",
                area: bbox.area,
                areaPct: +(bbox.areaRatio * 100).toFixed(1),
                areaMin: +(BBOX_MIN_AREA_RATIO * 100).toFixed(1),
                areaMax: +(BBOX_MAX_AREA_RATIO * 100).toFixed(1),
                valid: bbox.valid === 1,
                reason: bbox.reason,
                score: +score.toFixed(3),
              }),
            );
          } else {
            console.log(
              JSON.stringify({
                ts: Date.now(),
                scope: "bbox",
                cam: tag,
                frame: f,
                type: "partial",
                reason: bbox.reason,
                valid: false,
                score: +score.toFixed(3),
              }),
            );
          }
        }

        if (score > overlayScore) {
          overlayScore = score;
          overlayBox = {
            x1: bbox.x1,
            y1: bbox.y1,
            x2: bbox.x2,
            y2: bbox.y2,
            score,
            type: bbox.type,
            valid: bbox.valid === 1,
            areaRatio: bbox.areaRatio,
          };
        }

        if (bbox.valid !== 1) continue;
        validCount++;
        if (score > bestBoxScore) bestBoxScore = score;
      }
      const detectMs = Date.now() - detectStart;

      if (f % 10 === 0 && totalAboveConf > 0) {
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "detection-summary",
            cam: tag,
            frame: f,
            totalAboveConf,
            validCount,
            partialCount,
            areaFailCount,
            bestScore: +bestBoxScore.toFixed(3),
            overlayScore: +overlayScore.toFixed(3),
            confirmed: eyeConfirmedSV.value === 1,
            detectMs,
          }),
        );
      }

      if (overlayScore >= CONF_THRESHOLD) {
        setDetectedBoxJS(overlayBox);
      } else {
        setDetectedBoxJS(null);
      }

      // ── Step 5: Best-of-N window ──────────────────────────────────
      const window = frameWindow.value;
      window.push(bestBoxScore);
      if (window.length > BEST_OF_FRAMES) window.shift();
      frameWindow.value = window;

      const allAbove =
        window.length === BEST_OF_FRAMES &&
        window.every((s) => s >= CONF_THRESHOLD);

      if (allAbove && eyeConfirmedSV.value === 0) {
        eyeConfirmedSV.value = 1;
        setEyeConfirmedJS(true);
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "eye-confirmed",
            cam: tag,
            frame: f,
            windowScores: window.map((s) => +s.toFixed(3)),
            windowMin: +Math.min(...window).toFixed(3),
            windowAvg: +(
              window.reduce((a, b) => a + b, 0) / window.length
            ).toFixed(3),
            inferMs,
            resizeMs,
            quality: {
              brightness: +brightness.toFixed(1),
              lightStatus,
              blurScore: +blurScore.toFixed(1),
              blurStatus,
              cct: +cct.toFixed(0),
              tempStatus,
            },
          }),
        );
      }

      // ── Timing aggregation ────────────────────────────────────────
      const totalMs = Date.now() - startTotal;
      trackPerfJS(resizeMs, qualityMs, inferMs, detectMs, totalMs);

      if (f % 30 === 0) {
        console.log(
          JSON.stringify({
            ts: Date.now(),
            scope: "frame-perf",
            cam: tag,
            frame: f,
            timing: { totalMs, resizeMs, qualityMs, inferMs, detectMs },
            detection: {
              bestScore: +bestBoxScore.toFixed(3),
              validCount,
              partialCount,
              areaFailCount,
              totalAboveConf,
            },
            window: {
              scores: window.map((s) => +s.toFixed(3)),
              size: window.length,
              allAbove,
            },
            confirmed: eyeConfirmedSV.value === 1,
            quality: {
              brightness: +brightness.toFixed(1),
              lightStatus,
              blurScore: +blurScore.toFixed(1),
              blurStatus,
              cct: +cct.toFixed(0),
              tempStatus,
            },
            frameMeta: {
              rawW: rawFw,
              rawH: rawFh,
              logicalW: fw,
              logicalH: fh,
              rotation,
              mirror,
              fallbackResize: usedFallbackResize,
            },
          }),
        );
        logPerfSummaryJS(f, tag);
      }

      if (bestBoxScore >= CONF_THRESHOLD) {
        setDebugInfoJS(
          `[${tag}] score:${bestBoxScore.toFixed(2)} ` +
            `window:[${window.map((s) => s.toFixed(2)).join(",")}] ` +
            `confirmed:${eyeConfirmedSV.value === 1 ? "✅" : "⏳"}`,
        );
      } else if (f % 15 === 0) {
        setDebugInfoJS(
          `[${tag}] no det score:${bestBoxScore.toFixed(2)} ` +
            `window:[${window.map((s) => s.toFixed(2)).join(",")}] ` +
            `confirmed:${eyeConfirmedSV.value === 1 ? "✅" : "⏳"}`,
        );
      }
    },
    [model, camReady],
  );

  // ── Guards ─────────────────────────────────────────────────────
  if (!hasPermission)
    return (
      <View style={s.center}>
        <Text style={s.permText}>No Camera Permission</Text>
        <TouchableOpacity style={s.permBtn} onPress={requestPermission}>
          <Text style={s.permBtnText}>Grant Permission</Text>
        </TouchableOpacity>
      </View>
    );
  if (device == null)
    return (
      <View style={s.center}>
        <Text style={s.permText}>No Camera Found</Text>
      </View>
    );
  if (state === "loading")
    return (
      <View style={s.center}>
        <ActivityIndicator size="large" color="#00c853" />
        <Text style={[s.permText, { marginTop: 14 }]}>Loading model…</Text>
      </View>
    );
  if (state === "error")
    return (
      <View style={s.center}>
        <Text style={[s.permText, { color: "#ff5555" }]}>
          ❌ Model failed to load
        </Text>
      </View>
    );

  const instrText = VOICE[problemKey]?.[lang]?.text ?? "";
  const guideColor = allGood ? "#00ff66" : eyeConfirmed ? "#FFD700" : "#4c9fff";
  const sideLabel = eyeSide === "left" ? "LEFT EYE" : "RIGHT EYE";

  let captureBtnLabel = "Waiting…";
  if (capturing) captureBtnLabel = "Capturing…";
  else if (allGood) captureBtnLabel = "📸 Auto-capturing…";
  else if (eyeConfirmed && !qualityGood)
    captureBtnLabel = "Eye ✅ — Fix quality";
  else if (!eyeConfirmed) captureBtnLabel = "Align eye…";

  return (
    <View style={s.root}>
      <Camera
        ref={cameraRef}
        key={facing}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        photo={true}
        onInitialized={() => {
          clog("camera-initialized", {
            facing,
            deviceId: device?.id,
            deviceName: device?.name,
            hasFlash: device?.hasFlash,
            minZoom: device?.minZoom,
            maxZoom: device?.maxZoom,
            formats: device?.formats?.length ?? 0,
            initMs: Date.now() - sessionStartRef.current,
          });
          setCamReady(true);
        }}
        onError={(e) => {
          clog("camera-error", {
            facing,
            error: e.message,
            code: (e as any).code,
          });
          setDebugInfo(`Cam error: ${e.message}`);
          setCamReady(false);
          setTimeout(() => setCamReady(true), 1200);
        }}
      />

      {!camReady && (
        <View style={s.switchingOverlay}>
          <ActivityIndicator color="#fff" />
          <Text style={[s.permText, { marginTop: 8 }]}>
            Switching to {facing}…
          </Text>
        </View>
      )}

      {/* TOP PANEL */}
      <View style={s.topPanel}>
        <View style={s.controlRow}>
          <TouchableOpacity style={s.backBtn} onPress={() => router.back()}>
            <Text style={s.backBtnText}>← Back</Text>
          </TouchableOpacity>

          <View style={s.sideLabel}>
            <Text style={s.sideLabelText}>{sideLabel}</Text>
          </View>

          <View style={s.rightControls}>
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
                  onPress={() => handleLangChange(l)}
                >
                  <Text
                    style={[s.radioLabel, lang === l && s.radioLabelActive]}
                  >
                    {l === "en" ? "EN" : l === "hi" ? "हि" : "తె"}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              style={s.flipBtn}
              onPress={() => {
                const next = facing === "front" ? "back" : "front";
                clog("camera-flip", {
                  from: facing,
                  to: next,
                  frame: jsFrameCountRef.current,
                  eyeConfirmed,
                });
                setCamReady(false);
                setFacing((f) => (f === "front" ? "back" : "front"));
              }}
            >
              <IconFlip size={16} color="#fff" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Indicators */}
        <View style={s.indicatorRow}>
          {[
            {
              icon: <IconEye size={18} color="#fff" />,
              label: "EYE",
              ok: eyeConfirmed,
            },
            {
              icon: <IconLight size={18} color="#fff" />,
              label: "LIGHT",
              ok: quality.lightStatus === "good",
            },
            {
              icon: <IconBlur size={18} color="#fff" />,
              label: "SHARP",
              ok: quality.blurStatus === "good",
            },
            {
              icon: <IconLight size={18} color="#fff" />,
              label: "COLOR",
              ok: quality.tempStatus === "good",
            },
          ].map(({ icon, label, ok }) => (
            <View key={label} style={s.indicator}>
              <View style={s.indicatorHeader}>
                {icon}
                <Text style={s.indicatorLabel}>{label}</Text>
              </View>
              <View
                style={[
                  s.indicatorBlock,
                  { backgroundColor: ok ? "#a8e6a3" : "#f4a97f" },
                ]}
              />
            </View>
          ))}
        </View>

        <View
          style={[
            s.instrBox,
            {
              borderColor: allGood
                ? "#00e676"
                : eyeConfirmed
                  ? "#FFD700"
                  : "rgba(255,255,255,0.15)",
            },
          ]}
        >
          {isSpeaking && (
            <View style={s.speakingDot}>
              <View
                style={[
                  s.speakingPulse,
                  { backgroundColor: allGood ? "#00e676" : "#f4a97f" },
                ]}
              />
            </View>
          )}
          <Text
            style={[
              s.instrText,
              {
                color: allGood ? "#00e676" : eyeConfirmed ? "#FFD700" : "#fff",
              },
            ]}
          >
            {instrText}
          </Text>
          <TouchableOpacity
            style={s.speakBtn}
            onPress={() => speakNow(problemKey)}
          >
            <IconSpeaker size={20} color={allGood ? "#00e676" : "#fff"} />
          </TouchableOpacity>
        </View>

        <View style={s.hintBox}>
          <Text style={s.hintText}>
            📱 Hold 10–15 cm • Pull lower eyelid • Keep full eye inside box
          </Text>
        </View>
      </View>

      {/* CAMERA WINDOW */}
      <View style={s.cameraWindow}>
        <View
          style={[
            s.guide,
            {
              borderColor: guideColor,
              width: GUIDE_SCREEN_SIZE,
              height: GUIDE_SCREEN_SIZE,
            },
          ]}
        >
          {(["TL", "TR", "BL", "BR"] as const).map((c) => (
            <View
              key={c}
              style={[
                s.corner,
                (s as any)[`corner${c}`],
                { borderColor: guideColor },
              ]}
            />
          ))}
          <View style={s.verticalLine} />
          <View style={s.horizontalLine} />
          <View style={[s.centerDot, { borderColor: guideColor }]} />
        </View>
        <Text style={s.guideLabel}>512 × 512</Text>
        {detectedBox && (
          <Text
            style={[
              s.guideLabel,
              {
                color: detectedBox.valid
                  ? "#00e676"
                  : detectedBox.type === "partial"
                    ? "#ff5252"
                    : "#ffb74d",
              },
            ]}
          >
            {detectedBox.type === "partial"
              ? "partial — touching border"
              : `${detectedBox.valid ? "valid" : "full — area out of range"} · area ${(detectedBox.areaRatio * 100).toFixed(1)}% · score ${detectedBox.score.toFixed(2)}`}
          </Text>
        )}
      </View>

      {/* BOTTOM AREA */}
      <View style={s.captureArea}>
        <Text style={s.debugText}>{debugInfo}</Text>
        <Text style={s.debugText}>
          Blur:{quality.blurScore.toFixed(0)} | Bright:
          {quality.brightness.toFixed(0)} | {quality.cct.toFixed(0)}K | cam:
          {facing} | ready:{camReady ? "✅" : "⏳"} | eye:
          {eyeConfirmed ? "✅" : "⏳"}
        </Text>
        <Text style={s.debugText}>
          avg infer:{avgPerf("infer").toFixed(0)}ms | avg total:
          {avgPerf("total").toFixed(0)}ms | p95 total:
          {p95Perf("total").toFixed(0)}ms | frames:{jsFrameCountRef.current}
        </Text>
        <Text style={s.debugText}>
          rawDims:{frameDimsRef.current.rawW}×{frameDimsRef.current.rawH} | rot:
          {frameDimsRef.current.rotation} | mir:
          {frameDimsRef.current.mirror ? "Y" : "N"} | plat:{Platform.OS}
        </Text>

        <TouchableOpacity
          style={[
            s.captureBtn,
            {
              backgroundColor: allGood
                ? "#00c853"
                : eyeConfirmed
                  ? "#7c4dff"
                  : "#2a2a2a",
              borderColor: allGood
                ? "#00ff66"
                : eyeConfirmed
                  ? "#b39ddb"
                  : "#444",
            },
            capturing && { opacity: 0.6 },
          ]}
          onPress={handleCapture}
          disabled={capturing}
          activeOpacity={0.75}
        >
          {capturing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text
              style={[s.captureBtnText, { opacity: eyeConfirmed ? 1 : 0.4 }]}
            >
              {captureBtnLabel}
            </Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

// ─────────────────────────────────────────────────────────────────
//  STYLES
// ─────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#000" },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
    padding: 24,
  },
  permText: {
    color: "#fff",
    fontSize: 16,
    marginBottom: 20,
    textAlign: "center",
  },
  permBtn: {
    backgroundColor: "#00c853",
    paddingVertical: 14,
    paddingHorizontal: 30,
    borderRadius: 12,
  },
  permBtnText: { color: "#fff", fontWeight: "700", fontSize: 16 },

  switchingOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: "#000000cc",
    justifyContent: "center",
    alignItems: "center",
    zIndex: 99,
  },

  topPanel: {
    paddingTop: 54,
    paddingHorizontal: 12,
    paddingBottom: 10,
    backgroundColor: "rgba(0,0,0,0.80)",
    gap: 8,
    zIndex: 10,
  },
  controlRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },

  backBtn: {
    paddingVertical: 7,
    paddingHorizontal: 14,
    borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.1)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  backBtnText: { color: "#fff", fontSize: 13, fontWeight: "600" },

  sideLabel: {
    backgroundColor: "rgba(91,45,142,0.8)",
    paddingVertical: 5,
    paddingHorizontal: 14,
    borderRadius: 20,
  },
  sideLabelText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 13,
    letterSpacing: 1,
  },

  rightControls: { flexDirection: "row", alignItems: "center", gap: 8 },

  radioGroup: {
    flexDirection: "row",
    borderRadius: 10,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  radioBtn: {
    paddingVertical: 7,
    paddingHorizontal: 12,
    backgroundColor: "rgba(255,255,255,0.07)",
    borderRightWidth: 1,
    borderRightColor: "rgba(255,255,255,0.15)",
  },
  radioBtnFirst: { borderTopLeftRadius: 10, borderBottomLeftRadius: 10 },
  radioBtnLast: {
    borderTopRightRadius: 10,
    borderBottomRightRadius: 10,
    borderRightWidth: 0,
  },
  radioBtnActive: { backgroundColor: "rgba(255,215,0,0.18)" },
  radioLabel: {
    color: "rgba(255,255,255,0.6)",
    fontSize: 12,
    fontWeight: "700",
  },
  radioLabelActive: { color: "#FFD700" },

  flipBtn: {
    paddingVertical: 7,
    paddingHorizontal: 10,
    borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.1)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },

  indicatorRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 10,
  },
  indicator: { flex: 1, alignItems: "flex-start", gap: 6 },
  indicatorHeader: { flexDirection: "row", alignItems: "center", gap: 6 },
  indicatorLabel: {
    color: "#fff",
    fontSize: 13,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  indicatorBlock: { width: "100%", height: 22, borderRadius: 10 },

  instrBox: {
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 11,
    paddingHorizontal: 14,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.06)",
  },
  speakingDot: {
    marginRight: 10,
    justifyContent: "center",
    alignItems: "center",
  },
  speakingPulse: { width: 10, height: 10, borderRadius: 5, opacity: 0.9 },
  instrText: { flex: 1, fontSize: 14, fontWeight: "700" },
  speakBtn: { paddingLeft: 10 },

  hintBox: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: 10,
    paddingVertical: 7,
    paddingHorizontal: 12,
  },
  hintText: {
    color: "rgba(255,255,255,0.55)",
    fontSize: 11,
    textAlign: "center",
  },

  cameraWindow: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    gap: 8,
  },

  guide: {
    borderWidth: 2,
    borderRadius: 16,
    borderStyle: "dashed",
    justifyContent: "center",
    alignItems: "center",
  },
  guideLabel: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 11,
    fontFamily: "monospace",
  },

  corner: { position: "absolute", width: 22, height: 22, borderWidth: 3 },
  cornerTL: {
    top: -2,
    left: -2,
    borderBottomWidth: 0,
    borderRightWidth: 0,
    borderTopLeftRadius: 8,
  },
  cornerTR: {
    top: -2,
    right: -2,
    borderBottomWidth: 0,
    borderLeftWidth: 0,
    borderTopRightRadius: 8,
  },
  cornerBL: {
    bottom: -2,
    left: -2,
    borderTopWidth: 0,
    borderRightWidth: 0,
    borderBottomLeftRadius: 8,
  },
  cornerBR: {
    bottom: -2,
    right: -2,
    borderTopWidth: 0,
    borderLeftWidth: 0,
    borderBottomRightRadius: 8,
  },
  verticalLine: {
    position: "absolute",
    width: 1,
    height: "80%",
    backgroundColor: "rgba(255,255,255,0.2)",
  },
  horizontalLine: {
    position: "absolute",
    height: 1,
    width: "80%",
    backgroundColor: "rgba(255,255,255,0.2)",
  },
  centerDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 2 },

  captureArea: {
    paddingVertical: 16,
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(0,0,0,0.65)",
  },

  captureBtn: {
    paddingVertical: 18,
    paddingHorizontal: 56,
    borderRadius: 80,
    borderWidth: 2,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 200,
    minHeight: 60,
  },
  captureBtnText: { color: "#fff", fontSize: 20, fontWeight: "bold" },
  debugText: {
    color: "rgba(255,255,255,0.35)",
    fontSize: 10,
    fontFamily: "monospace",
  },
} as any);
