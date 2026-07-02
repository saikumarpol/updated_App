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
import { useTensorflowModel } from "react-native-fast-tflite";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { Worklets } from "react-native-worklets-core";
import * as ImageManipulator from "expo-image-manipulator";
import * as Speech from "expo-speech";
import Svg, { Path, Circle, Line } from "react-native-svg";

const TAG = `[EYE:${Platform.OS}]`;
const LOG_EVERY_N_FRAMES = 15;

const MODEL_SIZE = 320;
const NUM_DETECTIONS = 300;
const VALS_PER_BOX = 6;
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

const BLUR_THRESHOLD = 30;
const BRIGHTNESS_LOW = 65;
const BRIGHTNESS_HIGH = 205;
const CCT_WARM_THRESHOLD = 3900;
const CCT_WARM_BIAS = 1.0;
const Q_STEP = 4;

type DebugEyeBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  score: number;
  insideGuide: boolean;
};

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

function normalizeModelCoordinate(value: number) {
  "worklet";
  return value > 1 ? value / MODEL_SIZE : value;
}

function getEyeBoxOnScreen(x1: number, y1: number, x2: number, y2: number) {
  "worklet";
  const left = normalizeModelCoordinate(Math.min(x1, x2)) * SW;
  const top = normalizeModelCoordinate(Math.min(y1, y2)) * SH;
  const right = normalizeModelCoordinate(Math.max(x1, x2)) * SW;
  const bottom = normalizeModelCoordinate(Math.max(y1, y2)) * SH;
  return {
    left,
    top,
    right,
    bottom,
    width: right - left,
    height: bottom - top,
  };
}

function checkEyeWithinGuideBox(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
) {
  "worklet";
  const box = getEyeBoxOnScreen(x1, y1, x2, y2);
  const innerLeft = GUIDE_LEFT + GUIDE_INNER_MARGIN;
  const innerTop = GUIDE_TOP + GUIDE_INNER_MARGIN;
  const innerRight = GUIDE_RIGHT - GUIDE_INNER_MARGIN;
  const innerBottom = GUIDE_BOTTOM - GUIDE_INNER_MARGIN;
  return (
    box.width > 0 &&
    box.height > 0 &&
    box.left >= innerLeft &&
    box.top >= innerTop &&
    box.right <= innerRight &&
    box.bottom <= innerBottom
  );
}

function getResizeRotation(
  orientation: string,
): "0deg" | "90deg" | "180deg" | "270deg" {
  "worklet";
  if (Platform.OS === "android") {
    return "270deg";
  }
  switch (orientation) {
    case "landscape-right":
      return "90deg";
    case "landscape-left":
      return "270deg";
    case "portrait-upside-down":
      return "180deg";
    case "portrait":
    default:
      return "0deg";
  }
}

function computeFrameQuality(
  qd: Float32Array,
  qW: number,
  qH: number,
): QualityState {
  "worklet";

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
        0.299 * qd[b] * 255 + 0.587 * qd[b + 1] * 255 + 0.114 * qd[b + 2] * 255;
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

  return { blurScore, blurStatus, brightness, lightStatus, cct, tempStatus };
}

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
    <Line x1="12" y1="2" x2="12" y2="5" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="12" y1="19" x2="12" y2="22" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="2" y1="12" x2="5" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="19" y1="12" x2="22" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="4.93" y1="4.93" x2="7.05" y2="7.05" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="16.95" y1="16.95" x2="19.07" y2="19.07" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="19.07" y1="4.93" x2="16.95" y2="7.05" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="7.05" y1="16.95" x2="4.93" y2="19.07" stroke={color} strokeWidth="2" strokeLinecap="round" />
  </Svg>
);

const IconBlur = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Circle cx="12" cy="12" r="10" stroke={color} strokeWidth="2" />
    <Path d="M12 2C12 2 15 7 14 12C13 17 12 22 12 22" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
    <Path d="M2 12C2 12 7 9 12 10C17 11 22 12 22 12" stroke={color} strokeWidth="1.5" strokeLinecap="round" />
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

  const { model, state: modelState } = useTensorflowModel(
    require("../assets/model/11-06-2026-yolo-26-n-best_float32.tflite"),
  );
  const { resize } = useResizePlugin();

  const [detected, setDetected] = useState(false);
  const [score, setScore] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [statusMsg, setStatusMsg] = useState("Position eye in the box");
  const [capturing, setCapturing] = useState(false);
  const [debugging, setDebugging] = useState(false);
  const [debugEyeBox, setDebugEyeBox] = useState<DebugEyeBox | null>(null);


  const [quality, setQuality] = useState<QualityState>({
    blurScore: 0,
    blurStatus: "blurry",
    brightness: 0,
    lightStatus: "low",
    cct: 5500,
    tempStatus: "good",
  });

  const [lang, setLang] = useState<Lang>("en");
  const [isSpeaking, setIsSpeaking] = useState(false);

  const cameraReadyRef = useRef(false);
  const capturingRef = useRef(false);
  const consecutiveRef = useRef(0);
  const frameLogCounterRef = useRef(0);

  const lastSpokenKey = useRef<string | null>(null);
  const lastSpokenTime = useRef<number>(0);

  const setDetectedJS = Worklets.createRunOnJS(setDetected);
  const setScoreJS = Worklets.createRunOnJS(setScore);
  const setFrameCountJS = Worklets.createRunOnJS(setFrameCount);
  const setStatusMsgJS = Worklets.createRunOnJS(setStatusMsg);
  const setDebugEyeBoxJS = Worklets.createRunOnJS(setDebugEyeBox);
  const setQualityJS = Worklets.createRunOnJS(setQuality);

  useEffect(() => () => Speech.stop(), []);

  useEffect(() => {
    console.log(
      `${TAG} mounted. Platform.Version=${Platform.Version} hasPermission=${hasPermission}`,
    );
  }, []);

  useEffect(() => {
    console.log(`${TAG} hasPermission -> ${hasPermission}`);
  }, [hasPermission]);

  useEffect(() => {
    console.log(
      `${TAG} device -> ${
        device
          ? `id=${device.id} position=${device.position} formats=${device.formats?.length ?? 0}`
          : "null"
      }`,
    );
  }, [device]);

  useEffect(() => {
    console.log(`${TAG} modelState -> ${modelState}`);
    if (modelState === "error") {
      console.error(`${TAG} ❌ MODEL FAILED TO LOAD on this platform`);
    }
  }, [modelState]);

  // ── ONLY CHANGE: router.replace now sends eyeImage instead of left/rightEyeImage ──
// ── Updated captureAndNavigate ─────────────────────────────────────
const captureAndNavigate = useCallback(async () => {
  if (capturingRef.current) return;

  const camera = cameraRef.current;
  if (!cameraReadyRef.current || !camera) {
    console.warn(`${TAG} ⚠️ capture blocked`);
    return;
  }

  capturingRef.current = true;
  setCapturing(true);
  setStatusMsg("Capturing high-quality image...");

  try {
    const photo = await camera.takeSnapshot({ quality: 95 });
    const photoUri = photo.path.startsWith("file://") 
      ? photo.path 
      : `file://${photo.path}`;

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

    console.log(`${TAG} ✅ Captured ${eyeSide} eye: ${manipulated.uri}`);

    // === IMPORTANT FIX: Send BOTH eye images ===
    router.replace({
      pathname: "/",
      params: {
        eyeImage: manipulated.uri,
        eyeSide: eyeSide,

        // Preserve the other eye image
        leftEyeImage: eyeSide === "left" 
          ? manipulated.uri 
          : (params.leftEyeImage ?? ""),

        rightEyeImage: eyeSide === "right" 
          ? manipulated.uri 
          : (params.rightEyeImage ?? ""),

        name: params.name,
        parentName: params.parentName,
        phoneNumber: params.phoneNumber,
        age: params.age,
        gender: params.gender,
        eyeSessionId: params.eyeSessionId,
      },
    });
  } catch (err: any) {
    console.error(`${TAG} ❌ CAPTURE FAILED:`, err);
    setStatusMsg("Capture failed – try again");
  } finally {
    setCapturing(false);
    capturingRef.current = false;
  }
}, [router, eyeSide, params]);
  const triggerCaptureJS = Worklets.createRunOnJS(captureAndNavigate);

  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      frameLogCounterRef.current += 1;
      const shouldLog = frameLogCounterRef.current % LOG_EVERY_N_FRAMES === 0;

      if (!model) {
        if (shouldLog) console.log(`${TAG} ⏳ model not ready yet`);
        return;
      }
      if (capturingRef.current) return;

      const resizeRotation = getResizeRotation(frame.orientation);

      if (shouldLog) {
        console.log(
          `${TAG} frame w=${frame.width} h=${frame.height} orientation=${frame.orientation} mirrored=${frame.isMirrored} pixelFormat=${frame.pixelFormat} rotation=${resizeRotation}`,
        );
      }

      try {
        let resized: Float32Array;
        try {
          resized = resize(frame, {
            scale: { width: MODEL_SIZE, height: MODEL_SIZE },
            pixelFormat: "rgb",
            dataType: "float32",
            rotation: resizeRotation,
          }) as Float32Array;
        } catch (resizeErr: any) {
          console.error(
            `${TAG} ❌ RESIZE FAILED:`,
            resizeErr?.message ?? String(resizeErr),
          );
          return;
        }

        if (shouldLog) {
          console.log(
            `${TAG} resized len=${resized?.length} sample=[${resized?.[0]?.toFixed?.(3)},${resized?.[1]?.toFixed?.(3)},${resized?.[2]?.toFixed?.(3)}]`,
          );
        }

        try {
          const q = computeFrameQuality(resized, MODEL_SIZE, MODEL_SIZE);
          setQualityJS(q);
          if (shouldLog) {
            console.log(
              `${TAG} quality blur=${q.blurScore.toFixed(1)}(${q.blurStatus}) bright=${q.brightness.toFixed(1)}(${q.lightStatus}) cct=${q.cct.toFixed(0)}(${q.tempStatus})`,
            );
          }
        } catch (qErr: any) {
          console.error(
            `${TAG} ❌ QUALITY CALC FAILED:`,
            qErr?.message ?? String(qErr),
          );
        }

        let outputs: any;
        try {
          outputs = model.runSync([resized]);
        } catch (modelErr: any) {
          console.error(
            `${TAG} ❌ MODEL RUNSYNC FAILED:`,
            modelErr?.message ?? String(modelErr),
          );
          return;
        }

        if (!outputs?.length) {
          if (shouldLog) console.warn(`${TAG} ⚠️ model returned no outputs`);
          return;
        }

        const raw = outputs[0] as Float32Array;
        let bestScore = 0;
        let bestX1 = 0;
        let bestY1 = 0;
        let bestX2 = 0;
        let bestY2 = 0;

        for (let i = 0; i < NUM_DETECTIONS; i++) {
          const offset = i * VALS_PER_BOX;
          const conf = raw[offset + 4];

          if (conf > bestScore) {
            bestScore = conf;
            bestX1 = raw[offset];
            bestY1 = raw[offset + 1];
            bestX2 = raw[offset + 2];
            bestY2 = raw[offset + 3];
          }
        }

        setScoreJS(bestScore);

        const eyeInsideGuide =
          bestScore >= CONF_THRESHOLD &&
          checkEyeWithinGuideBox(bestX1, bestY1, bestX2, bestY2);

        if (shouldLog) {
          console.log(
            `${TAG} bestScore=${bestScore.toFixed(3)} box=[${bestX1.toFixed(1)},${bestY1.toFixed(1)},${bestX2.toFixed(1)},${bestY2.toFixed(1)}] insideGuide=${eyeInsideGuide}`,
          );
        }

        if (bestScore >= CONF_THRESHOLD) {
          const box = getEyeBoxOnScreen(bestX1, bestY1, bestX2, bestY2);
          setDebugEyeBoxJS({
            left: box.left,
            top: box.top,
            width: box.width,
            height: box.height,
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
            console.log(
              `${TAG} 🎯 triggering capture (held ${FRAMES_NEEDED} frames)`,
            );
            consecutiveRef.current = 0;
            triggerCaptureJS();
          } else {
            setStatusMsgJS(
              `Hold still... ${consecutiveRef.current}/${FRAMES_NEEDED}`,
            );
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
      } catch (err: any) {
        console.error(
          `${TAG} ❌ FRAME PROCESSOR CRASHED:`,
          err?.message ?? String(err),
        );
      }
    },
    [model],
  );

  const qualityGood =
    quality.blurStatus === "good" &&
    quality.lightStatus === "good" &&
    quality.tempStatus === "good";
  const allGood = detected && qualityGood;
  const problemKey = getProblemKey(
    detected,
    quality.lightStatus,
    quality.blurStatus,
    quality.tempStatus,
  );
  const instrText = VOICE[problemKey]?.[lang]?.text ?? "";

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
        onDone: () => setIsSpeaking(false),
        onError: () => setIsSpeaking(false),
      });
    },
    [lang],
  );

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

  const handleLangChange = useCallback((l: Lang) => setLang(l), []);

  if (!hasPermission) {
    return (
      <View style={s.center}>
        <Text
          style={s.permText}
          onPress={async () => {
            console.log(`${TAG} requesting camera permission...`);
            const granted = await requestPermission();
            console.log(`${TAG} permission request result=${granted}`);
          }}
        >
          Tap to grant camera permission
        </Text>
      </View>
    );
  }

  if (!device) {
    console.warn(`${TAG} ⚠️ no front camera device found`);
    return (
      <View style={s.center}>
        <Text style={s.label}>No front camera found</Text>
      </View>
    );
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
        <Text style={[s.permText, { color: "#ff5555" }]}>
          ❌ Model failed to load
        </Text>
      </View>
    );
  }

  const progressPct = Math.round((frameCount / FRAMES_NEEDED) * 100);
  const guideColor = capturing
    ? "#ffeb3b"
    : allGood
      ? "#00ff66"
      : detected
        ? "#FFD700"
        : "#4c9fff";

  let captureBtnLabel = "Waiting…";
  if (capturing) captureBtnLabel = "Capturing…";
  else if (allGood) captureBtnLabel = "📸 Auto-capturing…";
  else if (detected && !qualityGood) captureBtnLabel = "Eye ✅ — Fix quality";
  else if (!detected) captureBtnLabel = "Align eye…";

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
          console.log(`${TAG} ✅ camera initialized`);
        }}
        onError={(error) => {
          cameraReadyRef.current = false;
          console.error(
            `${TAG} ❌ camera onError:`,
            error?.message ?? String(error),
            (error as any)?.code ?? "",
          );
          setStatusMsg("Camera error - try again");
        }}
        photo
      />

      <View style={[s.vig, { top: 0, left: 0, right: 0, height: GUIDE_TOP }]} />
      <View
        style={[
          s.vig,
          { top: GUIDE_TOP + GUIDE_SIZE, left: 0, right: 0, bottom: 0 },
        ]}
      />
      <View
        style={[
          s.vig,
          { top: GUIDE_TOP, left: 0, width: GUIDE_LEFT, height: GUIDE_SIZE },
        ]}
      />
      <View
        style={[
          s.vig,
          {
            top: GUIDE_TOP,
            left: GUIDE_LEFT + GUIDE_SIZE,
            right: 0,
            height: GUIDE_SIZE,
          },
        ]}
      />

      <View
        style={[
          s.guide,
          {
            borderColor: guideColor,
            borderStyle: capturing || detected ? "solid" : "dashed",
          },
        ]}
      >
        {(["TL", "TR", "BL", "BR"] as const).map((c) => (
          <View
            key={c}
            style={[s.corner, (s as any)[`corner${c}`], { borderColor: guideColor }]}
          />
        ))}
        <View style={s.verticalLine} />
        <View style={s.horizontalLine} />
        <View style={[s.centerDot, { borderColor: guideColor }]} />
      </View>

      {debugEyeBox && debugging && (
        <View
          pointerEvents="none"
          style={[
            s.debugEyeBox,
            debugEyeBox.insideGuide
              ? s.debugEyeBoxInside
              : s.debugEyeBoxOutside,
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
              debugEyeBox.insideGuide
                ? s.debugEyeBoxLabelInside
                : s.debugEyeBoxLabelOutside,
            ]}
          >
            {debugEyeBox.insideGuide ? "IN" : "OUT"} {debugEyeBox.score.toFixed(2)}
          </Text>
        </View>
      )}

      <View style={s.progressTrack}>
        <View
          style={[
            s.progressFill,
            { width: `${progressPct}%`, backgroundColor: guideColor },
          ]}
        />
      </View>

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
                onPress={() => handleLangChange(l)}
              >
                <Text style={[s.radioLabel, lang === l && s.radioLabelActive]}>
                  {l === "en" ? "EN" : l === "hi" ? "हि" : "తె"}
                </Text>
              </TouchableOpacity>
            ))}
          </View>
        </View>

        <View style={s.indicatorRow}>
          {[
            {
              icon: <IconEye size={18} color="#fff" />,
              label: "EYE",
              ok: detected,
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
                : detected
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
              { color: allGood ? "#00e676" : detected ? "#FFD700" : "#fff" },
            ]}
          >
            {instrText}
          </Text>
          <TouchableOpacity style={s.speakBtn} onPress={() => speakNow(problemKey)}>
            <IconSpeaker size={20} color={allGood ? "#00e676" : "#fff"} />
          </TouchableOpacity>
        </View>

        <View style={s.hintBox}>
          <Text style={s.hintText}>
            📱 Hold 10–15 cm • Pull lower eyelid • Keep full eye inside box
          </Text>
        </View>
      </View>

      <View style={s.captureArea}>
        <Text style={s.debugText}>{statusMsg}</Text>
        <Text style={s.debugText}>
          Blur:{quality.blurScore.toFixed(0)} | Bright:
          {quality.brightness.toFixed(0)} | {quality.cct.toFixed(0)}K | score:
          {score.toFixed(2)}
        </Text>
        <Text style={s.debugText}>
          Frames:{frameCount}/{FRAMES_NEEDED} | model:{modelState} | side:
          {eyeSide}
        </Text>

        <TouchableOpacity
          style={[
            s.captureBtn,
            {
              backgroundColor: allGood
                ? "#00c853"
                : detected
                  ? "#7c4dff"
                  : "#2a2a2a",
              borderColor: allGood
                ? "#00ff66"
                : detected
                  ? "#b39ddb"
                  : "#444",
            },
            capturing && { opacity: 0.6 },
          ]}
          onPress={captureAndNavigate}
          disabled={capturing}
          activeOpacity={0.75}
        >
          {capturing ? (
            <ActivityIndicator color="#fff" />
          ) : (
            <Text style={[s.captureBtnText, { opacity: detected ? 1 : 0.4 }]}>
              {captureBtnLabel}
            </Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

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
    fontSize: 16,
    color: "#fff",
    textAlign: "center",
    marginBottom: 4,
  },
  label: { fontSize: 16, color: "#fff" },

  vig: { position: "absolute", backgroundColor: "rgba(0,0,0,0.58)" },

  guide: {
    position: "absolute",
    left: GUIDE_LEFT,
    top: GUIDE_TOP,
    width: GUIDE_SIZE,
    height: GUIDE_SIZE,
    borderWidth: 2.5,
    borderRadius: 6,
    justifyContent: "center",
    alignItems: "center",
  },
  corner: { position: "absolute", width: 22, height: 22, borderWidth: 3 },
  cornerTL: { top: -2, left: -2, borderBottomWidth: 0, borderRightWidth: 0, borderTopLeftRadius: 8 },
  cornerTR: { top: -2, right: -2, borderBottomWidth: 0, borderLeftWidth: 0, borderTopRightRadius: 8 },
  cornerBL: { bottom: -2, left: -2, borderTopWidth: 0, borderRightWidth: 0, borderBottomLeftRadius: 8 },
  cornerBR: { bottom: -2, right: -2, borderTopWidth: 0, borderLeftWidth: 0, borderBottomRightRadius: 8 },
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

  debugEyeBox: {
    position: "absolute",
    borderWidth: 2,
    borderRadius: 4,
    zIndex: 5,
  },
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
  progressFill: { height: 4, borderRadius: 2 },

  topPanel: {
    position: "absolute",
    top: 0,
    left: 0,
    right: 0,
    paddingTop: 46,
    paddingHorizontal: 12,
    paddingBottom: 8,
    backgroundColor: "rgba(0,0,0,0.80)",
    gap: 6,
    zIndex: 10,
  },
  controlRow: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-between",
  },
  backBtn: {
    paddingVertical: 6,
    paddingHorizontal: 12,
    borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.1)",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  backBtnText: { color: "#fff", fontSize: 12, fontWeight: "600" },

  sideLabel: {
    backgroundColor: "rgba(91,45,142,0.8)",
    paddingVertical: 4,
    paddingHorizontal: 12,
    borderRadius: 20,
  },
  sideLabelText: {
    color: "#fff",
    fontWeight: "700",
    fontSize: 12,
    letterSpacing: 1,
  },

  radioGroup: {
    flexDirection: "row",
    borderRadius: 10,
    overflow: "hidden",
    borderWidth: 1,
    borderColor: "rgba(255,255,255,0.2)",
  },
  radioBtn: {
    paddingVertical: 6,
    paddingHorizontal: 10,
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
  radioLabel: { color: "rgba(255,255,255,0.6)", fontSize: 11, fontWeight: "700" },
  radioLabelActive: { color: "#FFD700" },

  indicatorRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    gap: 8,
  },
  indicator: { flex: 1, alignItems: "flex-start", gap: 4 },
  indicatorHeader: { flexDirection: "row", alignItems: "center", gap: 5 },
  indicatorLabel: {
    color: "#fff",
    fontSize: 11,
    fontWeight: "800",
    letterSpacing: 0.5,
  },
  indicatorBlock: { width: "100%", height: 16, borderRadius: 8 },

  instrBox: {
    borderWidth: 1,
    borderRadius: 12,
    paddingVertical: 9,
    paddingHorizontal: 12,
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(255,255,255,0.06)",
  },
  speakingDot: { marginRight: 8, justifyContent: "center", alignItems: "center" },
  speakingPulse: { width: 9, height: 9, borderRadius: 5, opacity: 0.9 },
  instrText: { flex: 1, fontSize: 12.5, fontWeight: "700" },
  speakBtn: { paddingLeft: 8 },

  hintBox: {
    backgroundColor: "rgba(255,255,255,0.06)",
    borderRadius: 10,
    paddingVertical: 5,
    paddingHorizontal: 10,
  },
  hintText: { color: "rgba(255,255,255,0.55)", fontSize: 10, textAlign: "center" },

  captureArea: {
    position: "absolute",
    left: 0,
    right: 0,
    bottom: 0,
    paddingVertical: 14,
    paddingHorizontal: 16,
    alignItems: "center",
    gap: 6,
    backgroundColor: "rgba(0,0,0,0.65)",
  },
  debugText: {
    color: "rgba(255,255,255,0.55)",
    fontSize: 11,
    fontFamily: "monospace",
    textAlign: "center",
  },
  captureBtn: {
    marginTop: 4,
    paddingVertical: 16,
    paddingHorizontal: 48,
    borderRadius: 80,
    borderWidth: 2,
    alignItems: "center",
    justifyContent: "center",
    minWidth: 200,
    minHeight: 56,
  },
  captureBtnText: { color: "#fff", fontSize: 18, fontWeight: "bold" },
});
