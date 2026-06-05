// app/eye-capture.tsx
import React, { useEffect, useRef, useState, useCallback } from "react";
import {
  Image, StyleSheet, Text, View, TouchableOpacity,
  ActivityIndicator, Platform,
} from "react-native";
import * as ImageManipulator from "expo-image-manipulator";
import {
  Camera, useCameraDevice, useCameraPermission, useFrameProcessor,
} from "react-native-vision-camera";
import { useTensorflowModel } from "react-native-fast-tflite";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { Worklets } from "react-native-worklets-core";
import { useSharedValue } from "react-native-reanimated";
import Svg, { Path, Circle, Line } from "react-native-svg";
import * as Speech from "expo-speech";

import { useRouter, useLocalSearchParams } from "expo-router";
import { Camera as VisionCamera } from "react-native-vision-camera";

// ─────────────────────────────────────────────────────────────────
//  MODEL CONFIG
// ─────────────────────────────────────────────────────────────────
const MODEL_INPUT_SIZE  = 320;
const NUM_DETECTIONS    = 300;
const VALS_PER_BOX      = 6;
const CONF_THRESHOLD    = 0.25;
const STABLE_FRAMES_ON  = 4;
const STABLE_FRAMES_OFF = 6;
const SMOOTHING         = 0.35;
const MAX_BOX_AREA      = 0.85;
const MIN_BOX_AREA      = 0.05;

const BACK_ROTATION  = "90deg" as const;
const FRONT_ROTATION = "90deg" as const;
const FRONT_MIRROR   = true;

// ─────────────────────────────────────────────────────────────────
//  QUALITY THRESHOLDS
// ─────────────────────────────────────────────────────────────────
const BLUR_THRESHOLD     = 30;
const BRIGHTNESS_LOW     = 65;
const BRIGHTNESS_HIGH    = 205;
const CCT_WARM_THRESHOLD = 3900;
const CCT_WARM_BIAS      = 1.0;
const Q_STEP             = 4;

// Auto-capture delay in ms after all checks go green
const AUTO_CAPTURE_DELAY = 700;

// ─────────────────────────────────────────────────────────────────
//  TYPES
// ─────────────────────────────────────────────────────────────────
type Lang        = "en" | "hi" | "te";
type LightStatus = "good" | "low" | "high";
type BlurStatus  = "good" | "blurry";
type TempStatus  = "good" | "warm";

interface QualityState {
  blurScore:   number;
  blurStatus:  BlurStatus;
  brightness:  number;
  lightStatus: LightStatus;
  cct:         number;
  tempStatus:  TempStatus;
}

// ─────────────────────────────────────────────────────────────────
//  ICONS
// ─────────────────────────────────────────────────────────────────
const IconEye = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path d="M1 12C1 12 5 5 12 5C19 5 23 12 23 12C23 12 19 19 12 19C5 19 1 12 1 12Z" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    <Circle cx="12" cy="12" r="3.5" stroke={color} strokeWidth="2" />
  </Svg>
);

const IconLight = ({ size = 24, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Circle cx="12" cy="12" r="4" stroke={color} strokeWidth="2" />
    <Line x1="12" y1="2"  x2="12" y2="5"  stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="12" y1="19" x2="12" y2="22" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="2"  y1="12" x2="5"  y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="19" y1="12" x2="22" y2="12" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="4.93"  y1="4.93"  x2="7.05"  y2="7.05"  stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="16.95" y1="16.95" x2="19.07" y2="19.07" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="19.07" y1="4.93"  x2="16.95" y2="7.05"  stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Line x1="7.05"  y1="16.95" x2="4.93"  y2="19.07" stroke={color} strokeWidth="2" strokeLinecap="round" />
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
    <Path d="M11 5L6 9H3C2.45 9 2 9.45 2 10V14C2 14.55 2.45 15 3 15H6L11 19V5Z" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    <Path d="M15.54 8.46C16.48 9.4 17 10.67 17 12C17 13.33 16.48 14.6 15.54 15.54" stroke={color} strokeWidth="2" strokeLinecap="round" />
    <Path d="M19.07 4.93C20.96 6.82 22 9.35 22 12C22 14.65 20.96 17.18 19.07 19.07" stroke={color} strokeWidth="2" strokeLinecap="round" />
  </Svg>
);

const IconFlip = ({ size = 18, color = "#fff" }) => (
  <Svg width={size} height={size} viewBox="0 0 24 24" fill="none">
    <Path d="M20 7H4C2.9 7 2 7.9 2 9V18C2 19.1 2.9 20 4 20H20C21.1 20 22 19.1 22 18V9C22 7.9 21.1 7 20 7Z" stroke={color} strokeWidth="2" strokeLinejoin="round" />
    <Circle cx="12" cy="13.5" r="3.5" stroke={color} strokeWidth="2" />
    <Path d="M9 4L12 1L15 4" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
  </Svg>
);

// ─────────────────────────────────────────────────────────────────
//  VOICE GUIDANCE
// ─────────────────────────────────────────────────────────────────
const VOICE: Record<string, Record<Lang, { text: string; bcp47: string }>> = {
  alignFace: {
    en: { text: "Pull down your lower eyelid and fill the box with the pink area.", bcp47: "en-IN" },
    hi: { text: "निचली पलक को नीचे खींचें और गुलाबी हिस्से से बॉक्स भरें।", bcp47: "hi-IN" },
    te: { text: "కింది రెప్పను కిందకు లాగి, గులాబీ భాగంతో బాక్స్ నింపండి.", bcp47: "te-IN" },
  },
  lowLight: {
    en: { text: "Move to a brighter place.", bcp47: "en-IN" },
    hi: { text: "रोशनी वाली जगह पर जाएं।", bcp47: "hi-IN" },
    te: { text: "వెలుతురు ఉన్న చోటికి వెళ్ళండి.", bcp47: "te-IN" },
  },
  highLight: {
    en: { text: "Too much light. Move away from window.", bcp47: "en-IN" },
    hi: { text: "बहुत रोशनी है। खिड़की से दूर जाएं।", bcp47: "hi-IN" },
    te: { text: "చాలా వెలుతురు. కిటికీ నుండి దూరంగా వెళ్ళండి.", bcp47: "te-IN" },
  },
  holdStill: {
    en: { text: "Hold your phone steady.", bcp47: "en-IN" },
    hi: { text: "फोन को स्थिर रखें।", bcp47: "hi-IN" },
    te: { text: "ఫోన్‌ను స్థిరంగా పట్టుకోండి.", bcp47: "te-IN" },
  },
  warmLight: {
    en: { text: "Avoid yellow light. Use neutral white light for better result.", bcp47: "en-IN" },
    hi: { text: "पीली रोशनी से बचें। बेहतर परिणाम के लिए सफेद रोशनी का उपयोग करें।", bcp47: "hi-IN" },
    te: { text: "పసుపు వెలుతురు నుండి దూరంగా ఉండండి. మంచి ఫలితం కోసం తెల్లటి వెలుతురు ఉపయోగించండి.", bcp47: "te-IN" },
  },
  allGood: {
    en: { text: "Good! Tap Capture now.", bcp47: "en-IN" },
    hi: { text: "बढ़िया! अब कैप्चर दबाएं।", bcp47: "hi-IN" },
    te: { text: "బాగుంది! ఇప్పుడు క్యాప్చర్ నొక్కండి.", bcp47: "te-IN" },
  },
};

function getProblemKey(
  detected: boolean,
  lightStatus: string,
  blurStatus: string,
  tempStatus: string,
): string {
  if (!detected)              return "alignFace";
  if (lightStatus === "low")  return "lowLight";
  if (lightStatus === "high") return "highLight";
  if (tempStatus  === "warm") return "warmLight";
  if (blurStatus  === "blurry") return "holdStill";
  return "allGood";
}

// ─────────────────────────────────────────────────────────────────
//  COMPONENT
// ─────────────────────────────────────────────────────────────────
export default function EyeCaptureScreen() {
  const router = useRouter();

  // Params from form screen
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
  const [facing, setFacing] = useState<"front" | "back">(
    Platform.OS === "android" ? "back" : "front"
  );
  const [isDetected, setIsDetected] = useState(false);
  const [debugInfo, setDebugInfo]   = useState("Waiting...");
  const [camReady, setCamReady]     = useState(false);
  const [lang, setLang]             = useState<Lang>("en");
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [capturing, setCapturing]   = useState(false);
  const [quality, setQuality]       = useState<QualityState>({
    blurScore: 0, blurStatus: "blurry",
    brightness: 0, lightStatus: "low",
    cct: 5500, tempStatus: "good",
  });

  // Auto-capture timer ref — cleared whenever conditions stop being "all good"
  const autoCaptureTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  // Track whether auto-capture countdown is running (for UI feedback)
  const [autoCapturePending, setAutoCapturePending] = useState(false);

  const lastSpokenKey  = useRef<string | null>(null);
  const lastSpokenTime = useRef<number>(0);
  const cameraRef      = useRef<any>(null);

  const backDevice  = useCameraDevice("back");
  const frontDevice = useCameraDevice("front");
  const device      = facing === "back" ? backDevice : frontDevice;

  const { model, state } = useTensorflowModel(
    require("../assets/model/conjuctiva.tflite")
      // require("../assets/model/best_float32_conjunctiva.tflite")
  );
  const { resize } = useResizePlugin();

  // Shared values
  const smoothedScore  = useSharedValue(0);
  const stableOnCount  = useSharedValue(0);
  const stableOffCount = useSharedValue(0);
  const confirmedState = useSharedValue(0);
  const isFrontCam     = useSharedValue(0);
  const frameCounter   = useSharedValue(0);
  const bboxX          = useSharedValue(0);
  const bboxY          = useSharedValue(0);
  const bboxW          = useSharedValue(0);
  const bboxH          = useSharedValue(0);
  const bboxValid      = useSharedValue(0);

  useEffect(() => { if (!hasPermission) requestPermission(); }, [hasPermission]);
  useEffect(() => () => { Speech.stop(); }, []);

  useEffect(() => {
    isFrontCam.value     = facing === "front" ? 1 : 0;
    smoothedScore.value  = 0;
    stableOnCount.value  = 0;
    stableOffCount.value = 0;
    confirmedState.value = 0;
    frameCounter.value   = 0;
    setIsDetected(false);
    setCamReady(false);
    setDebugInfo(`Switching to ${facing}...`);
    const delay = facing === "back" ? 1400 : 800;
    const t = setTimeout(() => setCamReady(true), delay);
    return () => clearTimeout(t);
  }, [facing]);

  const setIsDetectedJS = Worklets.createRunOnJS(setIsDetected);
  const setDebugInfoJS  = Worklets.createRunOnJS(setDebugInfo);
  const setQualityJS    = Worklets.createRunOnJS(setQuality);

  const speakNow = useCallback((key: string, l: Lang = lang) => {
    const entry = VOICE[key]?.[l];
    if (!entry) return;
    Speech.stop();
    setIsSpeaking(true);
    Speech.speak(entry.text, {
      language: entry.bcp47, rate: 1.0, pitch: 1.0,
      onDone:  () => setIsSpeaking(false),
      onError: () => setIsSpeaking(false),
    });
  }, [lang]);

  const getImageSize = (uri: string) => new Promise<{ width: number; height: number }>((resolve, reject) => {
    Image.getSize(uri, (width, height) => resolve({ width, height }), reject);
  });

  const cropUriByBoundingBox = async (uri: string) => {
    if (!bboxValid.value) return uri;
    try {
      const { width, height } = await getImageSize(uri);
      const centerX = bboxX.value;
      const centerY = bboxY.value;
      const boxW = bboxW.value;
      const boxH = bboxH.value;
      const originX = Math.max(0, Math.min(1, centerX - boxW / 2));
      const originY = Math.max(0, Math.min(1, centerY - boxH / 2));
      const cropWidth = Math.max(1, Math.min(width, Math.round(boxW * width)));
      const cropHeight = Math.max(1, Math.min(height, Math.round(boxH * height)));
      const cropX = Math.max(0, Math.min(width - cropWidth, Math.round(originX * width)));
      const cropY = Math.max(0, Math.min(height - cropHeight, Math.round(originY * height)));
      const crop = { originX: cropX, originY: cropY, width: cropWidth, height: cropHeight };
      const result = await ImageManipulator.manipulateAsync(uri, [{ crop }], {
        compress: 0.9,
        format: ImageManipulator.SaveFormat.JPEG,
      });
      return result.uri ?? uri;
    } catch (err: any) {
      console.warn("[CROP] failed", err);
      return uri;
    }
  };

  const allGood =
    isDetected &&
    quality.blurStatus  === "good" &&
    quality.lightStatus === "good" &&
    quality.tempStatus  === "good";

  const problemKey = getProblemKey(
    isDetected, quality.lightStatus, quality.blurStatus, quality.tempStatus
  );

  useEffect(() => {
    const now = Date.now();
    if (problemKey !== lastSpokenKey.current || now - lastSpokenTime.current > 8000) {
      lastSpokenKey.current  = problemKey;
      lastSpokenTime.current = now;
      speakNow(problemKey, lang);
    }
  }, [problemKey, lang, speakNow]);

  // ── Auto-capture logic ─────────────────────────────────────────
  // When allGood transitions true → start 500 ms timer → fire capture.
  // If allGood goes false before the timer fires → cancel.
  useEffect(() => {
    if (allGood && !capturing) {
      // Start countdown
      setAutoCapturePending(true);
      autoCaptureTimer.current = setTimeout(() => {
        setAutoCapturePending(false);
        // Re-check allGood inside timeout (capturing ref avoids stale closure)
        handleCapture();
      }, AUTO_CAPTURE_DELAY);
    } else {
      // Cancel any pending countdown
      if (autoCaptureTimer.current) {
        clearTimeout(autoCaptureTimer.current);
        autoCaptureTimer.current = null;
      }
      setAutoCapturePending(false);
    }

    return () => {
      if (autoCaptureTimer.current) {
        clearTimeout(autoCaptureTimer.current);
        autoCaptureTimer.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [allGood, capturing]);

  // ── Capture handler ────────────────────────────────────────────
  const handleCapture = useCallback(async () => {
    if (capturing || !cameraRef.current) return;
    setCapturing(true);
    Speech.stop();

    try {
      const photo = await cameraRef.current.takePhoto({
        qualityPrioritization: "balanced",
        flash: "off",
        enableAutoRedEyeReduction: false,
      });

      const uri = Platform.OS === "android"
        ? `file://${photo.path}`
        : photo.path;

      const croppedUri = await cropUriByBoundingBox(uri);

      // Return to form screen with the cropped eye image + all existing params
      router.replace({
        pathname: "/",
        params: {
          name:          params.name          ?? "",
          parentName:    params.parentName    ?? "",
          phoneNumber:   params.phoneNumber   ?? "",
          age:           params.age           ?? "",
          gender:        params.gender        ?? "",
          eyeSessionId:  params.eyeSessionId  ?? "",
          leftEyeImage:  eyeSide === "left"  ? croppedUri : (params.leftEyeImage  ?? ""),
          rightEyeImage: eyeSide === "right" ? croppedUri : (params.rightEyeImage ?? ""),
        },
      });
    } catch (err: any) {
      console.error("[CAPTURE] error:", err);
      setCapturing(false);
    }
  }, [capturing, cropUriByBoundingBox, eyeSide, params, router]);

  // ── Frame processor ────────────────────────────────────────────
  const frameProcessor = useFrameProcessor((frame) => {
    "worklet";
    frameCounter.value += 1;
    const f       = frameCounter.value;
    const isFront = isFrontCam.value === 1;
    const tag     = isFront ? "FRONT" : "BACK";

    if (model == null || camReady === false) return;

    let rotation: "0deg" | "90deg" | "180deg" | "270deg";
    let mirror: boolean;
    if (isFront) { rotation = FRONT_ROTATION; mirror = FRONT_MIRROR; }
    else         { rotation = BACK_ROTATION;  mirror = false; }

    let resized: any;
    try {
      resized = resize(frame, {
        scale:       { width: MODEL_INPUT_SIZE, height: MODEL_INPUT_SIZE },
        pixelFormat: "rgb",
        dataType:    "float32",
        rotation,
        mirror,
      });
    } catch (e: any) {
      setDebugInfoJS(`[${tag}] resize fail: ${e?.message ?? "?"}`);
      return;
    }

    // Quality checks
    const qd = resized as Float32Array;
    const qW = MODEL_INPUT_SIZE;
    const qH = MODEL_INPUT_SIZE;

    let lumaSum = 0, lumaCount = 0;
    for (let qy = 0; qy < qH; qy += Q_STEP) {
      for (let qx = 0; qx < qW; qx += Q_STEP) {
        const b = (qy * qW + qx) * 3;
        const r = qd[b] * 255, g = qd[b + 1] * 255, bv = qd[b + 2] * 255;
        lumaSum += 0.299 * r + 0.587 * g + 0.114 * bv;
        lumaCount++;
      }
    }
    const brightness = lumaCount > 0 ? lumaSum / lumaCount : 0;
    const lightStatus: "good" | "low" | "high" =
      brightness < BRIGHTNESS_LOW  ? "low" :
      brightness > BRIGHTNESS_HIGH ? "high" : "good";

    const gsW  = Math.floor(qW / Q_STEP);
    const gsH  = Math.floor(qH / Q_STEP);
    const gray = new Float32Array(gsW * gsH);
    for (let sy = 0; sy < gsH; sy++) {
      for (let sx = 0; sx < gsW; sx++) {
        const b = ((sy * Q_STEP) * qW + (sx * Q_STEP)) * 3;
        gray[sy * gsW + sx] =
          0.299 * qd[b] * 255 + 0.587 * qd[b + 1] * 255 + 0.114 * qd[b + 2] * 255;
      }
    }
    let lapSum = 0, lapSumSq = 0, lapCount = 0;
    for (let ly = 1; ly < gsH - 1; ly++) {
      for (let lx = 1; lx < gsW - 1; lx += 2) {
        const idx = ly * gsW + lx;
        const lap =
          gray[(ly - 1) * gsW + lx] + gray[(ly + 1) * gsW + lx] +
          gray[ly * gsW + (lx - 1)] + gray[ly * gsW + (lx + 1)] -
          4 * gray[idx];
        lapSum += lap; lapSumSq += lap * lap; lapCount++;
      }
    }
    const blurMean  = lapCount > 0 ? lapSum / lapCount : 0;
    const blurScore = lapCount > 0 ? lapSumSq / lapCount - blurMean * blurMean : 0;
    const blurStatus: "good" | "blurry" = blurScore >= BLUR_THRESHOLD ? "good" : "blurry";

    let cctSumX = 0, cctSumY = 0, cctSumZ = 0, cctValid = 0;
    for (let cy = 0; cy < qH; cy += Q_STEP) {
      for (let cx = 0; cx < qW; cx += Q_STEP) {
        const b = (cy * qW + cx) * 3;
        const rv = qd[b], gv = qd[b + 1], bv2 = qd[b + 2];
        const r8 = rv * 255, g8 = gv * 255, b8 = bv2 * 255;
        const maxV = r8 > g8 ? (r8 > b8 ? r8 : b8) : (g8 > b8 ? g8 : b8);
        const minV = r8 < g8 ? (r8 < b8 ? r8 : b8) : (g8 < b8 ? g8 : b8);
        const sat  = maxV === 0 ? 0 : (maxV - minV) / maxV;
        if (maxV < 35 || sat > 0.82 || (r8 > 210 && g8 < 100 && b8 < 100)) continue;
        const rl  = rv  <= 0.04045 ? rv  / 12.92 : Math.pow((rv  + 0.055) / 1.055, 2.4);
        const gl2 = gv  <= 0.04045 ? gv  / 12.92 : Math.pow((gv  + 0.055) / 1.055, 2.4);
        const bl2 = bv2 <= 0.04045 ? bv2 / 12.92 : Math.pow((bv2 + 0.055) / 1.055, 2.4);
        cctSumX += 0.4124564 * rl + 0.3575761 * gl2 + 0.1804375 * bl2;
        cctSumY += 0.2126729 * rl + 0.7151522 * gl2 + 0.0721750 * bl2;
        cctSumZ += 0.0193339 * rl + 0.1191920 * gl2 + 0.9503041 * bl2;
        cctValid++;
      }
    }
    let cct = 5500;
    if (cctValid >= 20) {
      const Xv = cctSumX / cctValid, Yv = cctSumY / cctValid, Zv = cctSumZ / cctValid;
      const tot = Xv + Yv + Zv;
      if (tot > 0) {
        const xc = Xv / tot, yc = Yv / tot;
        const nc  = (xc - 0.3320) / (yc - 0.1858);
        const raw = -449 * nc * nc * nc + 3525 * nc * nc - 6823.3 * nc + 5520.33;
        cct = (raw < 2000 ? 2000 : raw > 10000 ? 10000 : raw) * CCT_WARM_BIAS;
      }
    }
    const tempStatus: "good" | "warm" = cct < CCT_WARM_THRESHOLD ? "warm" : "good";

    setQualityJS({ blurScore, blurStatus, brightness, lightStatus, cct, tempStatus });

    let outputs: any;
    try {
      outputs = model.runSync([resized]);
    } catch (e: any) {
      setDebugInfoJS(`[${tag}] infer fail: ${e?.message ?? "?"}`);
      return;
    }
    if (!outputs || outputs.length === 0) return;

    const raw = outputs[0] as Float32Array;
    let bestScore = 0, bestRawScore = 0, bestRawArea = 0;
    let bestBoxScore = 0;
    let bestBoxX = 0, bestBoxY = 0, bestBoxW = 0, bestBoxH = 0;
    for (let i = 0; i < NUM_DETECTIONS; i++) {
      const base  = i * VALS_PER_BOX;
      const score = raw[base + 4];
      if (score < 0.05) continue;
      const x = raw[base + 0];
      const y = raw[base + 1];
      const w = raw[base + 2], h = raw[base + 3];
      if (w <= 0 || h <= 0 || w > 1.5 || h > 1.5) continue;
      const area = w * h;
      if (score > bestRawScore) { bestRawScore = score; bestRawArea = area; }
      if (area < MIN_BOX_AREA || area > MAX_BOX_AREA) continue;
      if (score > bestBoxScore) {
        bestBoxScore = score;
        bestBoxX = x;
        bestBoxY = y;
        bestBoxW = w;
        bestBoxH = h;
      }
      if (score > bestScore) bestScore = score;
    }

    if (confirmedState.value === 1 && bestBoxScore > 0) {
      const normalizedX = isFront ? 1 - bestBoxX : bestBoxX;
      bboxX.value = Math.max(0, Math.min(1, normalizedX));
      bboxY.value = Math.max(0, Math.min(1, bestBoxY));
      bboxW.value = Math.max(0, Math.min(1, bestBoxW));
      bboxH.value = Math.max(0, Math.min(1, bestBoxH));
      bboxValid.value = 1;
    } else {
      bboxValid.value = 0;
    }

    smoothedScore.value = SMOOTHING * bestScore + (1 - SMOOTHING) * smoothedScore.value;
    const isGoodFrame   = smoothedScore.value >= CONF_THRESHOLD;
    if (isGoodFrame) { stableOnCount.value++;  stableOffCount.value = 0; }
    else             { stableOffCount.value++; stableOnCount.value  = 0; }

    let newState = confirmedState.value;
    if (confirmedState.value === 0 && stableOnCount.value  >= STABLE_FRAMES_ON)  newState = 1;
    if (confirmedState.value === 1 && stableOffCount.value >= STABLE_FRAMES_OFF) newState = 0;
    confirmedState.value = newState;

    setIsDetectedJS(newState === 1);
    setDebugInfoJS(
      `[${tag}] score:${bestScore.toFixed(2)} smooth:${smoothedScore.value.toFixed(2)} ` +
      `on:${stableOnCount.value} off:${stableOffCount.value}`
    );
  }, [model, camReady]);

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
    return <View style={s.center}><Text style={s.permText}>No Camera Found</Text></View>;
  if (state === "loading")
    return (
      <View style={s.center}>
        <ActivityIndicator size="large" color="#00c853" />
        <Text style={[s.permText, { marginTop: 14 }]}>Loading model…</Text>
      </View>
    );
  if (state === "error")
    return <View style={s.center}><Text style={[s.permText, { color: "#ff5555" }]}>❌ Model failed to load</Text></View>;

  const instrText  = VOICE[problemKey]?.[lang]?.text ?? "";
  const guideColor = allGood ? "#00ff66" : "#4c9fff";
  const sideLabel  = eyeSide === "left" ? "LEFT EYE" : "RIGHT EYE";

  // ── Capture button label ───────────────────────────────────────
  let captureBtnLabel = "Waiting…";
  if (capturing)           captureBtnLabel = "Capturing…";
  else if (autoCapturePending) captureBtnLabel = "📸 Auto…";
  else if (allGood)        captureBtnLabel = "CAPTURE";

  // ─────────────────────────────────────────────────────────────
  return (
    <View style={s.root}>
      <Camera
        ref={cameraRef}
        key={facing}
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="rgb"
        photo={true}
        onInitialized={() => { console.log(`[CAM] ${facing} initialized ✅`); setCamReady(true); }}
        onError={(e) => {
          console.error(`[CAM ERROR] ${facing}: ${e.message}`);
          setDebugInfo(`Cam error: ${e.message}`);
          setCamReady(false);
          setTimeout(() => setCamReady(true), 1200);
        }}
      />

      {!camReady && (
        <View style={s.switchingOverlay}>
          <ActivityIndicator color="#fff" />
          <Text style={[s.permText, { marginTop: 8 }]}>Switching to {facing}…</Text>
        </View>
      )}

      {/* TOP PANEL */}
      <View style={s.topPanel}>
        {/* Back + Side label + Lang + Flip */}
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
                  onPress={() => setLang(l)}
                >
                  <Text style={[s.radioLabel, lang === l && s.radioLabelActive]}>
                    {l === "en" ? "EN" : l === "hi" ? "हि" : "తె"}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>

            <TouchableOpacity
              style={s.flipBtn}
              onPress={() => { setCamReady(false); setFacing(f => f === "front" ? "back" : "front"); }}
            >
              <IconFlip size={16} color="#fff" />
            </TouchableOpacity>
          </View>
        </View>

        {/* Quality indicators */}
        <View style={s.indicatorRow}>
          {[
            { icon: <IconEye   size={18} color="#fff" />, label: "EYE",   ok: isDetected },
            { icon: <IconLight size={18} color="#fff" />, label: "LIGHT", ok: quality.lightStatus === "good" },
            { icon: <IconBlur  size={18} color="#fff" />, label: "SHARP", ok: quality.blurStatus  === "good" },
            { icon: <IconLight size={18} color="#fff" />, label: "COLOR", ok: quality.tempStatus  === "good" },
          ].map(({ icon, label, ok }) => (
            <View key={label} style={s.indicator}>
              <View style={s.indicatorHeader}>{icon}<Text style={s.indicatorLabel}>{label}</Text></View>
              <View style={[s.indicatorBlock, { backgroundColor: ok ? "#a8e6a3" : "#f4a97f" }]} />
            </View>
          ))}
        </View>

        {/* Instruction */}
        <View style={[s.instrBox, { borderColor: allGood ? "#00e676" : "rgba(255,255,255,0.15)" }]}>
          {isSpeaking && (
            <View style={s.speakingDot}>
              <View style={[s.speakingPulse, { backgroundColor: allGood ? "#00e676" : "#f4a97f" }]} />
            </View>
          )}
          <Text style={[s.instrText, { color: allGood ? "#00e676" : "#fff" }]}>{instrText}</Text>
          <TouchableOpacity style={s.speakBtn} onPress={() => speakNow(problemKey)}>
            <IconSpeaker size={20} color={allGood ? "#00e676" : "#fff"} />
          </TouchableOpacity>
        </View>

        <View style={s.hintBox}>
          <Text style={s.hintText}>📱 Hold 10–15 cm • Pull lower eyelid • Keep full eye inside box</Text>
        </View>
      </View>

      {/* Guide box */}
      <View style={s.cameraWindow}>
        <View style={[s.guide, { borderColor: guideColor }]}>
          {(["TL", "TR", "BL", "BR"] as const).map((c) => (
            <View key={c} style={[s.corner, (s as any)[`corner${c}`], { borderColor: guideColor }]} />
          ))}
          <View style={s.verticalLine} />
          <View style={s.horizontalLine} />
          <View style={[s.centerDot, { borderColor: guideColor }]} />
        </View>
      </View>

      {/* Capture area */}
      <View style={s.captureArea}>
        <Text style={s.debugText}>{debugInfo}</Text>
        <Text style={s.debugText}>
          Blur:{quality.blurScore.toFixed(0)} | Bright:{quality.brightness.toFixed(0)} |{" "}
          {quality.cct.toFixed(0)}K | cam:{facing} | ready:{camReady ? "✅" : "⏳"}
        </Text>

        {/* Auto-capture countdown bar */}
        {autoCapturePending && (
          <View style={s.autoBarTrack}>
            <View style={s.autoBarFill} />
          </View>
        )}

        <TouchableOpacity
          style={[
            s.captureBtn,
            {
              backgroundColor: allGood ? "#00c853" : "#2a2a2a",
              borderColor: allGood ? (autoCapturePending ? "#fff" : "#00ff66") : "#444",
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
            <Text style={[s.captureBtnText, { opacity: allGood ? 1 : 0.4 }]}>
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
  root:   { flex: 1, backgroundColor: "#000" },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: "#000", padding: 24 },
  permText:    { color: "#fff", fontSize: 16, marginBottom: 20, textAlign: "center" },
  permBtn:     { backgroundColor: "#00c853", paddingVertical: 14, paddingHorizontal: 30, borderRadius: 12 },
  permBtnText: { color: "#fff", fontWeight: "700", fontSize: 16 },

  switchingOverlay: {
    ...StyleSheet.absoluteFillObject, backgroundColor: "#000000cc",
    justifyContent: "center", alignItems: "center", zIndex: 99,
  },

  topPanel: {
    paddingTop: 54, paddingHorizontal: 12, paddingBottom: 10,
    backgroundColor: "rgba(0,0,0,0.80)", gap: 8, zIndex: 10,
  },
  controlRow: { flexDirection: "row", alignItems: "center", justifyContent: "space-between" },

  backBtn:     { paddingVertical: 7, paddingHorizontal: 14, borderRadius: 10, backgroundColor: "rgba(255,255,255,0.1)", borderWidth: 1, borderColor: "rgba(255,255,255,0.2)" },
  backBtnText: { color: "#fff", fontSize: 13, fontWeight: "600" },

  sideLabel: { backgroundColor: "rgba(91,45,142,0.8)", paddingVertical: 5, paddingHorizontal: 14, borderRadius: 20 },
  sideLabelText: { color: "#fff", fontWeight: "700", fontSize: 13, letterSpacing: 1 },

  rightControls: { flexDirection: "row", alignItems: "center", gap: 8 },

  radioGroup: {
    flexDirection: "row", borderRadius: 10, overflow: "hidden",
    borderWidth: 1, borderColor: "rgba(255,255,255,0.2)",
  },
  radioBtn: {
    paddingVertical: 7, paddingHorizontal: 12,
    backgroundColor: "rgba(255,255,255,0.07)",
    borderRightWidth: 1, borderRightColor: "rgba(255,255,255,0.15)",
  },
  radioBtnFirst:    { borderTopLeftRadius: 10, borderBottomLeftRadius: 10 },
  radioBtnLast:     { borderTopRightRadius: 10, borderBottomRightRadius: 10, borderRightWidth: 0 },
  radioBtnActive:   { backgroundColor: "rgba(255,215,0,0.18)" },
  radioLabel:       { color: "rgba(255,255,255,0.6)", fontSize: 12, fontWeight: "700" },
  radioLabelActive: { color: "#FFD700" },

  flipBtn: {
    paddingVertical: 7, paddingHorizontal: 10, borderRadius: 10,
    backgroundColor: "rgba(255,255,255,0.1)",
    borderWidth: 1, borderColor: "rgba(255,255,255,0.2)",
  },

  indicatorRow:    { flexDirection: "row", justifyContent: "space-between", gap: 10 },
  indicator:       { flex: 1, alignItems: "flex-start", gap: 6 },
  indicatorHeader: { flexDirection: "row", alignItems: "center", gap: 6 },
  indicatorLabel:  { color: "#fff", fontSize: 13, fontWeight: "800", letterSpacing: 0.5 },
  indicatorBlock:  { width: "100%", height: 22, borderRadius: 10 },

  instrBox: {
    borderWidth: 1, borderRadius: 12, paddingVertical: 11, paddingHorizontal: 14,
    flexDirection: "row", alignItems: "center", backgroundColor: "rgba(255,255,255,0.06)",
  },
  speakingDot:   { marginRight: 10, justifyContent: "center", alignItems: "center" },
  speakingPulse: { width: 10, height: 10, borderRadius: 5, opacity: 0.9 },
  instrText:     { flex: 1, fontSize: 14, fontWeight: "700" },
  speakBtn:      { paddingLeft: 10 },

  hintBox:  { backgroundColor: "rgba(255,255,255,0.06)", borderRadius: 10, paddingVertical: 7, paddingHorizontal: 12 },
  hintText: { color: "rgba(255,255,255,0.55)", fontSize: 11, textAlign: "center" },

  cameraWindow: { flex: 1, justifyContent: "center", alignItems: "center" },
  guide: {
    width: 260, height: 240,
    borderWidth: 2, borderRadius: 20, borderStyle: "dashed",
    justifyContent: "center", alignItems: "center",
  },
  corner:   { position: "absolute", width: 22, height: 22, borderWidth: 3 },
  cornerTL: { top: -2,    left: -2,   borderBottomWidth: 0, borderRightWidth: 0, borderTopLeftRadius: 8 },
  cornerTR: { top: -2,    right: -2,  borderBottomWidth: 0, borderLeftWidth: 0,  borderTopRightRadius: 8 },
  cornerBL: { bottom: -2, left: -2,   borderTopWidth: 0,    borderRightWidth: 0, borderBottomLeftRadius: 8 },
  cornerBR: { bottom: -2, right: -2,  borderTopWidth: 0,    borderLeftWidth: 0,  borderBottomRightRadius: 8 },
  verticalLine:   { position: "absolute", width: 1,  height: "80%", backgroundColor: "rgba(255,255,255,0.2)" },
  horizontalLine: { position: "absolute", height: 1, width:  "80%", backgroundColor: "rgba(255,255,255,0.2)" },
  centerDot: { width: 12, height: 12, borderRadius: 6, borderWidth: 2 },

  captureArea: {
    paddingVertical: 16, alignItems: "center", gap: 6,
    backgroundColor: "rgba(0,0,0,0.65)",
  },

  // Auto-capture countdown bar
  autoBarTrack: {
    width: 200, height: 4, borderRadius: 2,
    backgroundColor: "rgba(255,255,255,0.15)", overflow: "hidden",
  },
  autoBarFill: {
    height: "100%", borderRadius: 2,
    backgroundColor: "#00ff66",
    // CSS animation not available in RN — the bar is shown as a static "ready" indicator.
    // For a true animated fill you can swap this for an Animated.View with useEffect.
    width: "100%",
  },

  captureBtn:     { paddingVertical: 18, paddingHorizontal: 56, borderRadius: 80, borderWidth: 2, alignItems: "center", justifyContent: "center", minWidth: 200, minHeight: 60 },
  captureBtnText: { color: "#fff", fontSize: 20, fontWeight: "bold" },
  debugText:      { color: "rgba(255,255,255,0.35)", fontSize: 10, fontFamily: "monospace" },
} as any);