// app/eye-capture.tsx
import React, { useRef, useState, useCallback } from "react";
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
import { useTensorflowModel } from "react-native-fast-tflite";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { Worklets } from "react-native-worklets-core";
import * as ImageManipulator from "expo-image-manipulator";

// ─── Constants ─────────────────────────────────────────────
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

type DebugEyeBox = {
  left: number;
  top: number;
  width: number;
  height: number;
  score: number;
  insideGuide: boolean;
};

function normalizeModelCoordinate(value: number) {
  "worklet";

  return value > 1 ? value / MODEL_SIZE : value;
}

function getEyeBoxOnScreen(
  x1: number,
  y1: number,
  x2: number,
  y2: number,
) {
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

  const { model, state: modelState } = useTensorflowModel(
    require("../assets/model/11-06-2026-yolo-26-n-best_float32.tflite"),
  );
  const { resize } = useResizePlugin();

  const [detected, setDetected] = useState(false);
  const [score, setScore] = useState(0);
  const [frameCount, setFrameCount] = useState(0);
  const [statusMsg, setStatusMsg] = useState("Position eye in the box");
  const [capturing, setCapturing] = useState(false);
  const [debugEyeBox, setDebugEyeBox] = useState<DebugEyeBox | null>(null);

  const cameraReadyRef = useRef(false);
  const capturingRef = useRef(false);
  const consecutiveRef = useRef(0);

  const setDetectedJS = Worklets.createRunOnJS(setDetected);
  const setScoreJS = Worklets.createRunOnJS(setScore);
  const setFrameCountJS = Worklets.createRunOnJS(setFrameCount);
  const setStatusMsgJS = Worklets.createRunOnJS(setStatusMsg);
  const setDebugEyeBoxJS = Worklets.createRunOnJS(setDebugEyeBox);

  // ── Capture & Navigate ─────────────────
  const captureAndNavigate = useCallback(async () => {
    if (capturingRef.current) return;

    const camera = cameraRef.current;

    if (!cameraReadyRef.current || !camera) {
      consecutiveRef.current = 0;
      setDetected(false);
      setFrameCount(0);
      setStatusMsg("Preparing camera...");
      return;
    }

    capturingRef.current = true;
    setCapturing(true);
    setStatusMsg("Capturing high-quality image...");

    try {
      // 1. Take Snapshot
      const photo = await camera.takeSnapshot({ quality: 95 });

      const photoUri = photo.path.startsWith("file://")
        ? photo.path
        : `file://${photo.path}`;

      // 2. Calculate Crop
      const scaleX = photo.width / SW;
      const scaleY = photo.height / SH;

      const cropX = Math.max(0, Math.round(GUIDE_LEFT * scaleX));
      const cropY = Math.max(0, Math.round(GUIDE_TOP * scaleY));
      const cropW = Math.min(
        Math.round(GUIDE_SIZE * scaleX),
        photo.width - cropX,
      );
      const cropH = Math.min(
        Math.round(GUIDE_SIZE * scaleY),
        photo.height - cropY,
      );

      // 3. Crop + Resize to 512x512
      const manipulated = await ImageManipulator.manipulateAsync(
        photoUri,
        [
          {
            crop: {
              originX: cropX,
              originY: cropY,
              width: cropW,
              height: cropH,
            },
          },
          { resize: { width: OUTPUT_SIZE, height: OUTPUT_SIZE } },
        ],
        { compress: 0.92, format: ImageManipulator.SaveFormat.JPEG },
      );

      // 4. Navigate back to Form
      if (!eyeSide) {
        console.error("[EYE] ❌ eyeSide missing!");
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
      console.error("[EYE] ❌ CAPTURE FAILED:", err.message);
      setStatusMsg("Capture failed – try again");
    } finally {
      setCapturing(false);
      capturingRef.current = false;
    }
  }, [router, eyeSide, params]);

  const triggerCaptureJS = Worklets.createRunOnJS(captureAndNavigate);

  // ── Frame Processor ─────────────────
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      if (!model || capturingRef.current) return;

      try {
        const resized = resize(frame, {
          scale: { width: MODEL_SIZE, height: MODEL_SIZE },
          pixelFormat: "rgb",
          dataType: "float32",
          rotation: "270deg",
        });

        const outputs = model.runSync([resized]);
        if (!outputs?.length) return;

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
      } catch {
        // Keep the frame processor alive if a single frame fails.
      }
    },
    [model],
  );

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

  const progressPct = Math.round((frameCount / FRAMES_NEEDED) * 100);

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
        }}
        onError={() => {
          cameraReadyRef.current = false;
          setStatusMsg("Camera error - try again");
        }}
        photo
      />

      {/* UI remains same */}
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
          detected && !capturing && s.guideDetected,
          capturing && s.guideCapturing,
        ]}
      />

      {debugEyeBox && (
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
            {debugEyeBox.insideGuide ? "IN" : "OUT"}{" "}
            {debugEyeBox.score.toFixed(2)}
          </Text>
        </View>
      )}

      <View style={s.topLabel}>
        <Text style={s.eyeTitle}>EYE DETECTION</Text>
        <Text style={s.eyeSub}>
          Centre your {eyeSide.toUpperCase()} eye inside the box
        </Text>
      </View>

      <View style={s.progressTrack}>
        <View style={[s.progressFill, { width: `${progressPct}%` }]} />
      </View>

      <View style={s.hud}>
        {capturing ? (
          <View style={s.row}>
            <ActivityIndicator color="#fff" size="small" />
            <Text style={s.hudText}> Processing image…</Text>
          </View>
        ) : (
          <>
            <Text
              style={[s.statusText, detected ? s.statusGreen : s.statusGray]}
            >
              {statusMsg}
            </Text>
            <View style={s.row}>
              <HudPill label="Side" value={eyeSide.toUpperCase()} />
              <HudPill label="Model" value={modelState} />
              <HudPill label="Score" value={score.toFixed(3)} />
              <HudPill
                label="Frames"
                value={`${frameCount}/${FRAMES_NEEDED}`}
              />
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
  // ... (your existing styles - unchanged)
  root: { flex: 1, backgroundColor: "#000" },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  permText: {
    fontSize: 17,
    color: "#5b2d8e",
    textAlign: "center",
    padding: 24,
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
    borderStyle: "dashed",
    borderColor: "#00e676",
    borderRadius: 6,
  },
  guideDetected: { borderStyle: "solid", borderColor: "#00e676" },
  guideCapturing: { borderStyle: "solid", borderColor: "#ffeb3b" },
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

  topLabel: {
    position: "absolute",
    top: 56,
    left: 0,
    right: 0,
    alignItems: "center",
  },
  eyeTitle: {
    fontSize: 20,
    fontWeight: "900",
    color: "#fff",
    letterSpacing: 3,
  },
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
  progressFill: { height: 4, borderRadius: 2 },

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
  row: {
    flexDirection: "row",
    alignItems: "center",
    justifyContent: "space-around",
  },
  hudText: { color: "#fff", fontSize: 15, fontWeight: "600" },
  statusText: { fontSize: 14, fontWeight: "700", textAlign: "center" },
  statusGreen: { color: "#00e676" },
  statusGray: { color: "rgba(255,255,255,0.55)" },

  pill: { alignItems: "center", gap: 2 },
  pillLabel: {
    fontSize: 10,
    color: "rgba(255,255,255,0.45)",
    textTransform: "uppercase",
    letterSpacing: 0.5,
  },
  pillValue: { fontSize: 13, color: "#fff", fontWeight: "600" },
});
