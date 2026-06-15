// app/index.tsx
import { useState, useEffect, useRef } from "react";
import {
  View, TextInput, ScrollView, Image, Text, Alert,
  StyleSheet, TouchableOpacity, ActivityIndicator,
  Platform, KeyboardAvoidingView, Modal, Dimensions, StatusBar,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";
import axios from "axios";
import { useRouter, useLocalSearchParams } from "expo-router";
import { MaterialCommunityIcons } from "@expo/vector-icons";

// expo-av audio recording (install: npx expo install expo-av)
// If expo-av is not yet installed, the audio section will show a placeholder.
let Audio: any = null;
try {
  Audio = require("expo-av").Audio;
} catch {
  // expo-av not installed — audio consent will be disabled
}

const Icon = MaterialCommunityIcons;

// ─── CONFIG ──────────────────────────────────────────────────────────────────
// const BASE_URL = "http://10.2.136.235:5001";
const BASE_URL = "https://pl-api.iiit.ac.in/rcts/anemiav2/";
const { width: SW } = Dimensions.get("window");

const C = {
  purple: "#5b2d8e",
  purpleLight: "#f3eefe",
  purpleBorder: "#d1c4e9",
  page: "#f7f4fc",
  card: "#fff",
  text: "#1a1a2e",
  muted: "#888",
  hint: "#bbb",
  border: "#ede7f6",
  success: "#2f9e44",
};

// ─────────────────────────────────────────────────────────────────────────────
export default function FormScreen() {
  const router = useRouter();

  const params = useLocalSearchParams<{
    name?: string;
    parentName?: string;
    phoneNumber?: string;
    age?: string;
    gender?: string;
    eyeSessionId?: string;
    leftEyeImage?: string;
    rightEyeImage?: string;
  }>();

  // ── Form state ─────────────────────────────────────────────────
  const [name, setName]               = useState("");
  const [parentName, setParentName]   = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [age, setAge]                 = useState("");
  const [gender, setGender]           = useState("");

  // ── Eye images ─────────────────────────────────────────────────
  const [leftEyeImage, setLeftEyeImage]   = useState<string | null>(null);
  const [rightEyeImage, setRightEyeImage] = useState<string | null>(null);
  const [eyeSessionId, setEyeSessionId]   = useState("");

  // ── Audio consent ──────────────────────────────────────────────
  const [audioUri, setAudioUri]       = useState<string | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const recordingRef                  = useRef<any>(null);
  const audioAvailable                = Audio !== null;

  // ── UI ─────────────────────────────────────────────────────────
  const [loading, setLoading]       = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("Processing...");
  const [previewUri, setPreviewUri] = useState<string | null>(null);

  // Restore form fields when returning from eye-capture
  useEffect(() => {
    if (params.name)          setName(params.name);
    if (params.parentName)    setParentName(params.parentName);
    if (params.phoneNumber)   setPhoneNumber(params.phoneNumber);
    if (params.age)           setAge(params.age);
    if (params.gender)        setGender(params.gender);
    if (params.eyeSessionId)  setEyeSessionId(params.eyeSessionId);
    if (params.leftEyeImage)  setLeftEyeImage(params.leftEyeImage);
    if (params.rightEyeImage) setRightEyeImage(params.rightEyeImage);
  }, [params]);

  // ── Helpers ────────────────────────────────────────────────────
  const isMinor       = age ? parseInt(age) < 18 : false;
  const capturedCount = [leftEyeImage, rightEyeImage].filter(Boolean).length;
  const progress      = Math.min(
    100,
    (capturedCount / 2) * 50 + (name && age && gender ? 50 : 0)
  );

  const resetForm = () => {
    setName(""); setParentName(""); setPhoneNumber("");
    setAge(""); setGender("");
    setLeftEyeImage(null); setRightEyeImage(null); setEyeSessionId("");
    setAudioUri(null);
  };

  const goToCapture = (side: "left" | "right") => {
    router.push({
      pathname: "/eye-capture",
      params: {
        name, parentName, phoneNumber, age, gender, eyeSessionId,
        leftEyeImage:  leftEyeImage  ?? "",
        rightEyeImage: rightEyeImage ?? "",
        eyeSide: side,
      },
    });
  };

  // ── Audio recording ────────────────────────────────────────────
  const startRecording = async () => {
    if (!audioAvailable) {
      Alert.alert(
        "expo-av not installed",
        "Run: npx expo install expo-av\nThen rebuild the app.",
      );
      return;
    }
    try {
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        Alert.alert("Permission Denied", "Microphone permission is required.");
        return;
      }
      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });
      const { recording } = await Audio.Recording.createAsync(
        Audio.RecordingOptionsPresets.HIGH_QUALITY
      );
      recordingRef.current = recording;
      setIsRecording(true);
    } catch (e: any) {
      Alert.alert("Error", `Failed to start recording: ${e?.message ?? ""}`);
    }
  };

  const stopRecording = async () => {
    try {
      if (!recordingRef.current) return;
      await recordingRef.current.stopAndUnloadAsync();
      const uri = recordingRef.current.getURI();
      setAudioUri(uri);
      setIsRecording(false);
    } catch (e: any) {
      Alert.alert("Error", `Failed to stop recording: ${e?.message ?? ""}`);
    }
  };

  // ── Submit ─────────────────────────────────────────────────────
  const submitData = async () => {
    if (!leftEyeImage || !rightEyeImage) {
      Alert.alert("Missing", "Please capture both eye images.");
      return;
    }
    if (!name || !age || !gender) {
      Alert.alert("Missing", "Please fill all required fields (*).");
      return;
    }
    if (isMinor && !parentName) {
      Alert.alert("Missing", "Parent/Guardian name is required for minors.");
      return;
    }
    if (audioAvailable && !audioUri) {
      Alert.alert("Consent Required", "Please record your audio consent.");
      return;
    }

    setLoading(true);
    setLoadingMsg("Uploading images...");

    const formData = new FormData();
    formData.append("name", name);
    formData.append("parentName", parentName || "");
    formData.append("phoneNumber", phoneNumber || "");
    formData.append("age", age);
    formData.append("gender", gender);
    if (eyeSessionId) formData.append("eyeSessionId", eyeSessionId);

    formData.append("leftEyeImage", {
      uri: leftEyeImage,
      type: "image/jpeg",
      name: `left_${Date.now()}.jpg`,
    } as any);
    formData.append("rightEyeImage", {
      uri: rightEyeImage,
      type: "image/jpeg",
      name: `right_${Date.now()}.jpg`,
    } as any);
    if (audioUri) {
      formData.append("audioConsent", {
        uri: audioUri,
        type: "audio/m4a",
        name: `consent_${Date.now()}.m4a`,
      } as any);
    }

    const MESSAGES = [
      "Uploading images…",
      "Analysing left eye…",
      "Analysing right eye…",
      "Finalizing…",
    ];
    let msgIdx = 0;
    const msgTimer = setInterval(() => {
      msgIdx = Math.min(msgIdx + 1, MESSAGES.length - 1);
      setLoadingMsg(MESSAGES[msgIdx]);
    }, 6000);

    try {
      const res = await axios.post(`${BASE_URL}/api/children`, formData, {
        headers: { "Content-Type": "multipart/form-data" },
        timeout: 360000,
      });
      clearInterval(msgTimer);
      const d = res.data;

      Alert.alert(
        "✅ Analysis Complete",
        `Left Eye  → Hb: ${d.leftHb ?? "N/A"} | ${d.leftAnemia ?? "Unknown"}\n\n` +
        `Right Eye → Hb: ${d.rightHb ?? "N/A"} | ${d.rightAnemia ?? "Unknown"}`,
        [{
          text: "View Dashboard",
          onPress: () => { resetForm(); router.replace("/(tabs)/dashboard"); },
        }]
      );
    } catch (error: any) {
      clearInterval(msgTimer);
      Alert.alert("Upload Failed", error.message || "Please check your connection.");
    } finally {
      setLoading(false);
    }
  };

  const PREVIEW_SIZE = Math.min(480, SW - 32);

  // ─────────────────────────────────────────────────────────────────────────
  return (
    <SafeAreaView style={s.safe}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      {/* Header */}
      <View style={s.header}>
        <View style={{ flex: 1 }}>
          <Text style={s.headerTitle}>New Assessment</Text>
          <Text style={s.headerSub}>Fill all required fields</Text>
        </View>
        <TouchableOpacity
          style={s.dashBtn}
          onPress={() => router.push("/(tabs)/dashboard")}
        >
          <Icon name="view-dashboard-outline" size={18} color={C.purple} />
          <Text style={s.dashBtnText}>Dashboard</Text>
        </TouchableOpacity>
      </View>

      {/* Progress */}
      <View style={s.progressTrack}>
        <View style={[s.progressFill, { width: `${progress}%` }]} />
      </View>

      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
      >
        <ScrollView
          style={{ backgroundColor: C.page }}
          contentContainerStyle={s.scroll}
          keyboardShouldPersistTaps="always"
        >
          {/* ── Personal Info ── */}
          <View style={s.card}>
            <Text style={s.sect}>Personal Info</Text>
            <Field label="Name *"         value={name}        onChange={setName}        placeholder="Enter full name" />
            <Field label="Age (years) *"  value={age}         onChange={setAge}         placeholder="Age" keyboard="numeric" />
            <Field label="Phone Number"   value={phoneNumber} onChange={setPhoneNumber} placeholder="10-digit mobile number" keyboard="phone-pad" />
            {isMinor && (
              <Field label="Parent/Guardian Name *" value={parentName} onChange={setParentName} placeholder="Enter parent name" />
            )}
            <Text style={s.fieldLabel}>Gender *</Text>
            <View style={s.genderRow}>
              {(["Male", "Female"] as const).map((g) => (
                <TouchableOpacity
                  key={g}
                  style={[s.genderBtn, gender === g && s.genderBtnActive]}
                  onPress={() => setGender(g)}
                >
                  <Text style={[s.genderText, gender === g && s.genderTextActive]}>
                    {g === "Male" ? "♂ " : "♀ "}{g}
                  </Text>
                </TouchableOpacity>
              ))}
            </View>
          </View>

          {/* ── Eye Images ── */}
          <View style={s.card}>
            <Text style={s.sect}>Capture Eye Images</Text>
            <Text style={s.eyeHint}>Both eyes are required for analysis</Text>
            <View style={s.eyeRow}>
              {([
                { side: "left",  label: "LEFT",  img: leftEyeImage,  color: C.purple },
                { side: "right", label: "RIGHT", img: rightEyeImage, color: "#c2185b" },
              ] as const).map(({ side, label, img, color }) => (
                <View key={side} style={s.eyeCol}>
                  <TouchableOpacity
                    style={[s.imgBtn, img ? s.imgBtnDone : s.imgBtnReady]}
                    onPress={() => goToCapture(side)}
                  >
                    <Icon
                      name={img ? "camera-retake-outline" : "camera-plus-outline"}
                      size={18}
                      color="#fff"
                    />
                    <View>
                      <Text style={s.imgBtnText}>{img ? "Re-capture" : "Capture"}</Text>
                      <Text style={s.imgBtnSub}>{label} Eye</Text>
                    </View>
                  </TouchableOpacity>
                  {img && (
                    <TouchableOpacity onPress={() => setPreviewUri(img)} style={s.thumb}>
                      <Image source={{ uri: img }} style={s.thumbImg} resizeMode="cover" />
                      <View style={[s.thumbBadge, { backgroundColor: color }]}>
                        <Text style={s.thumbBadgeText}>{label[0]}</Text>
                      </View>
                    </TouchableOpacity>
                  )}
                </View>
              ))}
            </View>
            <View style={s.statusRow}>
              <StatusPill ok={!!leftEyeImage}  label="Left eye" />
              <StatusPill ok={!!rightEyeImage} label="Right eye" />
            </View>
          </View>

          {/* ── Audio Consent ── */}
          <View style={s.card}>
            <Text style={s.sect}>Audio Consent</Text>
            {!audioAvailable ? (
              <View style={s.audioWarning}>
                <Icon name="alert-circle-outline" size={18} color="#f59e0b" />
                <Text style={s.audioWarningText}>
                  Install expo-av to enable audio consent:{"\n"}
                  <Text style={{ fontFamily: "monospace" }}>npx expo install expo-av</Text>
                </Text>
              </View>
            ) : (
              <>
                <Text style={[s.fieldLabel, { marginBottom: 14 }]}>
                  Record your verbal consent before submitting
                </Text>
                <TouchableOpacity
                  style={[s.recordBtn, isRecording && s.recordBtnActive]}
                  onPress={isRecording ? stopRecording : startRecording}
                >
                  <Icon name={isRecording ? "stop-circle" : "microphone"} size={24} color="#fff" />
                  <Text style={s.recordBtnText}>
                    {isRecording ? "Stop Recording" : "Start Recording"}
                  </Text>
                </TouchableOpacity>
                {audioUri && (
                  <View style={s.audioSuccess}>
                    <Icon name="check-circle" size={18} color={C.success} />
                    <Text style={{ color: C.success, fontSize: 14, fontWeight: "600" }}>
                      Audio consent recorded
                    </Text>
                  </View>
                )}
              </>
            )}
          </View>

          {/* ── Submit ── */}
          <TouchableOpacity
            style={[s.submitBtn, loading && s.submitBtnDisabled]}
            onPress={submitData}
            disabled={loading}
          >
            {loading ? (
              <View style={{ alignItems: "center", gap: 8 }}>
                <ActivityIndicator color="#fff" />
                <Text style={[s.submitBtnText, { fontSize: 13 }]}>{loadingMsg}</Text>
              </View>
            ) : (
              <Text style={s.submitBtnText}>Analyze Health →</Text>
            )}
          </TouchableOpacity>

          <TouchableOpacity style={s.resetBtn} onPress={resetForm}>
            <Icon name="refresh" size={16} color={C.muted} />
            <Text style={s.resetBtnText}>Clear Form</Text>
          </TouchableOpacity>
        </ScrollView>
      </KeyboardAvoidingView>

      {/* Preview modal */}
      <Modal visible={!!previewUri} transparent animationType="fade" onRequestClose={() => setPreviewUri(null)}>
        <TouchableOpacity style={s.modalOverlay} activeOpacity={1} onPress={() => setPreviewUri(null)}>
          {previewUri && (
            <View style={s.modalCard}>
              <Image
                source={{ uri: previewUri }}
                style={{ width: PREVIEW_SIZE, height: PREVIEW_SIZE, borderRadius: 16 }}
                resizeMode="contain"
              />
              <Text style={s.modalHint}>Tap anywhere to close</Text>
            </View>
          )}
        </TouchableOpacity>
      </Modal>
    </SafeAreaView>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────
function Field({ label, value, onChange, placeholder, keyboard = "default" }: {
  label: string; value: string; onChange: (v: string) => void;
  placeholder: string; keyboard?: any;
}) {
  return (
    <>
      <Text style={s.fieldLabel}>{label}</Text>
      <TextInput
        style={s.input}
        value={value}
        onChangeText={onChange}
        placeholder={placeholder}
        placeholderTextColor="#bbb"
        keyboardType={keyboard}
      />
    </>
  );
}

function StatusPill({ ok, label }: { ok: boolean; label: string }) {
  return (
    <View style={[s.pill, { backgroundColor: ok ? "#e8f5e9" : "#f1f3f5" }]}>
      <Text style={[s.pillText, { color: ok ? "#2f9e44" : "#868e96" }]}>
        {ok ? "✓" : "○"} {label}
      </Text>
    </View>
  );
}

// ─── Styles ───────────────────────────────────────────────────────────────────
const s = StyleSheet.create({
  safe: { flex: 1, backgroundColor: "#fff" },

  header: {
    flexDirection: "row", alignItems: "center",
    backgroundColor: "#fff", paddingHorizontal: 20,
    paddingTop: 14, paddingBottom: 12,
    borderBottomWidth: 0.5, borderBottomColor: "#ede7f6",
  },
  headerTitle: { fontSize: 20, fontWeight: "800", color: "#1a1a2e" },
  headerSub:   { fontSize: 11, color: "#bbb", marginTop: 2 },
  dashBtn: {
    flexDirection: "row", alignItems: "center", gap: 6,
    backgroundColor: "#f3eefe", borderRadius: 10,
    paddingVertical: 8, paddingHorizontal: 14,
  },
  dashBtnText: { color: "#5b2d8e", fontWeight: "700", fontSize: 13 },

  progressTrack: { height: 3, backgroundColor: "#d1c4e9" },
  progressFill:  { height: 3, backgroundColor: "#5b2d8e" },

  scroll: { padding: 16, paddingBottom: 40 },

  card: {
    backgroundColor: "#fff", borderRadius: 16, padding: 18,
    marginBottom: 16, borderWidth: 0.5, borderColor: "#ede7f6",
  },
  sect: {
    fontSize: 11, fontWeight: "700", color: "#5b2d8e",
    textTransform: "uppercase", letterSpacing: 1, marginBottom: 14,
  },
  fieldLabel: { fontSize: 12, fontWeight: "600", color: "#888", marginBottom: 6 },
  input: {
    backgroundColor: "#f7f4fc", borderWidth: 0.5, borderColor: "#d1c4e9",
    borderRadius: 10, padding: 12, marginBottom: 14, fontSize: 14, color: "#1a1a2e",
  },

  genderRow: { flexDirection: "row", gap: 10 },
  genderBtn: {
    flex: 1, padding: 12, backgroundColor: "#f7f4fc",
    borderRadius: 10, alignItems: "center",
    borderWidth: 1, borderColor: "transparent",
  },
  genderBtnActive:  { backgroundColor: "#f3eefe", borderColor: "#5b2d8e" },
  genderText:       { fontWeight: "600", color: "#666", fontSize: 14 },
  genderTextActive: { color: "#5b2d8e" },

  eyeHint: { fontSize: 13, fontWeight: "600", color: "#1a1a2e", marginBottom: 14 },
  eyeRow:  { flexDirection: "row", gap: 12, marginBottom: 4 },
  eyeCol:  { flex: 1 },

  imgBtn: {
    borderRadius: 12, paddingVertical: 14, paddingHorizontal: 10,
    alignItems: "center", marginBottom: 10,
    flexDirection: "row", justifyContent: "center", gap: 8,
  },
  imgBtnReady: { backgroundColor: "#5b2d8e" },
  imgBtnDone:  { backgroundColor: "#6c757d" },
  imgBtnText:  { color: "#fff", fontWeight: "700", fontSize: 13 },
  imgBtnSub:   { color: "rgba(255,255,255,0.7)", fontSize: 11 },

  thumb: {
    marginBottom: 8, borderRadius: 10, overflow: "hidden",
    position: "relative", aspectRatio: 1, width: "100%",
  },
  thumbImg:       { width: "100%", height: "100%" },
  thumbBadge:     { position: "absolute", top: 6, left: 6, paddingVertical: 2, paddingHorizontal: 8, borderRadius: 6 },
  thumbBadgeText: { color: "#fff", fontWeight: "700", fontSize: 10 },

  statusRow: { flexDirection: "row", gap: 8, flexWrap: "wrap", marginTop: 10 },
  pill:      { paddingVertical: 5, paddingHorizontal: 12, borderRadius: 20 },
  pillText:  { fontSize: 11, fontWeight: "600" },

  audioWarning: {
    flexDirection: "row", gap: 10, alignItems: "flex-start",
    backgroundColor: "#fffbeb", borderRadius: 10, padding: 14,
    borderWidth: 1, borderColor: "#fde68a",
  },
  audioWarningText: { flex: 1, fontSize: 13, color: "#92400e", lineHeight: 20 },

  recordBtn: {
    backgroundColor: "#5b2d8e", padding: 16, borderRadius: 12,
    flexDirection: "row", alignItems: "center", justifyContent: "center", gap: 10,
  },
  recordBtnActive: { backgroundColor: "#d32f2f" },
  recordBtnText:   { color: "#fff", fontWeight: "700", fontSize: 15 },
  audioSuccess: { flexDirection: "row", alignItems: "center", gap: 8, marginTop: 12 },

  submitBtn:         { backgroundColor: "#5b2d8e", borderRadius: 16, padding: 18, alignItems: "center", marginBottom: 12 },
  submitBtnDisabled: { backgroundColor: "#a07bc4" },
  submitBtnText:     { color: "#fff", fontWeight: "700", fontSize: 16 },

  resetBtn: {
    borderRadius: 14, padding: 14, alignItems: "center",
    borderWidth: 0.5, borderColor: "#ede7f6",
    backgroundColor: "#fff", flexDirection: "row",
    justifyContent: "center", gap: 8,
  },
  resetBtnText: { color: "#888", fontWeight: "600", fontSize: 14 },

  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.92)", justifyContent: "center", alignItems: "center" },
  modalCard:    { alignItems: "center" },
  modalHint:    { color: "rgba(255,255,255,0.4)", marginTop: 14, fontSize: 13 },
});