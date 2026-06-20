// app/index.tsx
import { MaterialCommunityIcons } from "@expo/vector-icons";
import axios from "axios";
import { useLocalSearchParams, useRouter } from "expo-router";
import { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Alert,
  Dimensions,
  Image,
  KeyboardAvoidingView,
  Modal,
  Platform,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from "react-native";
import { SafeAreaView } from "react-native-safe-area-context";

const Icon = MaterialCommunityIcons;

// ─── CONFIG ──────────────────────────────────────────────────────────────────
// const BASE_URL = "https://pl-api.iiit.ac.in/rcts/anemiav2/";
const BASE_URL = "http://192.168.1.11:5001";
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
  const [name, setName] = useState("");
  const [parentName, setParentName] = useState("");
  const [phoneNumber, setPhoneNumber] = useState("");
  const [age, setAge] = useState("");
  const [gender, setGender] = useState("");

  // ── Eye images ─────────────────────────────────────────────────
  const [leftEyeImage, setLeftEyeImage] = useState<string | null>(null);
  const [rightEyeImage, setRightEyeImage] = useState<string | null>(null);
  const [eyeSessionId, setEyeSessionId] = useState("");

  // ── UI ─────────────────────────────────────────────────────────
  const [loading, setLoading] = useState(false);
  const [loadingMsg, setLoadingMsg] = useState("Processing...");
  const [previewUri, setPreviewUri] = useState<string | null>(null);

  // Restore form fields when returning from eye-capture
  useEffect(() => {
    if (params.name) setName(params.name);
    if (params.parentName) setParentName(params.parentName);
    if (params.phoneNumber) setPhoneNumber(params.phoneNumber);
    if (params.age) setAge(params.age);
    if (params.gender) setGender(params.gender);
    if (params.eyeSessionId) setEyeSessionId(params.eyeSessionId);
    if (params.leftEyeImage) setLeftEyeImage(params.leftEyeImage);
    if (params.rightEyeImage) setRightEyeImage(params.rightEyeImage);
  }, [params]);

  // ── Helpers ────────────────────────────────────────────────────
  const isMinor = age ? parseInt(age) < 18 : false;
  const capturedCount = [leftEyeImage, rightEyeImage].filter(Boolean).length;
  const hasAnyEyeImage = capturedCount > 0;

  // Progress: 50% for having at least one eye image, 50% for any personal info
  const progress = Math.min(
    100,
    (hasAnyEyeImage ? 50 : 0) + (name || age || gender ? 50 : 0)
  );

  const resetForm = () => {
    setName(""); setParentName(""); setPhoneNumber("");
    setAge(""); setGender("");
    setLeftEyeImage(null); setRightEyeImage(null); setEyeSessionId("");
  };

  const goToCapture = (side: "left" | "right") => {
    router.push({
      pathname: "/eye-capture",
      params: {
        name, parentName, phoneNumber, age, gender, eyeSessionId,
        leftEyeImage: leftEyeImage ?? "",
        rightEyeImage: rightEyeImage ?? "",
        eyeSide: side,
      },
    });
  };

  // ── Submit ─────────────────────────────────────────────────────
  const submitData = async () => {
    // Only require at least one eye image — everything else is optional
    if (!leftEyeImage && !rightEyeImage) {
      Alert.alert("Missing Image", "Please capture at least one eye image.");
      return;
    }

    setLoading(true);
    setLoadingMsg("Uploading images...");

    const formData = new FormData();
    formData.append("name", name || "");
    formData.append("parentName", parentName || "");
    formData.append("phoneNumber", phoneNumber || "");
    formData.append("age", age || "");
    formData.append("gender", gender || "");
    if (eyeSessionId) formData.append("eyeSessionId", eyeSessionId);

    if (leftEyeImage) {
      formData.append("leftEyeImage", {
        uri: leftEyeImage,
        type: "image/jpeg",
        name: `left_${Date.now()}.jpg`,
      } as any);
    }
    if (rightEyeImage) {
      formData.append("rightEyeImage", {
        uri: rightEyeImage,
        type: "image/jpeg",
        name: `right_${Date.now()}.jpg`,
      } as any);
    }

    const MESSAGES = [
      "Uploading image…",
      "Analysing eye…",
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

      const leftLine = leftEyeImage
        ? `Left Eye  → Hb: ${d.leftHb ?? "N/A"} | ${d.leftAnemia ?? "Unknown"}\n\n`
        : "";
      const rightLine = rightEyeImage
        ? `Right Eye → Hb: ${d.rightHb ?? "N/A"} | ${d.rightAnemia ?? "Unknown"}`
        : "";

      Alert.alert(
        "✅ Analysis Complete",
        leftLine + rightLine,
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
          <Text style={s.headerSub}>All fields are optional</Text>
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
            <Field label="Name" value={name} onChange={setName} placeholder="Enter full name" />
            <Field label="Age (years)" value={age} onChange={setAge} placeholder="Age" keyboard="numeric" />
            <Field label="Phone Number" value={phoneNumber} onChange={setPhoneNumber} placeholder="10-digit mobile number" keyboard="phone-pad" />
            {isMinor && (
              <Field label="Parent/Guardian Name" value={parentName} onChange={setParentName} placeholder="Enter parent name" />
            )}
            <Text style={s.fieldLabel}>Gender</Text>
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
            <Text style={s.sect}>Capture Eye Image</Text>
            <Text style={s.eyeHint}>At least one eye image is required</Text>
            <View style={s.eyeRow}>
              {([
                { side: "left", label: "LEFT", img: leftEyeImage, color: C.purple },
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
              <StatusPill ok={!!leftEyeImage} label="Left eye" />
              <StatusPill ok={!!rightEyeImage} label="Right eye" />
            </View>
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
  headerSub: { fontSize: 11, color: "#bbb", marginTop: 2 },
  dashBtn: {
    flexDirection: "row", alignItems: "center", gap: 6,
    backgroundColor: "#f3eefe", borderRadius: 10,
    paddingVertical: 8, paddingHorizontal: 14,
  },
  dashBtnText: { color: "#5b2d8e", fontWeight: "700", fontSize: 13 },

  progressTrack: { height: 3, backgroundColor: "#d1c4e9" },
  progressFill: { height: 3, backgroundColor: "#5b2d8e" },

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
  genderBtnActive: { backgroundColor: "#f3eefe", borderColor: "#5b2d8e" },
  genderText: { fontWeight: "600", color: "#666", fontSize: 14 },
  genderTextActive: { color: "#5b2d8e" },

  eyeHint: { fontSize: 13, fontWeight: "600", color: "#1a1a2e", marginBottom: 14 },
  eyeRow: { flexDirection: "row", gap: 12, marginBottom: 4 },
  eyeCol: { flex: 1 },

  imgBtn: {
    borderRadius: 12, paddingVertical: 14, paddingHorizontal: 10,
    alignItems: "center", marginBottom: 10,
    flexDirection: "row", justifyContent: "center", gap: 8,
  },
  imgBtnReady: { backgroundColor: "#5b2d8e" },
  imgBtnDone: { backgroundColor: "#6c757d" },
  imgBtnText: { color: "#fff", fontWeight: "700", fontSize: 13 },
  imgBtnSub: { color: "rgba(255,255,255,0.7)", fontSize: 11 },

  thumb: {
    marginBottom: 8, borderRadius: 10, overflow: "hidden",
    position: "relative", aspectRatio: 1, width: "100%",
  },
  thumbImg: { width: "100%", height: "100%" },
  thumbBadge: { position: "absolute", top: 6, left: 6, paddingVertical: 2, paddingHorizontal: 8, borderRadius: 6 },
  thumbBadgeText: { color: "#fff", fontWeight: "700", fontSize: 10 },

  statusRow: { flexDirection: "row", gap: 8, flexWrap: "wrap", marginTop: 10 },
  pill: { paddingVertical: 5, paddingHorizontal: 12, borderRadius: 20 },
  pillText: { fontSize: 11, fontWeight: "600" },

  submitBtn: { backgroundColor: "#5b2d8e", borderRadius: 16, padding: 18, alignItems: "center", marginBottom: 12 },
  submitBtnDisabled: { backgroundColor: "#a07bc4" },
  submitBtnText: { color: "#fff", fontWeight: "700", fontSize: 16 },

  resetBtn: {
    borderRadius: 14, padding: 14, alignItems: "center",
    borderWidth: 0.5, borderColor: "#ede7f6",
    backgroundColor: "#fff", flexDirection: "row",
    justifyContent: "center", gap: 8,
  },
  resetBtnText: { color: "#888", fontWeight: "600", fontSize: 14 },

  modalOverlay: { flex: 1, backgroundColor: "rgba(0,0,0,0.92)", justifyContent: "center", alignItems: "center" },
  modalCard: { alignItems: "center" },
  modalHint: { color: "rgba(255,255,255,0.4)", marginTop: 14, fontSize: 13 },
});