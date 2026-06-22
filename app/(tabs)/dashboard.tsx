// app/(tabs)/dashboard.tsx
import { useEffect, useState, useCallback } from "react";
import {
  View, Text, ScrollView, RefreshControl, TouchableOpacity,
  ActivityIndicator, SafeAreaView, StatusBar, StyleSheet, Dimensions,
} from "react-native";
import axios from "axios";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useRouter } from "expo-router";
import { PieChart } from "react-native-chart-kit";

const Icon = MaterialCommunityIcons;

const BASE_URL = "https://pl-api.iiit.ac.in/rcts/anemiav2";
// const BASE_URL = "http://192.168.1.6:5001";
const { width: SW } = Dimensions.get("window");

const C = {
  purple: "#5b2d8e",
  purpleLight: "#f3eefe",
  page: "#f7f4fc",
  card: "#ffffff",
  text: "#1a1a2e",
  muted: "#666666",
  hint: "#999999",
  success: "#4ade80",
  warning: "#fbbf24",
  danger: "#f87171",
};

const chartConfig = {
  backgroundGradientFrom: "#fff",
  backgroundGradientTo: "#fff",
  color: () => "#000",
  labelColor: () => "#000",
};

// TODO: revert to leftHb/rightHb when API Hb values are reliable
const getDisplayHb = (item: any): string => {
  if (item.demoHb !== null && item.demoHb !== undefined) {
    return `${parseFloat(item.demoHb).toFixed(1)} g/dL`;
  }
  const left = item.leftHb || 0, right = item.rightHb || 0;
  if (left === 0 && right === 0) return "—";
  return `${((left + right) / 2).toFixed(1)} g/dL`;
};

export default function Dashboard() {
  const router = useRouter();
  const [data, setData]             = useState<any[]>([]);
  const [loading, setLoading]       = useState(true);
  const [refreshing, setRefreshing] = useState(false);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${BASE_URL}/api/children`);
      setData(res.data || []);
    } catch (err) {
      console.log("Dashboard fetch error:", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchData().finally(() => setRefreshing(false));
  }, [fetchData]);

  const totalRecords  = data.length;
  const normalCount   = data.filter(i => i.anemiaStatus === "Normal").length;
  const anemicCount   = data.filter(i => i.anemiaStatus === "Anemic").length;
  const grosslyCount  = data.filter(i => i.anemiaStatus === "Grossly Anemic").length;
  const anemiaPercent = totalRecords > 0
    ? Math.round(((anemicCount + grosslyCount) / totalRecords) * 100)
    : 0;

  const pieData = [
    { name: "Normal",         population: normalCount,  color: C.success },
    { name: "Anemic",         population: anemicCount,  color: C.warning },
    { name: "Grossly Anemic", population: grosslyCount, color: C.danger },
  ].filter(item => item.population > 0);

  const recentRecords = [...data]
    .sort((a, b) => new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime())
    .slice(0, 5);

  if (loading && !refreshing) {
    return (
      <SafeAreaView style={s.safeArea}>
        <View style={[s.safeArea, { justifyContent: "center", alignItems: "center" }]}>
          <ActivityIndicator size="large" color={C.purple} />
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={s.safeArea}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      <View style={s.header}>
        <View>
          <Text style={s.screenTitle}>Dashboard</Text>
          <Text style={s.screenSub}>Health Overview</Text>
        </View>
        <TouchableOpacity style={s.newAssessmentBtn} onPress={() => router.push("/")}>
          <Icon name="plus" size={18} color={C.purple} />
          <Text style={s.newAssessmentText}>New</Text>
        </TouchableOpacity>
      </View>

      <ScrollView
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.purple} />}
        contentContainerStyle={s.scrollContent}
      >
        {/* Stats row */}
        <View style={s.statsRow}>
          <View style={s.statCard}>
            <Icon name="account-group" size={28} color={C.purple} />
            <Text style={s.statNumber}>{totalRecords}</Text>
            <Text style={s.statLabel}>Total Assessments</Text>
          </View>
          <View style={s.statCard}>
            <Icon name="blood-bag" size={28} color={C.purple} />
            <Text style={[s.statNumber, { color: anemiaPercent > 30 ? C.danger : "#22c55e" }]}>
              {anemiaPercent}%
            </Text>
            <Text style={s.statLabel}>Anemic Rate</Text>
          </View>
        </View>

        {/* Pie chart */}
        <View style={s.chartCard}>
          <Text style={s.sectionTitle}>Anemia Status Distribution</Text>
          <View style={s.pieAndLegend}>
            <View style={s.pieContainer}>
              {pieData.length > 0 ? (
                <PieChart
                  data={pieData}
                  width={SW * 0.47}
                  height={195}
                  chartConfig={chartConfig}
                  accessor="population"
                  backgroundColor="transparent"
                  paddingLeft="35"
                  absolute
                  hasLegend={false}
                />
              ) : (
                <Text style={s.noData}>No data yet</Text>
              )}
            </View>
            <View style={s.legendContainer}>
              {[
                { color: C.success, label: "Normal",         count: normalCount },
                { color: C.warning, label: "Anemic",         count: anemicCount },
                { color: C.danger,  label: "Grossly Anemic", count: grosslyCount },
              ].map(({ color, label, count }) => (
                <View key={label} style={s.legendRow}>
                  <View style={[s.legendDot, { backgroundColor: color }]} />
                  <Text style={s.legendLabel}>{label}</Text>
                  <Text style={s.legendCount}>{count}</Text>
                </View>
              ))}
            </View>
          </View>
        </View>

        {/* Recent assessments */}
        <View style={s.recentCard}>
          <View style={s.recentHeader}>
            <Text style={s.sectionTitle}>Recent Assessments</Text>
            <TouchableOpacity onPress={() => router.push("/(tabs)/view")}>
              <Text style={s.seeAll}>See all →</Text>
            </TouchableOpacity>
          </View>

          {recentRecords.length === 0 && <Text style={s.noData}>No records yet</Text>}

          {recentRecords.map((item, i) => (
            <View key={item._id || i} style={s.recentItem}>
              <View style={{ flex: 1 }}>
                <Text style={s.recentName}>{item.name || "Unnamed"}</Text>
                <Text style={s.recentMeta}>
                  {item.age} yrs • {item.gender} • Hb: {getDisplayHb(item)}
                </Text>
              </View>
              <View style={[s.statusPill, { backgroundColor: getCategoryColor(item.anemiaStatus) }]}>
                <Text style={s.statusPillText}>{item.anemiaStatus || "—"}</Text>
              </View>
            </View>
          ))}
        </View>

        <TouchableOpacity style={s.newBtn} onPress={() => router.push("/")}>
          <Icon name="plus" size={24} color="#fff" />
          <Text style={s.newBtnText}>New Assessment</Text>
        </TouchableOpacity>
      </ScrollView>
    </SafeAreaView>
  );
}

const getCategoryColor = (status?: string) => {
  if (!status) return "#9ca3af";
  const s = status.toLowerCase();
  if (s.includes("grossly")) return C.danger;
  if (s.includes("anemic")) return C.warning;
  return C.success;
};

const s = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: "#f8f5ff" },
  header: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingHorizontal: 20, paddingVertical: 16, backgroundColor: "#fff" },
  screenTitle: { fontSize: 26, fontWeight: "800", color: "#1a1a2e" },
  screenSub: { fontSize: 13, color: C.hint },
  newAssessmentBtn: { flexDirection: "row", alignItems: "center", gap: 6, backgroundColor: "#f3eefe", borderRadius: 10, paddingVertical: 8, paddingHorizontal: 14 },
  newAssessmentText: { color: "#5b2d8e", fontWeight: "700", fontSize: 13 },
  scrollContent: { padding: 16, paddingBottom: 100 },
  statsRow: { flexDirection: "row", gap: 12, marginBottom: 20 },
  statCard: { flex: 1, backgroundColor: "#fff", borderRadius: 18, padding: 16, borderWidth: 1, borderColor: "#f0e6ff" },
  statNumber: { fontSize: 32, fontWeight: "800", color: "#1a1a2e", marginTop: 4 },
  statLabel: { fontSize: 13, color: C.muted },
  chartCard: { backgroundColor: "#fff", borderRadius: 18, padding: 18, marginBottom: 20, borderWidth: 1, borderColor: "#f0e6ff" },
  sectionTitle: { fontSize: 18, fontWeight: "700", marginBottom: 16 },
  pieAndLegend: { flexDirection: "row", alignItems: "center" },
  pieContainer: { width: SW * 0.48 },
  legendContainer: { flex: 1, paddingLeft: 8 },
  legendRow: { flexDirection: "row", alignItems: "center", marginBottom: 14 },
  legendDot: { width: 13, height: 13, borderRadius: 7, marginRight: 10 },
  legendLabel: { flex: 1, fontSize: 14, color: "#333" },
  legendCount: { fontSize: 15, fontWeight: "700", color: "#1a1a2e" },
  recentCard: { backgroundColor: "#fff", borderRadius: 18, padding: 18, borderWidth: 1, borderColor: "#f0e6ff" },
  recentHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", marginBottom: 12 },
  seeAll: { color: C.purple, fontWeight: "700" },
  recentItem: { flexDirection: "row", justifyContent: "space-between", alignItems: "center", paddingVertical: 12, borderBottomWidth: 0.5, borderBottomColor: "#f0e6ff" },
  recentName: { fontSize: 16, fontWeight: "700" },
  recentMeta: { fontSize: 13, color: C.muted, marginTop: 2 },
  statusPill: { paddingHorizontal: 14, paddingVertical: 6, borderRadius: 20 },
  statusPillText: { color: "#fff", fontSize: 13, fontWeight: "600" },
  newBtn: { backgroundColor: C.purple, flexDirection: "row", alignItems: "center", justifyContent: "center", padding: 18, borderRadius: 18, gap: 10, marginTop: 20 },
  newBtnText: { color: "#fff", fontSize: 17, fontWeight: "700" },
  noData: { color: C.muted, textAlign: "center", marginTop: 20, fontSize: 14 },
});
