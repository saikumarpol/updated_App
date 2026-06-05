// app/(tabs)/view.tsx
import { useEffect, useState, useCallback } from "react";
import {
  View, Text, FlatList, StyleSheet, ActivityIndicator,
  RefreshControl, TextInput, TouchableOpacity, StatusBar,
} from "react-native";
import { useSafeAreaInsets } from "react-native-safe-area-context";
import axios from "axios";
import { MaterialCommunityIcons } from "@expo/vector-icons";
import { useRouter } from "expo-router";

const Icon = MaterialCommunityIcons;

// const BASE_URL = "http://10.2.136.235:5001";
const BASE_URL = "https://pl-api.iiit.ac.in/rcts/anemiav2/";

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
  success: "#4ade80",
  warning: "#fbbf24",
  danger: "#f87171",
};

const FILTERS = ["All", "Anemic", "Grossly Anemic", "Normal"] as const;
type Filter = typeof FILTERS[number];

export default function ViewScreen() {
  const insets = useSafeAreaInsets();
  const router = useRouter();

  const [data, setData]               = useState<any[]>([]);
  const [filteredData, setFilteredData] = useState<any[]>([]);
  const [loading, setLoading]         = useState(true);
  const [refreshing, setRefreshing]   = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [activeFilter, setActiveFilter] = useState<Filter>("All");

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      const res = await axios.get(`${BASE_URL}/api/children`);
      setData(res.data);
      setFilteredData(res.data);
    } catch (err) {
      console.log("Error fetching records", err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchData(); }, [fetchData]);

  useEffect(() => {
    let result = [...data];
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      result = result.filter(i =>
        i.name?.toLowerCase().includes(q) ||
        i.parentName?.toLowerCase().includes(q)
      );
    }
    if (activeFilter !== "All") {
      result = result.filter(i => {
        const status = (i.anemiaStatus || "").toLowerCase();
        if (activeFilter === "Anemic") return status.includes("anemic") && !status.includes("grossly");
        if (activeFilter === "Grossly Anemic") return status.includes("grossly");
        if (activeFilter === "Normal") return status === "normal";
        return true;
      });
    }
    setFilteredData(result);
  }, [searchQuery, data, activeFilter]);

  const onRefresh = useCallback(() => {
    setRefreshing(true);
    fetchData().finally(() => setRefreshing(false));
  }, [fetchData]);

  const getAnemiaColor = (status?: string) => {
    if (!status) return "#adb5bd";
    const s = status.toLowerCase();
    if (s.includes("grossly")) return C.danger;
    if (s.includes("anemic")) return C.warning;
    if (s === "normal") return C.success;
    return "#adb5bd";
  };

  const calculateAvgHb = (item: any) => {
    const left = item.leftHb || 0, right = item.rightHb || 0;
    if (left === 0 && right === 0) return "—";
    return ((left + right) / 2).toFixed(2);
  };

  const formatDate = (date: string) =>
    new Date(date).toLocaleDateString("en-GB", { day: "numeric", month: "short", year: "numeric" });

  const renderItem = ({ item }: { item: any }) => (
    <TouchableOpacity style={s.card} activeOpacity={0.88}>
      <View style={s.cardHeader}>
        <View style={s.nameContainer}>
          <View style={s.nameIcon}>
            <Icon name="account-child" size={18} color={C.purple} />
          </View>
          <View>
            <Text style={s.childName}>{item.name || "Unnamed"}</Text>
            <Text style={s.childMeta}>
              {item.age} yrs • {item.gender} • Parent: {item.parentName || "—"}
            </Text>
          </View>
        </View>
        <Text style={s.date}>{formatDate(item.createdAt)}</Text>
      </View>

      <View style={s.divider} />

      <View style={s.statsContainer}>
        <View style={s.hbContainer}>
          <Text style={s.hbLabel}>Average Hb</Text>
          <Text style={s.hbValue}>{calculateAvgHb(item)} g/dL</Text>
        </View>
        <View style={[s.statusBadge, { backgroundColor: getAnemiaColor(item.anemiaStatus) }]}>
          <Text style={s.statusText}>{item.anemiaStatus || "Unknown"}</Text>
        </View>
      </View>
    </TouchableOpacity>
  );

  if (loading && !refreshing) {
    return (
      <View style={[s.center, { paddingTop: insets.top }]}>
        <ActivityIndicator size="large" color={C.purple} />
        <Text style={s.loadingText}>Loading records…</Text>
      </View>
    );
  }

  return (
    <View style={[s.container, { paddingTop: insets.top }]}>
      <StatusBar barStyle="dark-content" backgroundColor="#fff" />

      <View style={s.header}>
        <View>
          <Text style={s.screenTitle}>Records</Text>
          <Text style={s.screenSub}>{data.length} Assessments</Text>
        </View>
        <View style={{ flexDirection: "row", gap: 10 }}>
          <TouchableOpacity style={s.iconBtn} onPress={() => router.push("/")}>
            <Icon name="plus" size={20} color={C.purple} />
          </TouchableOpacity>
          <TouchableOpacity style={s.iconBtn} onPress={onRefresh}>
            <Icon name="refresh" size={20} color={C.purple} />
          </TouchableOpacity>
        </View>
      </View>

      <View style={s.searchWrap}>
        <View style={s.searchBox}>
          <Icon name="magnify" size={20} color={C.hint} />
          <TextInput
            style={s.searchInput}
            placeholder="Search by name or parent..."
            value={searchQuery}
            onChangeText={setSearchQuery}
            placeholderTextColor={C.hint}
          />
          {searchQuery.length > 0 && (
            <TouchableOpacity onPress={() => setSearchQuery("")}>
              <Icon name="close-circle" size={18} color={C.hint} />
            </TouchableOpacity>
          )}
        </View>
      </View>

      <View style={s.filterScroll}>
        <View style={s.filterRow}>
          {FILTERS.map((f) => (
            <TouchableOpacity
              key={f}
              style={[s.filterChip, activeFilter === f && s.filterChipActive]}
              onPress={() => setActiveFilter(f)}
            >
              <Text style={[s.filterText, activeFilter === f && s.filterTextActive]}>{f}</Text>
            </TouchableOpacity>
          ))}
        </View>
      </View>

      <FlatList
        data={filteredData}
        keyExtractor={(item, index) => item._id || `item-${index}`}
        renderItem={renderItem}
        contentContainerStyle={s.listContent}
        refreshControl={<RefreshControl refreshing={refreshing} onRefresh={onRefresh} tintColor={C.purple} />}
        ListEmptyComponent={
          <View style={s.emptyContainer}>
            <Icon name="database-search" size={60} color="#d1c4e9" />
            <Text style={s.emptyText}>
              {searchQuery ? "No matching records found" : "No records yet"}
            </Text>
          </View>
        }
      />
    </View>
  );
}

const s = StyleSheet.create({
  container: { flex: 1, backgroundColor: C.page },
  center: { flex: 1, justifyContent: "center", alignItems: "center", backgroundColor: C.page },
  loadingText: { marginTop: 16, fontSize: 16, color: C.muted },

  header: {
    flexDirection: "row", justifyContent: "space-between", alignItems: "center",
    paddingHorizontal: 20, paddingVertical: 16, backgroundColor: C.card,
    borderBottomWidth: 0.5, borderBottomColor: C.border,
  },
  screenTitle: { fontSize: 22, fontWeight: "800", color: C.text },
  screenSub:   { fontSize: 12, color: C.hint, marginTop: 2 },
  iconBtn: {
    width: 40, height: 40, borderRadius: 12,
    backgroundColor: C.purpleLight, alignItems: "center", justifyContent: "center",
  },

  searchWrap: { backgroundColor: C.card, padding: 16, borderBottomWidth: 0.5, borderBottomColor: C.border },
  searchBox: {
    flexDirection: "row", alignItems: "center", backgroundColor: C.page,
    borderRadius: 12, borderWidth: 0.5, borderColor: C.purpleBorder,
    paddingHorizontal: 12, height: 48, gap: 10,
  },
  searchInput: { flex: 1, fontSize: 15, color: C.text },

  filterScroll: { backgroundColor: C.card, borderBottomWidth: 0.5, borderBottomColor: C.border },
  filterRow: { flexDirection: "row", paddingHorizontal: 16, paddingVertical: 12, gap: 10, flexWrap: "wrap" },
  filterChip: { paddingVertical: 6, paddingHorizontal: 16, borderRadius: 20, borderWidth: 0.5, borderColor: C.purpleBorder, backgroundColor: C.card },
  filterChipActive: { backgroundColor: C.purple, borderColor: C.purple },
  filterText: { fontSize: 13, fontWeight: "600", color: C.muted },
  filterTextActive: { color: "#fff" },

  listContent: { padding: 16, paddingBottom: 30 },

  card: {
    backgroundColor: C.card, borderRadius: 16, padding: 16,
    marginBottom: 14, borderWidth: 0.5, borderColor: C.border,
  },
  cardHeader: { flexDirection: "row", justifyContent: "space-between", alignItems: "flex-start" },
  nameContainer: { flexDirection: "row", gap: 12, flex: 1 },
  nameIcon: { width: 38, height: 38, borderRadius: 10, backgroundColor: C.purpleLight, alignItems: "center", justifyContent: "center" },
  childName: { fontSize: 17, fontWeight: "700", color: C.text },
  childMeta: { fontSize: 12.5, color: C.hint, marginTop: 2 },
  date: { fontSize: 12, color: C.hint },

  divider: { height: 0.8, backgroundColor: C.border, marginVertical: 14 },

  statsContainer: { flexDirection: "row", justifyContent: "space-between", alignItems: "center" },
  hbContainer: { flex: 1 },
  hbLabel: { fontSize: 12, color: C.muted },
  hbValue: { fontSize: 18, fontWeight: "700", color: C.text, marginTop: 2 },

  statusBadge: { paddingVertical: 8, paddingHorizontal: 18, borderRadius: 20, alignSelf: "flex-start" },
  statusText: { color: "#fff", fontWeight: "700", fontSize: 14 },

  emptyContainer: { alignItems: "center", paddingTop: 80 },
  emptyText: { marginTop: 16, fontSize: 16, color: "#aaa", textAlign: "center" },
});