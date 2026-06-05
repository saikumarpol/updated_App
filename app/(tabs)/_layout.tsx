// app/(tabs)/_layout.tsx
import { Tabs } from "expo-router";
import { MaterialCommunityIcons } from "@expo/vector-icons";

const Icon = MaterialCommunityIcons;

const C = {
  purple: "#5b2d8e",
  inactive: "#aaa",
  bg: "#fff",
  border: "#ede7f6",
};

export default function TabsLayout() {
  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarActiveTintColor: C.purple,
        tabBarInactiveTintColor: C.inactive,
        tabBarStyle: {
          backgroundColor: C.bg,
          borderTopColor: C.border,
          borderTopWidth: 0.5,
          height: 62,
          paddingBottom: 8,
          paddingTop: 6,
        },
        tabBarLabelStyle: {
          fontSize: 11,
          fontWeight: "600",
        },
      }}
    >
      {/* Tab 1: Dashboard */}
      <Tabs.Screen
        name="dashboard"
        options={{
          title: "Dashboard",
          tabBarIcon: ({ color, size }) => (
            <Icon name="view-dashboard-outline" size={size} color={color} />
          ),
        }}
      />

      {/* Tab 2: Records */}
      <Tabs.Screen
        name="view"
        options={{
          title: "Records",
          tabBarIcon: ({ color, size }) => (
            <Icon name="clipboard-list-outline" size={size} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}