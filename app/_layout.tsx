// app/_layout.tsx
import { Stack } from "expo-router";
 
export default function RootLayout() {
  return (
    <Stack screenOptions={{ headerShown: false }}>
      {/* Route: / → Form screen */}
      <Stack.Screen name="index" options={{ headerShown: false }} />
 
      {/* Route: /eye-capture → Camera + TFLite detection */}
      <Stack.Screen
        name="eye-capture"
        options={{
          headerShown: false,
          animation: "slide_from_right",
          gestureEnabled: false, // prevent accidental swipe-back mid-capture
        }}
      />
 
      {/* Route: /(tabs)/* → Tab group (dashboard + view) */}
      <Stack.Screen name="(tabs)" options={{ headerShown: false }} />
    </Stack>
  );
}