import React, { useEffect, useState } from "react";
import { StyleSheet, Text, View, ActivityIndicator } from "react-native";
import {
  Camera,
  useCameraDevice,
  useCameraPermission,
  useFrameProcessor,
} from "react-native-vision-camera";
import { useTensorflowModel } from "react-native-fast-tflite";
import { useResizePlugin } from "vision-camera-resize-plugin";
import { Worklets } from "react-native-worklets-core";

export default function ObjectDetectorApp() {
  const { hasPermission, requestPermission } = useCameraPermission();
  const device = useCameraDevice("back");

  // 1. Load your local YOLO TFLite model
  // Replace with your actual asset path or require() depending on your bundler config
  const plugin = useTensorflowModel(
    require("../assets/model/yolo-26-n-best_int8.tflite"),
  );
  const model = plugin.model;

  // 2. Initialize the resize plugin
  const { resize } = useResizePlugin();

  // State to hold detection results on the JS thread
  const [detections, setDetections] = useState<any[]>([]);

  // Request camera permissions on mount
  useEffect(() => {
    if (!hasPermission) {
      requestPermission();
    }
  }, [hasPermission]);

  // JS callback to update state safely from the Worklet thread
  const updateDetectionsJS = Worklets.createRunOnJS((results: any[]) => {
    setDetections(results);
  });

  // 3. Define the Frame Processor
  const frameProcessor = useFrameProcessor(
    (frame) => {
      "worklet";

      if (model == null) return;

      // YOLO models typically require 640x640 or 320x320 RGB float32 inputs
      // Adjust 'width' and 'height' based on your exact model's specifications
      const resized = resize(frame, {
        scale: {
          width: 320,
          height: 320,
        },
        pixelFormat: "rgb",
        dataType: "float32",
      });

      // Run inference
      const output = model.runSync([resized]);

      // 4. Post-processing the output tensor
      // YOLO tensor structures vary. Typically it's [1, boxes, classes + 4]
      // Here is a conceptual mapping to pass raw output data back to JS
      if (output && output.length > 0) {
        const rawPredictions = output[0];

        // Implement your specific YOLO anchor parsing / Non-Maximum Suppression (NMS) here.
        // For simplicity, we pass a sliced segment or flag back to the main thread:
        updateDetectionsJS([
          { message: "Objects detected!", rawCount: rawPredictions.length },
        ]);
      }
    },
    [model],
  );

  // Loading/Permission states
  if (!hasPermission)
    return (
      <View style={styles.container}>
        <Text>No Camera Permission</Text>
      </View>
    );
  if (device == null)
    return (
      <View style={styles.container}>
        <Text>No Camera Device Found</Text>
      </View>
    );
  if (model == null) {
    return (
      <View style={styles.container}>
        <ActivityIndicator size="large" color="#0000ff" />
        <Text style={styles.text}>Loading TFLite Model...</Text>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <Camera
        style={StyleSheet.absoluteFill}
        device={device}
        isActive={true}
        frameProcessor={frameProcessor}
        pixelFormat="yuv" // 'yuv' is recommended for performance on Android/iOS frame processors
      />

      {/* Overlay UI to display raw bounding box metadata or status */}
      <View style={styles.overlay}>
        {detections.map((d, index) => (
          <Text key={index} style={styles.detectionText}>
            {d.message} (Data points: {d.rawCount})
          </Text>
        ))}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    backgroundColor: "#000",
  },
  text: {
    color: "#fff",
    marginTop: 10,
  },
  overlay: {
    position: "absolute",
    bottom: 50,
    backgroundColor: "rgba(0,0,0,0.7)",
    padding: 15,
    borderRadius: 10,
  },
  detectionText: {
    color: "#00ff00",
    fontSize: 16,
    fontWeight: "bold",
  },
});
