const { getDefaultConfig } = require("expo/metro-config");

const config = getDefaultConfig(__dirname);

// Push 'tflite' to the array of allowed asset extensions
config.resolver.assetExts.push("tflite");

module.exports = config;
