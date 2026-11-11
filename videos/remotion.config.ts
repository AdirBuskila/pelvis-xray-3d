import { Config } from "@remotion/cli/config";

// AMD iGPU, no NVIDIA: ANGLE gives headless Chrome a working WebGL backend
Config.setChromiumOpenGlRenderer("angle");
Config.setVideoImageFormat("jpeg");
Config.setJpegQuality(95);
Config.setConcurrency(4);
