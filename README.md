# HCI Posture Engine

A browser-based posture detection prototype that uses MediaPipe Pose Landmarker to monitor the user's head, neck, and shoulders through a webcam.

## Features

- 10-second personal posture calibration
- Neck and shoulder collapse detection
- Shoulder tilt detection
- Head offset and head tilt detection
- Personalized thresholds based on median and median absolute deviation (MAD)
- Local camera processing with GPU support and CPU fallback

The overall posture state is determined by the most severe individual signal. The numerical score is provided for debugging only.

## Run Locally

```bash
pnpm install
pnpm dev
```

Open the local URL shown in the terminal and allow camera access. On the original development machine, `start.cmd` can also be used.

## How It Works

The engine normalizes head, ear, and shoulder measurements by shoulder width. A 10-second calibration establishes the user's baseline posture and natural movement range. Calibration should be performed while sitting upright and typing normally with both hands on the keyboard.

See [docs/algorithm.md](docs/algorithm.md) for the current formulas, thresholds, and output format.

## UI Integration

The included page is only a debug view. A production UI can read `window.postureEngineState` or subscribe to the browser's `posturechange` event. This keeps the posture engine independent from any Figma-derived interface.

## Disclaimer

This is an HCI prototype and not a medical diagnostic tool.
