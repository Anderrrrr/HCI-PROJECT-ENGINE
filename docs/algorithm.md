# Head, Neck, and Shoulder Posture Engine

The engine receives MediaPipe Pose Landmarker landmarks and returns a structured `Assessment`. Consumers should use `status` and `signals` directly. The numerical `score` is only a continuous debugging aid and must not be used as the posture decision.

## Input Features

Distances are converted to image pixels and normalized by shoulder width:

```text
shoulderCenter = midpoint(leftShoulder, rightShoulder)
earCenter      = midpoint(leftEar, rightEar)
shoulderWidth  = distance(leftShoulder, rightShoulder)

neckRatio      = (shoulderCenter.y - earCenter.y) / shoulderWidth
headOffset     = (earCenter.x - shoulderCenter.x) / shoulderWidth
shoulderTilt   = angle(leftShoulder, rightShoulder)
headTilt       = angle(leftEar, rightEar)
sideAsymmetry  = distance(leftEar, leftShoulder) / shoulderWidth
               - distance(rightEar, rightShoulder) / shoulderWidth
```

A decrease in `neckRatio` means the visible vertical space between the head and shoulders has become smaller. This can be caused by slouching, shrugging, or retracting the neck, so the engine calls it neck/shoulder collapse rather than making a medical diagnosis.

## Personal Baseline

For each feature, calibration stores:

- `median`: the median of valid samples collected over 10 seconds.
- `mad`: the median absolute deviation, used to estimate natural movement and landmark noise.

Calibration is performed while the user sits upright and types normally with both hands on the keyboard. This makes ordinary working movement part of the personal baseline instead of treating the transition from resting hands to typing as a posture error.

Each effective threshold is `max(fixed floor, MAD × multiplier)`. This prevents thresholds from approaching zero during an unusually still calibration while allowing more variation when the measured natural movement is larger.

## Current Thresholds

| Signal | Warning | Bad |
|---|---:|---:|
| Neck/shoulder collapse | 4.5% of baseline, minimum 0.020 | 9% of baseline, minimum 0.040 |
| Shoulder tilt | 2° | 4.5° |
| Horizontal head offset | 0.035 shoulder widths | 0.070 shoulder widths |
| Head tilt | 3° | 6° |
| Left/right neck asymmetry | 0.045 shoulder widths | 0.090 shoulder widths |

Warning thresholds are at least `3 × MAD`; bad thresholds are at least `6 × MAD`. A shoulder-width change greater than 25% from calibration is treated as a camera-distance change, so no posture decision is returned.

## Decision Rule

Each signal is evaluated independently:

```ts
type SignalAssessment = {
  value: number;
  baseline: number;
  delta: number;
  warningThreshold: number;
  badThreshold: number;
  severity: number; // 1 = warning threshold; 2 = bad threshold
  status: "good" | "warning" | "bad";
};
```

The overall status is the worst individual signal; signals are never averaged together:

```text
Any bad signal     -> overall bad
Else any warning  -> overall warning
Else              -> overall good
```

`shoulderDirection` is `level`, `left_high`, or `right_high`. Left and right always refer to the user's body, not the mirrored preview.

The live pipeline uses the median of the latest 12 valid samples. A warning must persist for 0.9 seconds and a bad result for 1.5 seconds before it is confirmed.

## Integration Output

The browser test harness publishes the latest result in two ways:

```js
window.postureEngineState

window.addEventListener("posturechange", (event) => {
  console.log(event.detail);
});
```

The event detail includes `phase`, `timestamp`, and, while tracking, `assessment` and normalized `features`. A production UI can subscribe to this event without depending on the debug page markup or styles.

## Code Entry Points

- `extractFeatures(landmarks, width, height)`: converts landmarks into normalized features.
- `makeBaseline(samples)`: builds a personal calibration baseline.
- `assess(features, baseline)`: returns signal-level and overall decisions.
- `smoothFeatures(history)`: reduces short-term landmark jitter.
