export type Landmark = {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
  presence?: number;
};

export type Features = {
  neckRatio: number;
  neckLengthRatio: number;
  headOffsetRatio: number;
  shoulderTiltDeg: number;
  headTiltDeg: number;
  sideAsymmetry: number;
  shoulderWidthNormalized: number;
};

export type BaselineMetric = { median: number; mad: number };
export type Baseline = {
  createdAt: number;
  sampleCount: number;
  features: Record<keyof Features, BaselineMetric>;
};

export type IssueKey = "collapse" | "shoulder" | "head";
export type PostureStatus = "good" | "warning" | "bad";
export type ShoulderDirection = "level" | "left_high" | "right_high";
export type SignalKey = "neckCollapse" | "shoulderTilt" | "headOffset" | "headTilt" | "sideAsymmetry";
export type SignalAssessment = {
  value: number;
  baseline: number;
  delta: number;
  warningThreshold: number;
  badThreshold: number;
  /** 1 = warning threshold, 2 = bad threshold. */
  severity: number;
  status: PostureStatus;
};
export type Assessment = {
  /** Debug/visualization only. Do not use this field for the posture decision. */
  score: number;
  status: PostureStatus;
  shoulderDirection: ShoulderDirection;
  signals: Record<SignalKey, SignalAssessment>;
  issues: Record<IssueKey, number>;
  reasons: string[];
  positionChanged: boolean;
};

const IDX = {
  leftEye: 2,
  rightEye: 5,
  leftEar: 7,
  rightEar: 8,
  leftShoulder: 11,
  rightShoulder: 12,
} as const;

const required = Object.values(IDX);

function px(point: Landmark, width: number, height: number) {
  return { x: point.x * width, y: point.y * height };
}

function distance(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function midpoint(a: { x: number; y: number }, b: { x: number; y: number }) {
  return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 };
}

function angleDeg(a: { x: number; y: number }, b: { x: number; y: number }) {
  return Math.atan2(b.y - a.y, b.x - a.x) * (180 / Math.PI);
}

export function landmarkQuality(landmarks: Landmark[]): { ok: boolean; reason: string } {
  if (!landmarks || landmarks.length < 13) return { ok: false, reason: "找不到完整的頭肩骨架" };
  const weak = required.some((index) => {
    const point = landmarks[index];
    return !point || (point.visibility ?? 1) < 0.55 || (point.presence ?? 1) < 0.55;
  });
  if (weak) return { ok: false, reason: "請讓雙耳與雙肩保持在畫面內" };
  const shoulderWidth = Math.abs(landmarks[IDX.leftShoulder].x - landmarks[IDX.rightShoulder].x);
  if (shoulderWidth < 0.09) return { ok: false, reason: "請靠近鏡頭，讓肩膀清楚可見" };
  return { ok: true, reason: "" };
}

export function extractFeatures(landmarks: Landmark[], width: number, height: number): Features | null {
  if (!landmarkQuality(landmarks).ok || width <= 0 || height <= 0) return null;

  const leftShoulder = px(landmarks[IDX.leftShoulder], width, height);
  const rightShoulder = px(landmarks[IDX.rightShoulder], width, height);
  const leftEar = px(landmarks[IDX.leftEar], width, height);
  const rightEar = px(landmarks[IDX.rightEar], width, height);
  const shoulderMid = midpoint(leftShoulder, rightShoulder);
  const earMid = midpoint(leftEar, rightEar);
  const shoulderWidth = distance(leftShoulder, rightShoulder);
  if (shoulderWidth < 1) return null;

  const leftSide = distance(leftEar, leftShoulder) / shoulderWidth;
  const rightSide = distance(rightEar, rightShoulder) / shoulderWidth;

  return {
    neckRatio: (shoulderMid.y - earMid.y) / shoulderWidth,
    neckLengthRatio: distance(shoulderMid, earMid) / shoulderWidth,
    headOffsetRatio: (earMid.x - shoulderMid.x) / shoulderWidth,
    shoulderTiltDeg: angleDeg(leftShoulder, rightShoulder),
    headTiltDeg: angleDeg(leftEar, rightEar),
    sideAsymmetry: leftSide - rightSide,
    shoulderWidthNormalized: shoulderWidth / width,
  };
}

export function median(values: number[]): number {
  if (!values.length) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

export function makeBaseline(samples: Features[]): Baseline {
  if (samples.length < 10) throw new Error("校正樣本不足");
  const keys = Object.keys(samples[0]) as (keyof Features)[];
  const features = {} as Record<keyof Features, BaselineMetric>;
  for (const key of keys) {
    const values = samples.map((sample) => sample[key]);
    const center = median(values);
    features[key] = {
      median: center,
      mad: median(values.map((value) => Math.abs(value - center))),
    };
  }
  return { createdAt: Date.now(), sampleCount: samples.length, features };
}

function tolerance(metric: BaselineMetric, floor: number, madMultiplier = 3) {
  return Math.max(floor, metric.mad * madMultiplier);
}

function signal(
  value: number,
  baseline: BaselineMetric,
  delta: number,
  warningFloor: number,
  badFloor: number,
): SignalAssessment {
  const warningThreshold = tolerance(baseline, warningFloor, 3);
  const badThreshold = Math.max(
    tolerance(baseline, badFloor, 6),
    warningThreshold * 1.8,
  );
  const severity = delta <= warningThreshold
    ? Math.max(0, delta / warningThreshold)
    : 1 + (delta - warningThreshold) / (badThreshold - warningThreshold);
  const status: PostureStatus = delta >= badThreshold
    ? "bad"
    : delta >= warningThreshold
      ? "warning"
      : "good";
  return {
    value,
    baseline: baseline.median,
    delta,
    warningThreshold,
    badThreshold,
    severity: Math.max(0, Math.min(3, severity)),
    status,
  };
}

function worstStatus(signals: SignalAssessment[]): PostureStatus {
  if (signals.some((item) => item.status === "bad")) return "bad";
  if (signals.some((item) => item.status === "warning")) return "warning";
  return "good";
}

function diagnosticScore(worstSeverity: number, badSeverity: number) {
  if (worstSeverity < 1) return Math.round(100 - worstSeverity * 19);
  if (worstSeverity < badSeverity) {
    return Math.round(80 - ((worstSeverity - 1) / (badSeverity - 1)) * 20);
  }
  return Math.max(0, Math.round(59 - (worstSeverity - badSeverity) * 25));
}

export function assess(features: Features, baseline: Baseline): Assessment {
  const b = baseline.features;
  const distanceChange = Math.abs(
    features.shoulderWidthNormalized / b.shoulderWidthNormalized.median - 1,
  );
  const positionChanged = distanceChange > 0.25;

  const collapseDelta = b.neckRatio.median - features.neckRatio;
  const shoulderDelta = Math.abs(features.shoulderTiltDeg - b.shoulderTiltDeg.median);
  const offsetDelta = Math.abs(features.headOffsetRatio - b.headOffsetRatio.median);
  const tiltDelta = Math.abs(features.headTiltDeg - b.headTiltDeg.median);
  const asymmetryDelta = Math.abs(features.sideAsymmetry - b.sideAsymmetry.median);

  // Floors are deliberately strict. Calibration noise can only widen them when
  // the person's measured natural movement is larger than these values.
  const signals: Record<SignalKey, SignalAssessment> = {
    neckCollapse: signal(
      features.neckRatio,
      b.neckRatio,
      collapseDelta,
      Math.max(0.02, Math.abs(b.neckRatio.median) * 0.045),
      Math.max(0.04, Math.abs(b.neckRatio.median) * 0.09),
    ),
    shoulderTilt: signal(features.shoulderTiltDeg, b.shoulderTiltDeg, shoulderDelta, 2, 4.5),
    headOffset: signal(features.headOffsetRatio, b.headOffsetRatio, offsetDelta, 0.035, 0.07),
    headTilt: signal(features.headTiltDeg, b.headTiltDeg, tiltDelta, 3, 6),
    sideAsymmetry: signal(features.sideAsymmetry, b.sideAsymmetry, asymmetryDelta, 0.045, 0.09),
  };
  const issues = {
    collapse: signals.neckCollapse.severity,
    shoulder: signals.shoulderTilt.severity,
    head: Math.max(
      signals.headOffset.severity,
      signals.headTilt.severity,
      signals.sideAsymmetry.severity,
    ),
  };
  const status = worstStatus(Object.values(signals));
  const shoulderDirection: ShoulderDirection = signals.shoulderTilt.status === "good"
    ? "level"
    : features.shoulderTiltDeg - b.shoulderTiltDeg.median > 0
      ? "left_high"
      : "right_high";
  const worstSignal = Object.values(signals).reduce((worst, item) =>
    item.severity > worst.severity ? item : worst,
  );
  const score = positionChanged ? 0 : diagnosticScore(worstSignal.severity, 2);
  const reasons: string[] = [];
  if (signals.neckCollapse.status !== "good") reasons.push("頸肩空間正在縮短");
  if (shoulderDirection === "left_high") reasons.push("左肩偏高");
  if (shoulderDirection === "right_high") reasons.push("右肩偏高");
  if (
    signals.headOffset.status !== "good" ||
    signals.headTilt.status !== "good" ||
    signals.sideAsymmetry.status !== "good"
  ) reasons.push("頭部偏離校正位置");

  return { score, status, shoulderDirection, signals, issues, reasons, positionChanged };
}

export function smoothFeatures(history: Features[]): Features {
  const keys = Object.keys(history[0]) as (keyof Features)[];
  const result = {} as Features;
  for (const key of keys) result[key] = median(history.map((item) => item[key]));
  return result;
}
