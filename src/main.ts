import { FilesetResolver, PoseLandmarker, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import "./style.css";
import {
  assess,
  extractFeatures,
  landmarkQuality,
  makeBaseline,
  smoothFeatures,
  type Assessment,
  type Baseline,
  type Features,
  type IssueKey,
  type Landmark,
  type PostureStatus,
} from "./posture";

const MODEL_PATH = "/models/pose_landmarker_lite.task";
const WASM_PATH = "/wasm";
const CALIBRATION_MS = 10_000;
const WARNING_DELAY_MS = 900;
const BAD_POSTURE_DELAY_MS = 1_500;
const BASELINE_STORAGE_KEY = "posture-baseline-v1";

const video = document.querySelector<HTMLVideoElement>("#video")!;
const canvas = document.querySelector<HTMLCanvasElement>("#overlay")!;
const ctx = canvas.getContext("2d")!;
const placeholder = document.querySelector<HTMLElement>("#placeholder")!;
const cameraButton = document.querySelector<HTMLButtonElement>("#cameraButton")!;
const calibrateButton = document.querySelector<HTMLButtonElement>("#calibrateButton")!;
const calibrationOverlay = document.querySelector<HTMLElement>("#calibrationOverlay")!;
const calibrationHint = document.querySelector<HTMLElement>("#calibrationHint")!;
const countdown = document.querySelector<HTMLElement>("#countdown")!;
const ringProgress = document.querySelector<SVGCircleElement>("#ringProgress")!;
const stageMessage = document.querySelector<HTMLElement>("#stageMessage")!;
const postureAlert = document.querySelector<HTMLElement>("#postureAlert")!;
const statusPill = document.querySelector<HTMLElement>("#statusPill")!;
const statusText = document.querySelector<HTMLElement>("#statusText")!;
const postureLabel = document.querySelector<HTMLElement>("#postureLabel")!;
const scoreElement = document.querySelector<HTMLElement>("#score")!;
const scoreBar = document.querySelector<HTMLElement>("#scoreBar")!;
const suggestion = document.querySelector<HTMLElement>("#suggestion")!;
const debugToggle = document.querySelector<HTMLButtonElement>("#debugToggle")!;
const debugPanel = document.querySelector<HTMLElement>("#debugPanel")!;
const debugNeck = document.querySelector<HTMLElement>("#debugNeck")!;
const debugShoulder = document.querySelector<HTMLElement>("#debugShoulder")!;
const debugOffset = document.querySelector<HTMLElement>("#debugOffset")!;
const debugTilt = document.querySelector<HTMLElement>("#debugTilt")!;
const debugFps = document.querySelector<HTMLElement>("#debugFps")!;

let poseLandmarker: PoseLandmarker | null = null;
let stream: MediaStream | null = null;
let baseline: Baseline | null = loadBaseline();
let calibrating = false;
let calibrationStartedAt = 0;
let calibrationSamples: Features[] = [];
let featureHistory: Features[] = [];
let lastVideoTime = -1;
let lastInferenceAt = 0;
let badSince: number | null = null;
let candidateStatus: PostureStatus | null = null;
let stableAssessment: Assessment | null = null;
let lastFrameAt = performance.now();
let fps = 0;
let modelBackend = "GPU";

const connections: [number, number][] = [
  [7, 8],
  [7, 11],
  [8, 12],
  [11, 12],
  [2, 5],
  [2, 7],
  [5, 8],
];
const visiblePoints = [0, 2, 5, 7, 8, 11, 12];

async function createLandmarker(delegate: "GPU" | "CPU") {
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);
  return PoseLandmarker.createFromOptions(vision, {
    baseOptions: { modelAssetPath: MODEL_PATH, delegate },
    runningMode: "VIDEO",
    numPoses: 1,
    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55,
    outputSegmentationMasks: false,
  });
}

async function initializeModel() {
  if (poseLandmarker) return;
  setStageMessage("LOADING POSE MODEL…", true);
  try {
    try {
      poseLandmarker = await createLandmarker("GPU");
      modelBackend = "GPU";
    } catch (gpuError) {
      console.warn("GPU delegate unavailable; falling back to CPU", gpuError);
      poseLandmarker = await createLandmarker("CPU");
      modelBackend = "CPU";
    }
  } finally {
    setStageMessage("", false);
  }
}

async function startCamera() {
  cameraButton.disabled = true;
  cameraButton.textContent = "STARTING…";
  try {
    await initializeModel();
    stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 },
        frameRate: { ideal: 30 },
      },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    resizeCanvas();
    placeholder.hidden = true;
    statusPill.hidden = false;
    calibrateButton.disabled = false;
    calibrateButton.textContent = baseline ? "RECALIBRATE" : "CALIBRATE FOR 10 SECONDS";
    cameraButton.textContent = "STOP CAMERA";
    cameraButton.disabled = false;
    cameraButton.classList.add("is-stop");
    renderLoop();
  } catch (error) {
    console.error(error);
    setStageMessage(cameraErrorMessage(error), true);
    cameraButton.textContent = "RETRY CAMERA";
    cameraButton.disabled = false;
  }
}

function stopCamera() {
  stream?.getTracks().forEach((track) => track.stop());
  stream = null;
  video.srcObject = null;
  placeholder.hidden = false;
  statusPill.hidden = true;
  calibrateButton.disabled = true;
  calibrating = false;
  calibrationOverlay.hidden = true;
  cameraButton.textContent = "START CAMERA";
  cameraButton.classList.remove("is-stop");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function startCalibration() {
  if (!stream) return;
  calibrating = true;
  calibrationStartedAt = performance.now();
  calibrationSamples = [];
  featureHistory = [];
  stableAssessment = null;
  candidateStatus = null;
  badSince = null;
  calibrationOverlay.hidden = false;
  calibrateButton.disabled = true;
  postureLabel.textContent = "CALIBRATING";
  scoreElement.textContent = "—";
  updateCalibrationOverlay(0, "Face the camera and relax both shoulders");
  emitEngineState({ phase: "calibrating", message: "Calibration started" });
}

function finishCalibration() {
  if (calibrationSamples.length < 45) {
    calibrationStartedAt = performance.now();
    calibrationSamples = [];
    updateCalibrationOverlay(0, "Not enough valid frames. Keep both ears and shoulders visible.");
    return;
  }
  baseline = makeBaseline(calibrationSamples);
  localStorage.setItem(BASELINE_STORAGE_KEY, JSON.stringify(baseline));
  calibrating = false;
  calibrationOverlay.hidden = true;
  calibrateButton.disabled = false;
  calibrateButton.textContent = "RECALIBRATE";
  postureLabel.textContent = "CALIBRATED";
  suggestion.textContent = `Baseline created from ${baseline.sampleCount} valid frames.`;
  emitEngineState({ phase: "ready", message: "Calibration complete" });
}

function renderLoop() {
  if (!stream || !poseLandmarker) return;
  resizeCanvas();
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    const now = performance.now();
    // 15 FPS is enough for posture and leaves the page responsive.
    if (now - lastInferenceAt >= 60) {
      lastInferenceAt = now;
      lastVideoTime = video.currentTime;
      const result = poseLandmarker.detectForVideo(video, now);
      const landmarks = result.landmarks[0] as Landmark[] | undefined;
      drawFrame(landmarks);
      processLandmarks(landmarks, now);
      const frameDelta = now - lastFrameAt;
      lastFrameAt = now;
      fps = fps ? fps * 0.85 + (1000 / frameDelta) * 0.15 : 1000 / frameDelta;
    }
  }
  requestAnimationFrame(renderLoop);
}

function processLandmarks(landmarks: Landmark[] | undefined, now: number) {
  if (!landmarks) {
    showUnavailable("Move into frame and keep your head and both shoulders visible");
    return;
  }
  const quality = landmarkQuality(landmarks);
  if (!quality.ok) {
    showUnavailable(quality.reason);
    if (calibrating) calibrationHint.textContent = quality.reason;
    return;
  }
  const features = extractFeatures(landmarks, video.videoWidth, video.videoHeight);
  if (!features) return;

  featureHistory.push(features);
  if (featureHistory.length > 12) featureHistory.shift();
  const smoothed = smoothFeatures(featureHistory);
  updateDebug(smoothed);

  if (calibrating) {
    calibrationSamples.push(features);
    const elapsed = now - calibrationStartedAt;
    updateCalibrationOverlay(elapsed / CALIBRATION_MS, "Hold still and breathe normally");
    if (elapsed >= CALIBRATION_MS) finishCalibration();
    return;
  }

  if (!baseline) {
    setStatus("ready", "READY TO CALIBRATE");
    postureLabel.textContent = "NOT CALIBRATED";
    postureAlert.hidden = true;
    return;
  }

  const assessment = assess(smoothed, baseline);
  if (assessment.positionChanged) {
    showUnavailable("Camera distance changed. Return to the calibrated position or recalibrate.");
    badSince = null;
    candidateStatus = null;
    return;
  }

  const hasIssue = assessment.status !== "good";
  if (hasIssue) {
    if (candidateStatus !== assessment.status) {
      candidateStatus = assessment.status;
      badSince = now;
    }
    const delay = assessment.status === "bad" ? BAD_POSTURE_DELAY_MS : WARNING_DELAY_MS;
    if (now - (badSince ?? now) >= delay) stableAssessment = assessment;
  } else {
    badSince = null;
    candidateStatus = null;
    stableAssessment = assessment;
  }
  const display = stableAssessment ?? assessment;
  updateAssessment(display, smoothed);
}

function updateAssessment(assessment: Assessment, features: Features) {
  const score = assessment.score;
  scoreElement.textContent = String(score);
  scoreBar.style.width = `${score}%`;
  scoreBar.style.background = score >= 80 ? "var(--good)" : score >= 60 ? "var(--warn)" : "var(--bad)";

  if (assessment.status === "good") {
    setStatus("good", "POSTURE OK");
    postureLabel.textContent = "GOOD";
    suggestion.textContent = "All signals are inside the calibrated range.";
  } else if (assessment.status === "warning") {
    setStatus("warn", "WARNING");
    postureLabel.textContent = "WARNING";
    suggestion.textContent = assessment.reasons[0] ?? "Return to the calibrated posture.";
  } else {
    setStatus("bad", "BAD POSTURE");
    postureLabel.textContent = "BAD";
    suggestion.textContent = assessment.reasons.join("; ") || "Return to the calibrated posture.";
  }

  updatePostureAlert(assessment);
  emitEngineState({ phase: "tracking", assessment, features });

  updateMetric("collapse", assessment.issues.collapse, features.neckRatio, baseline!.features.neckRatio.median);
  updateMetric("shoulder", assessment.issues.shoulder, features.shoulderTiltDeg, baseline!.features.shoulderTiltDeg.median);
  updateMetric("head", assessment.issues.head, features.headOffsetRatio, baseline!.features.headOffsetRatio.median);
}

function updateMetric(key: IssueKey, severity: number, current: number, reference: number) {
  const element = document.querySelector<HTMLElement>(`[data-metric="${key}"]`)!;
  element.dataset.state = severity >= 1 ? "warn" : "good";
  const text = document.querySelector<HTMLElement>(`#${key}Text`)!;
  if (key === "collapse") text.textContent = severity >= 1 ? "SPACE REDUCED" : "NORMAL";
  if (key === "shoulder") {
    text.textContent = severity < 1
      ? "LEVEL"
      : current - reference > 0
        ? "LEFT SHOULDER HIGH"
        : "RIGHT SHOULDER HIGH";
  }
  if (key === "head") text.textContent = severity >= 1 ? "OUTSIDE BASELINE" : "NORMAL";
  element.title = `Current ${current.toFixed(3)} / baseline ${reference.toFixed(3)}`;
}

function showUnavailable(message: string) {
  setStatus("neutral", "UNAVAILABLE");
  postureAlert.hidden = true;
  emitEngineState({ phase: "unavailable", message });
  if (!calibrating) {
    postureLabel.textContent = "NO READING";
    suggestion.textContent = message;
    scoreElement.textContent = "—";
    scoreBar.style.width = "0";
  }
}

function updateCalibrationOverlay(progress: number, hint: string) {
  const clamped = Math.max(0, Math.min(1, progress));
  const circumference = 2 * Math.PI * 44;
  ringProgress.style.strokeDasharray = String(circumference);
  ringProgress.style.strokeDashoffset = String(circumference * (1 - clamped));
  countdown.textContent = String(Math.max(0, Math.ceil(10 * (1 - clamped))));
  calibrationHint.textContent = hint;
}

function drawFrame(landmarks?: Landmark[]) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!landmarks) return;
  ctx.save();
  ctx.translate(canvas.width, 0);
  ctx.scale(-1, 1);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "rgba(196, 255, 99, .9)";
  ctx.lineWidth = 3;
  for (const [from, to] of connections) {
    const a = landmarks[from];
    const b = landmarks[to];
    if (!a || !b || (a.visibility ?? 1) < 0.45 || (b.visibility ?? 1) < 0.45) continue;
    ctx.beginPath();
    ctx.moveTo(a.x * canvas.width, a.y * canvas.height);
    ctx.lineTo(b.x * canvas.width, b.y * canvas.height);
    ctx.stroke();
  }
  for (const index of visiblePoints) {
    const point = landmarks[index];
    if (!point || (point.visibility ?? 1) < 0.45) continue;
    ctx.fillStyle = index >= 11 ? "#ffffff" : "#c4ff63";
    ctx.beginPath();
    ctx.arc(point.x * canvas.width, point.y * canvas.height, index >= 11 ? 6 : 4, 0, Math.PI * 2);
    ctx.fill();
  }
  ctx.restore();
}

function resizeCanvas() {
  if (!video.videoWidth || !video.videoHeight) return;
  if (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
}

function setStatus(state: "good" | "warn" | "bad" | "neutral" | "ready", text: string) {
  statusPill.dataset.state = state;
  statusText.textContent = text;
}

function setStageMessage(message: string, visible: boolean) {
  stageMessage.textContent = message;
  stageMessage.hidden = !visible;
}

type PublicEngineState = {
  phase: "calibrating" | "ready" | "tracking" | "unavailable";
  message?: string;
  assessment?: Assessment;
  features?: Features;
  timestamp?: number;
};

declare global {
  interface Window {
    postureEngineState?: PublicEngineState;
  }
}

function emitEngineState(state: PublicEngineState) {
  const detail = { ...state, timestamp: Date.now() };
  window.postureEngineState = detail;
  window.dispatchEvent(new CustomEvent("posturechange", { detail }));
}

function updatePostureAlert(assessment: Assessment) {
  if (assessment.status === "good") {
    postureAlert.hidden = true;
    return;
  }
  let message = assessment.status === "bad" ? "POSTURE OUTSIDE BASELINE" : "POSTURE WARNING";
  if (assessment.shoulderDirection === "left_high") message = "LEFT SHOULDER TOO HIGH";
  else if (assessment.shoulderDirection === "right_high") message = "RIGHT SHOULDER TOO HIGH";
  else if (assessment.signals.neckCollapse.status !== "good") message = "NECK / SHOULDER SPACE TOO SMALL";
  else if (assessment.signals.headTilt.status !== "good") message = "HEAD TILT TOO LARGE";
  else if (assessment.signals.headOffset.status !== "good") message = "HEAD TOO FAR OFF CENTER";
  postureAlert.textContent = message;
  postureAlert.dataset.level = assessment.status;
  postureAlert.hidden = false;
}

function updateDebug(features: Features) {
  debugNeck.textContent = features.neckRatio.toFixed(3);
  debugShoulder.textContent = `${features.shoulderTiltDeg.toFixed(1)}°`;
  debugOffset.textContent = features.headOffsetRatio.toFixed(3);
  debugTilt.textContent = `${features.headTiltDeg.toFixed(1)}°`;
  debugFps.textContent = `${fps.toFixed(0)} FPS · ${modelBackend}`;
}

function loadBaseline(): Baseline | null {
  try {
    const raw = localStorage.getItem(BASELINE_STORAGE_KEY);
    return raw ? (JSON.parse(raw) as Baseline) : null;
  } catch {
    return null;
  }
}

function cameraErrorMessage(error: unknown) {
  if (error instanceof DOMException && error.name === "NotAllowedError") return "Camera permission denied. Allow camera access and retry.";
  if (error instanceof DOMException && error.name === "NotFoundError") return "No camera was found.";
  return "Could not start the camera or pose model. Reload and retry.";
}

cameraButton.addEventListener("click", () => (stream ? stopCamera() : void startCamera()));
calibrateButton.addEventListener("click", startCalibration);
debugToggle.addEventListener("click", () => {
  const willShow = debugPanel.hidden;
  debugPanel.hidden = !willShow;
  debugToggle.textContent = willShow ? "HIDE VALUES" : "SHOW VALUES";
  debugToggle.setAttribute("aria-expanded", String(willShow));
});
window.addEventListener("resize", resizeCanvas);
window.addEventListener("beforeunload", stopCamera);

if (baseline) {
  calibrateButton.textContent = "RECALIBRATE";
  suggestion.textContent = "A saved baseline was found. Start the camera or recalibrate.";
}

// Keep the imported MediaPipe type checked by TypeScript when package declarations evolve.
void (null as NormalizedLandmark | null);
