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
  setStageMessage("正在載入骨架模型…", true);
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
  cameraButton.textContent = "正在開啟…";
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
    calibrateButton.textContent = baseline ? "重新校正" : "開始 10 秒校正";
    cameraButton.textContent = "關閉相機";
    cameraButton.disabled = false;
    cameraButton.classList.add("is-stop");
    renderLoop();
  } catch (error) {
    console.error(error);
    setStageMessage(cameraErrorMessage(error), true);
    cameraButton.textContent = "重試開啟相機";
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
  cameraButton.textContent = "開啟相機";
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
  postureLabel.textContent = "校正中";
  scoreElement.textContent = "—";
  updateCalibrationOverlay(0, "正面看向鏡頭，雙肩放鬆");
}

function finishCalibration() {
  if (calibrationSamples.length < 45) {
    calibrationStartedAt = performance.now();
    calibrationSamples = [];
    updateCalibrationOverlay(0, "有效畫面不足，請保持雙耳與雙肩清楚可見");
    return;
  }
  baseline = makeBaseline(calibrationSamples);
  localStorage.setItem(BASELINE_STORAGE_KEY, JSON.stringify(baseline));
  calibrating = false;
  calibrationOverlay.hidden = true;
  calibrateButton.disabled = false;
  calibrateButton.textContent = "重新校正";
  postureLabel.textContent = "校正完成";
  suggestion.textContent = `已從 ${baseline.sampleCount} 個有效畫面建立個人基準。現在可以自然使用電腦。`;
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
    showUnavailable("請坐到鏡頭前，讓頭部與雙肩出現在畫面中");
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
    updateCalibrationOverlay(elapsed / CALIBRATION_MS, "很好，維持自然呼吸");
    if (elapsed >= CALIBRATION_MS) finishCalibration();
    return;
  }

  if (!baseline) {
    setStatus("ready", "可以開始校正");
    postureLabel.textContent = "等待校正";
    return;
  }

  const assessment = assess(smoothed, baseline);
  if (assessment.positionChanged) {
    showUnavailable("你與鏡頭的距離和校正時不同，請回到原本位置或重新校正");
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
    setStatus("good", "坐姿穩定");
    postureLabel.textContent = "姿勢很好";
    suggestion.textContent = "頭、頸、肩仍在你的舒適基準範圍內。";
  } else if (assessment.status === "warning") {
    setStatus("warn", "稍微調整一下");
    postureLabel.textContent = "有些偏離";
    suggestion.textContent = assessment.reasons[0] ?? "放鬆肩膀，讓頭部回到中央。";
  } else {
    setStatus("bad", "需要調整");
    postureLabel.textContent = "姿勢偏離";
    suggestion.textContent = assessment.reasons.join("；") || "請回到校正時的自然坐姿。";
  }

  updateMetric("collapse", assessment.issues.collapse, features.neckRatio, baseline!.features.neckRatio.median);
  updateMetric("shoulder", assessment.issues.shoulder, features.shoulderTiltDeg, baseline!.features.shoulderTiltDeg.median);
  updateMetric("head", assessment.issues.head, features.headOffsetRatio, baseline!.features.headOffsetRatio.median);
}

function updateMetric(key: IssueKey, severity: number, current: number, reference: number) {
  const element = document.querySelector<HTMLElement>(`[data-metric="${key}"]`)!;
  element.dataset.state = severity >= 1 ? "warn" : "good";
  const text = document.querySelector<HTMLElement>(`#${key}Text`)!;
  if (key === "collapse") text.textContent = severity >= 1 ? "比基準明顯縮短" : "維持個人基準";
  if (key === "shoulder") text.textContent = severity >= 1 ? "肩線出現傾斜" : "左右保持平衡";
  if (key === "head") text.textContent = severity >= 1 ? "偏離中央位置" : "位置自然穩定";
  element.title = `目前 ${current.toFixed(3)}／基準 ${reference.toFixed(3)}`;
}

function showUnavailable(message: string) {
  setStatus("neutral", "暫時無法判斷");
  if (!calibrating) {
    postureLabel.textContent = "調整畫面位置";
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
  if (error instanceof DOMException && error.name === "NotAllowedError") return "相機權限被拒絕。請允許此網站使用相機後再試一次。";
  if (error instanceof DOMException && error.name === "NotFoundError") return "找不到可用的相機。";
  return "無法啟動相機或模型，請重新整理後再試一次。";
}

cameraButton.addEventListener("click", () => (stream ? stopCamera() : void startCamera()));
calibrateButton.addEventListener("click", startCalibration);
debugToggle.addEventListener("click", () => {
  const willShow = debugPanel.hidden;
  debugPanel.hidden = !willShow;
  debugToggle.textContent = willShow ? "隱藏數值" : "顯示數值";
  debugToggle.setAttribute("aria-expanded", String(willShow));
});
window.addEventListener("resize", resizeCanvas);
window.addEventListener("beforeunload", stopCamera);

if (baseline) {
  calibrateButton.textContent = "重新校正";
  suggestion.textContent = "已找到上次的個人基準。開啟相機即可開始，或選擇重新校正。";
}

// Keep the imported MediaPipe type checked by TypeScript when package declarations evolve.
void (null as NormalizedLandmark | null);
