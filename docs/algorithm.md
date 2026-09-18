# 頭頸肩姿勢引擎

這個模組的主要輸入是 MediaPipe Pose Landmarker 的 landmarks，主要輸出是結構化的 `Assessment`。UI 不應自行用總分推斷姿勢，而應直接讀取 `status` 與各個 `signals`。

## 輸入特徵

所有長度先換算成影像像素座標，再以肩寬正規化：

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

`neckRatio` 下降代表頭與肩膀間的垂直空間縮短。它可能來自駝背、縮脖或聳肩，因此引擎將它命名為「頸肩塌縮」，不宣稱是醫療上的駝背診斷。

## 個人基準

校正期間每個特徵保存：

- `median`：10 秒有效樣本的中位數。
- `mad`：與中位數差值的中位數，用來估計自然晃動與偵測雜訊。

單一特徵的實際門檻為 `max(固定下限, MAD × 倍率)`，避免校正畫面極度穩定時門檻趨近零，也避免自然晃動較大時頻繁誤報。

## 目前門檻

| 訊號 | 警戒 | 不良 |
|---|---:|---:|
| 頸肩塌縮 | 基準的 4.5%，至少 0.020 | 基準的 9%，至少 0.040 |
| 肩膀傾斜 | 2° | 4.5° |
| 頭部水平側移 | 0.035 個肩寬 | 0.070 個肩寬 |
| 頭部傾斜 | 3° | 6° |
| 左右頸肩不對稱 | 0.045 個肩寬 | 0.090 個肩寬 |

警戒門檻至少為 `3 × MAD`，不良門檻至少為 `6 × MAD`。與校正時相比，肩寬變化超過 25% 時標記為攝影機位置改變，不輸出姿勢好壞。

## 決策規則

每個訊號獨立輸出：

```ts
type SignalAssessment = {
  value: number;
  baseline: number;
  delta: number;
  warningThreshold: number;
  badThreshold: number;
  severity: number; // 1 = 警戒門檻，2 = 不良門檻
  status: "good" | "warning" | "bad";
};
```

The assessment also returns `shoulderDirection` as `level`, `left_high`, or `right_high`. Left and right always refer to the user's body, not the mirrored preview image.

整體 `status` 取所有訊號中最嚴重者，不做加權平均：

```text
任一訊號 bad     → 整體 bad
否則任一 warning → 整體 warning
否則              → 整體 good
```

`score` 僅為開發時的連續除錯值，顯示層不得使用它決定狀態。即時畫面採最近 12 個有效樣本的中位數；`warning` 持續 0.9 秒、`bad` 持續 1.5 秒後才確認，避免單幀抖動。

## 程式入口

- `extractFeatures(landmarks, width, height)`：骨架轉為正規化特徵。
- `makeBaseline(samples)`：建立個人校正基準。
- `assess(features, baseline)`：輸出每個訊號與整體判斷。
- `smoothFeatures(history)`：處理短期骨架抖動。
