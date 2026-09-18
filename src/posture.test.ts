import { describe, expect, it } from "vitest";
import { assess, extractFeatures, makeBaseline, median, type Features, type Landmark } from "./posture";

const upright: Features = {
  neckRatio: 0.52,
  neckLengthRatio: 0.53,
  headOffsetRatio: 0,
  shoulderTiltDeg: 0,
  headTiltDeg: 0,
  sideAsymmetry: 0,
  shoulderWidthNormalized: 0.35,
};

function samples(): Features[] {
  return Array.from({ length: 80 }, (_, index) => ({
    ...upright,
    neckRatio: upright.neckRatio + ((index % 5) - 2) * 0.001,
    shoulderTiltDeg: ((index % 3) - 1) * 0.1,
  }));
}

function pose(leftShoulderY: number, rightShoulderY: number): Landmark[] {
  const points: Landmark[] = Array.from({ length: 13 }, () => ({
    x: 0.5,
    y: 0.3,
    visibility: 1,
    presence: 1,
  }));
  points[2] = { x: 0.55, y: 0.24, visibility: 1, presence: 1 };
  points[5] = { x: 0.45, y: 0.24, visibility: 1, presence: 1 };
  points[7] = { x: 0.6, y: 0.3, visibility: 1, presence: 1 };
  points[8] = { x: 0.4, y: 0.3, visibility: 1, presence: 1 };
  // Anatomical left appears on the image's right in the raw camera frame.
  points[11] = { x: 0.7, y: leftShoulderY, visibility: 1, presence: 1 };
  points[12] = { x: 0.3, y: rightShoulderY, visibility: 1, presence: 1 };
  return points;
}

describe("posture assessment", () => {
  it("calculates a median without outlier bias", () => {
    expect(median([1, 2, 2, 3, 100])).toBe(2);
  });

  it("accepts posture close to the personal baseline", () => {
    const result = assess({ ...upright, neckRatio: 0.51 }, makeBaseline(samples()));
    expect(result.status).toBe("good");
    expect(result.reasons).toHaveLength(0);
  });

  it("detects a collapsed neck-to-shoulder ratio", () => {
    const result = assess({ ...upright, neckRatio: 0.36 }, makeBaseline(samples()));
    expect(result.status).toBe("bad");
    expect(result.reasons).toContain("Neck and shoulder space is collapsing");
  });

  it("flags a five percent neck collapse even when all other signals are perfect", () => {
    const result = assess({ ...upright, neckRatio: 0.494 }, makeBaseline(samples()));
    expect(result.status).toBe("warning");
    expect(result.signals.neckCollapse.status).toBe("warning");
    expect(result.score).toBeLessThanOrEqual(80);
  });

  it("lets one serious signal determine the overall status", () => {
    const result = assess({ ...upright, shoulderTiltDeg: 5 }, makeBaseline(samples()));
    expect(result.status).toBe("bad");
    expect(result.signals.shoulderTilt.status).toBe("bad");
    expect(result.shoulderDirection).toBe("left_high");
    expect(result.reasons).toContain("Left shoulder is too high");
  });

  it("reports a raised right shoulder using the subject's perspective", () => {
    const result = assess({ ...upright, shoulderTiltDeg: -5 }, makeBaseline(samples()));
    expect(result.status).toBe("bad");
    expect(result.shoulderDirection).toBe("right_high");
    expect(result.reasons).toContain("Right shoulder is too high");
  });

  it("extracts a raised left shoulder without a 180-degree angle wrap", () => {
    const features = extractFeatures(pose(0.45, 0.55), 1000, 1000)!;
    expect(features.shoulderTiltDeg).toBeGreaterThan(0);
    expect(assess({ ...upright, shoulderTiltDeg: features.shoulderTiltDeg }, makeBaseline(samples())).shoulderDirection)
      .toBe("left_high");
  });

  it("extracts a raised right shoulder from raw anatomical landmarks", () => {
    const features = extractFeatures(pose(0.55, 0.45), 1000, 1000)!;
    expect(features.shoulderTiltDeg).toBeLessThan(0);
    expect(assess({ ...upright, shoulderTiltDeg: features.shoulderTiltDeg }, makeBaseline(samples())).shoulderDirection)
      .toBe("right_high");
  });

  it("does not score a major camera distance change as bad posture", () => {
    const result = assess({ ...upright, shoulderWidthNormalized: 0.55 }, makeBaseline(samples()));
    expect(result.positionChanged).toBe(true);
  });
});
