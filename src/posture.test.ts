import { describe, expect, it } from "vitest";
import { assess, makeBaseline, median, type Features } from "./posture";

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

  it("does not score a major camera distance change as bad posture", () => {
    const result = assess({ ...upright, shoulderWidthNormalized: 0.55 }, makeBaseline(samples()));
    expect(result.positionChanged).toBe(true);
  });
});
