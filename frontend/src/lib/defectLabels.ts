export const DEFECT_LABEL_ZH: Record<string, string> = {
  good: "良好",
  manipulated_front: "正面被篡改",
  scratch_head: "头部划痕",
  scratch_neck: "颈部划痕",
  thread_side: "侧面螺纹缺陷",
  thread_top: "顶部螺纹缺陷",
}

export function formatDefectTypes(raw: string): string {
  if (!raw) return "无"
  return raw
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean)
    .map((id) => DEFECT_LABEL_ZH[id] || id)
    .join("、")
}
