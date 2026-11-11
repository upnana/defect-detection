import { Badge } from "@/components/ui/badge"

export type Pose6D = {
  frame?: string
  translation_mm?: { x?: number; y?: number; z?: number; unit?: string }
  rotation?: {
    euler_xyz_deg?: { rx?: number; ry?: number; rz?: number }
    yaw_pitch_roll_deg?: { yaw?: number; pitch?: number; roll?: number }
    quaternion_wxyz?: { w?: number; x?: number; y?: number; z?: number }
  }
  vector_6d?: number[]
  vector_6d_desc?: string
}

export type PoseResult = {
  success?: boolean
  method?: string
  confidence?: number
  reprojection_error_px?: number
  message?: string
  pose_6d?: Pose6D
  pose_6d_robot?: Pose6D
  pose_6d_camera?: Pose6D
  camera_frame?: {
    translation_mm?: number[]
    rotation_deg?: number[]
  }
  robot_frame?: {
    translation_mm?: number[]
    rotation_deg?: number[]
  }
}

const methodLabel: Record<string, string> = {
  yolo_pnp: "路径1 · YOLOv8-Pose + solvePnP",
  tinydetr_pose: "路径2 · TinyDETR-Pose 端到端",
  contour_pnp: "开发回退 · 轮廓 + solvePnP",
  disabled: "已关闭",
}

export function parsePoseResult(raw: string): PoseResult | null {
  if (!raw) return null
  try {
    return JSON.parse(raw) as PoseResult
  } catch {
    return null
  }
}

export default function PoseSummary({ pose }: { pose: PoseResult | null }) {
  if (!pose) {
    return <p className="text-sm text-slate-500">暂无位姿数据</p>
  }
  if (!pose.success) {
    return (
      <p className="text-sm text-amber-700">
        {pose.message || "位姿估计未成功"}
        {pose.method ? ` (${pose.method})` : ""}
      </p>
    )
  }

  const cam = pose.camera_frame
  const rob = pose.robot_frame
  const p6 = pose.pose_6d_robot || pose.pose_6d
  const t = p6?.translation_mm
  const e = p6?.rotation?.euler_xyz_deg
  const ypr = p6?.rotation?.yaw_pitch_roll_deg
  const q = p6?.rotation?.quaternion_wxyz

  return (
    <div className="space-y-3 text-sm">
      <div className="flex flex-wrap gap-2 items-center">
        <Badge variant="outline" className="bg-cyan-50 border-cyan-200 text-cyan-800">
          {methodLabel[pose.method || ""] || pose.method}
        </Badge>
        {pose.confidence != null && (
          <span className="text-slate-500">置信度 {(pose.confidence * 100).toFixed(0)}%</span>
        )}
        {pose.reprojection_error_px != null && (
          <span className="text-slate-400 text-xs">重投影误差 {pose.reprojection_error_px.toFixed(2)} px</span>
        )}
      </div>
      {p6 && (
        <div className="rounded-lg bg-cyan-50/80 p-3 border border-cyan-100 space-y-2">
          <p className="text-xs font-medium text-cyan-900">6D 位姿（{p6.frame || "robot_base"}）</p>
          {p6.vector_6d && (
            <p className="font-mono text-sm font-semibold text-slate-800 bg-white/70 rounded px-2 py-1.5 border border-cyan-100">
              [tx, ty, tz, rx, ry, rz] = [{p6.vector_6d.map((v) => v.toFixed(2)).join(", ")}]
              <span className="block text-[10px] font-normal text-slate-500 mt-0.5">
                {p6.vector_6d_desc || "mm + deg"}
              </span>
            </p>
          )}
          <p className="font-mono text-xs text-slate-700">
            平移 X/Y/Z (mm): {t?.x ?? "-"}, {t?.y ?? "-"}, {t?.z ?? "-"}
          </p>
          <p className="font-mono text-xs text-slate-700">
            欧拉 rx/ry/rz (°): {e?.rx ?? "-"}, {e?.ry ?? "-"}, {e?.rz ?? "-"}
          </p>
          <p className="font-mono text-xs text-slate-600">
            yaw / pitch / roll (°): {ypr?.yaw ?? "-"}, {ypr?.pitch ?? "-"}, {ypr?.roll ?? "-"}
          </p>
          <p className="font-mono text-xs text-slate-600 break-all">
            四元数 w,x,y,z: {q?.w ?? "-"}, {q?.x ?? "-"}, {q?.y ?? "-"}, {q?.z ?? "-"}
          </p>
        </div>
      )}
      <div className="grid sm:grid-cols-2 gap-2">
        <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
          <p className="text-xs font-medium text-slate-500 mb-1">相机坐标系 (mm / °)</p>
          <p className="font-mono text-xs leading-relaxed">
            t: [{cam?.translation_mm?.map((v) => v.toFixed(1)).join(", ") || "-"}]
            <br />
            R: [{cam?.rotation_deg?.map((v) => v.toFixed(1)).join(", ") || "-"}]
          </p>
        </div>
        <div className="rounded-lg bg-slate-50 p-3 border border-slate-100">
          <p className="text-xs font-medium text-slate-500 mb-1">机器人基坐标系 (mm / °)</p>
          <p className="font-mono text-xs leading-relaxed">
            t: [{rob?.translation_mm?.map((v) => v.toFixed(1)).join(", ") || "-"}]
            <br />
            R: [{rob?.rotation_deg?.map((v) => v.toFixed(1)).join(", ") || "-"}]
          </p>
        </div>
      </div>
    </div>
  )
}
