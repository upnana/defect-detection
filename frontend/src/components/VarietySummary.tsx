import { Badge } from "@/components/ui/badge"

export type VarietyTopK = {
  class_id: string
  label_zh: string
  confidence: number
}

export type SortingStrategy = {
  strategy_id?: string
  grip_profile?: string
  target_bin?: string
  pose_profile?: string
  notes?: string
  class_id?: string
}

export type VarietyResult = {
  enabled?: boolean
  backbone?: string
  class_id?: string | null
  label_zh?: string | null
  confidence?: number
  top_k?: VarietyTopK[]
  sorting_strategy?: SortingStrategy
  low_confidence?: boolean
  message?: string
}

const backboneLabel: Record<string, string> = {
  mobilenet_v3_small: "MobileNetV3-Small",
  efficientnet_b0: "EfficientNet-B0",
}

export function parseVarietyResult(raw: string): VarietyResult | null {
  if (!raw?.trim()) return null
  try {
    return JSON.parse(raw) as VarietyResult
  } catch {
    return null
  }
}

export default function VarietySummary({ variety }: { variety: VarietyResult | null }) {
  if (!variety) {
    return <p className="text-sm text-slate-500">暂无品种分类数据</p>
  }
  if (!variety.enabled) {
    return <p className="text-sm text-slate-500">{variety.message || "品种分类未启用"}</p>
  }

  const strat = variety.sorting_strategy
  const bb = backboneLabel[variety.backbone || ""] || variety.backbone || "CNN"

  return (
    <div className="space-y-3 text-sm">
      <div className="flex flex-wrap gap-2 items-center">
        <Badge variant="outline" className="bg-violet-50 border-violet-200 text-violet-800">
          {bb} 品种分类
        </Badge>
        {variety.confidence != null && (
          <span className="text-slate-500">置信度 {(variety.confidence * 100).toFixed(1)}%</span>
        )}
        {variety.low_confidence && (
          <Badge variant="outline" className="text-amber-700 border-amber-200 bg-amber-50">
            低置信
          </Badge>
        )}
      </div>

      {variety.class_id ? (
        <div className="rounded-lg bg-violet-50/80 p-3 border border-violet-100">
          <p className="text-xs font-medium text-violet-900 mb-1">品类标签</p>
          <p className="font-semibold text-slate-800">
            {variety.label_zh}{" "}
            <span className="font-mono text-xs font-normal text-slate-500">({variety.class_id})</span>
          </p>
        </div>
      ) : (
        <p className="text-amber-700 text-sm">{variety.message || "未给出明确品类"}</p>
      )}

      {variety.top_k && variety.top_k.length > 0 && (
        <div className="text-xs text-slate-600 space-y-1">
          <p className="font-medium text-slate-500">Top-3 候选</p>
          {variety.top_k.map((t) => (
            <p key={t.class_id} className="font-mono">
              {t.label_zh} ({t.class_id}) — {(t.confidence * 100).toFixed(1)}%
            </p>
          ))}
        </div>
      )}

      {strat && (
        <div className="rounded-lg bg-slate-50 p-3 border border-slate-100 text-xs space-y-1">
          <p className="font-medium text-slate-600">分拣策略（架构图 Stage 0 输出）</p>
          <p>
            <span className="text-slate-500">strategy_id:</span>{" "}
            <span className="font-mono">{strat.strategy_id || "-"}</span>
          </p>
          <p>
            <span className="text-slate-500">夹爪:</span> {strat.grip_profile || "-"} ·{" "}
            <span className="text-slate-500">料箱:</span> {strat.target_bin || "-"}
          </p>
          {strat.notes && <p className="text-slate-500">{strat.notes}</p>}
        </div>
      )}
    </div>
  )
}
