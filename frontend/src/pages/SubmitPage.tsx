import { useEffect, useRef, useState } from "react"
import { createTask, fetchTask, createBatch, fetchBatch, fetchDefectClasses } from "../lib/api"
import type { DefectClassMeta } from "../lib/api"
import type { Task, Batch, MatchedRule } from "../types"
import { formatDefectTypes } from "../lib/defectLabels"
import PoseSummary, { parsePoseResult } from "../components/PoseSummary"
import VarietySummary, { parseVarietyResult } from "../components/VarietySummary"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import {
  Tabs, TabsList, TabsTrigger,
} from "@/components/ui/tabs"
import {
  Upload, Image, Loader2, CheckCircle2, XCircle, Eye, Sparkles, FolderUp, Trash2, FileIcon,
} from "lucide-react"

type ProcessLog = {
  time: string
  stage: string
  message: string
  payload?: unknown
}

const statusLabel: Record<string, { text: string; style: string; icon: typeof CheckCircle2 }> = {
  pending: { text: "排队中", style: "bg-blue-50 border-blue-200 text-blue-700", icon: Loader2 },
  processing: { text: "流水线执行中", style: "bg-indigo-50 border-indigo-200 text-indigo-700", icon: Loader2 },
  pass: { text: "良品放行", style: "bg-emerald-50 border-emerald-200 text-emerald-700", icon: CheckCircle2 },
  block: { text: "不合格拦截", style: "bg-red-50 border-red-200 text-red-700", icon: XCircle },
  review: { text: "待人工复检", style: "bg-amber-50 border-amber-200 text-amber-700", icon: Eye },
  error: { text: "处理失败", style: "bg-slate-100 border-slate-200 text-slate-600", icon: XCircle },
}

const runningStatuses = new Set(["pending", "processing"])

const stageMeta: Record<string, { label: string; dot: string }> = {
  created: { label: "任务创建", dot: "bg-slate-400" },
  start: { label: "开始处理", dot: "bg-indigo-400" },
  variety: { label: "品种分类", dot: "bg-violet-500" },
  coarse: { label: "YOLO 粗筛", dot: "bg-cyan-500" },
  model: { label: "VLM 精查", dot: "bg-blue-500" },
  rule_match: { label: "策略匹配", dot: "bg-violet-500" },
  decision: { label: "分拣决策", dot: "bg-emerald-500" },
  pose: { label: "6D 位姿", dot: "bg-teal-500" },
  error: { label: "处理异常", dot: "bg-red-500" },
}

const IMAGE_EXTS = ".jpg,.jpeg,.png,.gif,.bmp,.webp,.tiff"

function SingleSubmit({ classes }: { classes: DefectClassMeta[] }) {
  const [file, setFile] = useState<File | null>(null)
  const [submitting, setSubmitting] = useState(false)
  const [result, setResult] = useState<Task | null>(null)
  const [polling, setPolling] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const pollTimerRef = useRef<number | null>(null)

  const stopPolling = () => {
    if (pollTimerRef.current) {
      window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
    setPolling(false)
  }

  useEffect(() => () => stopPolling(), [])

  const poseReady = (task: Task) => {
    if (task.status === "error") return true
    if (runningStatuses.has(task.status)) return false
    const raw = task.pose_result?.trim()
    if (!raw) return false
    try {
      const p = JSON.parse(raw) as { success?: boolean; message?: string }
      return p.success === true || Boolean(p.message)
    } catch {
      return Boolean(raw)
    }
  }

  const startPolling = (taskId: number) => {
    stopPolling()
    setPolling(true)
    pollTimerRef.current = window.setInterval(async () => {
      try {
        const latest = await fetchTask(taskId)
        setResult(latest)
        if (poseReady(latest)) stopPolling()
      } catch { stopPolling() }
    }, 1200)
  }

  const handleSubmit = async () => {
    if (!file) return
    setSubmitting(true)
    stopPolling()
    setResult(null)
    const fd = new FormData()
    fd.append("content_type", "image")
    fd.append("file", file)
    try {
      const task = await createTask(fd)
      setResult(task)
      if (runningStatuses.has(task.status)) startPolling(task.id)
    } finally { setSubmitting(false) }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    const f = e.dataTransfer.files[0]
    if (f) setFile(f)
  }

  const parseRules = (s: string): MatchedRule[] => {
    if (!s) return []
    try { return JSON.parse(s) } catch { return [] }
  }
  const parseLogs = (s: string): ProcessLog[] => {
    if (!s) return []
    try { return JSON.parse(s) } catch { return [] }
  }

  const st = result ? statusLabel[result.status] : null
  const logs = result ? parseLogs(result.process_logs) : []

  return (
    <div className="space-y-6">
      <div className="rounded-lg border border-slate-200 bg-slate-50/80 p-4 text-sm text-slate-600">
        <p className="font-medium text-slate-800 mb-2">支持的缺陷类别（shrish23/screw-dataset）</p>
        <div className="flex flex-wrap gap-2">
          {classes.map((c) => (
            <Badge key={c.id} variant="outline" className={c.is_defect ? "border-amber-200 bg-amber-50/50" : "border-emerald-200 bg-emerald-50/50"}>
              {c.label_zh} <span className="text-slate-400 ml-1 font-mono text-[10px]">{c.id}</span>
            </Badge>
          ))}
        </div>
      </div>

      <div
        className="border-2 border-dashed border-slate-200 rounded-xl p-8 text-center hover:border-indigo-300 hover:bg-indigo-50/30 transition-colors cursor-pointer"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input ref={inputRef} type="file" accept="image/*" className="hidden" onChange={(e) => setFile(e.target.files?.[0] || null)} />
        {file ? (
          <div className="space-y-2">
            <div className="w-12 h-12 mx-auto rounded-full bg-indigo-100 flex items-center justify-center">
              <Image className="w-6 h-6 text-indigo-600" />
            </div>
            <p className="font-medium text-slate-700">{file.name}</p>
            <p className="text-xs text-slate-400">{(file.size / 1024 / 1024).toFixed(2)} MB</p>
          </div>
        ) : (
          <div className="space-y-2">
            <Upload className="w-10 h-10 mx-auto text-slate-300" />
            <p className="text-slate-500">点击或拖拽螺丝图片到此处</p>
            <p className="text-xs text-slate-400">流水线：品种分类 → YOLO 粗筛 → VLM 精查 → 分拣 → 6D 位姿</p>
          </div>
        )}
      </div>

      <Button onClick={handleSubmit} disabled={submitting || !file} className="bg-indigo-600 hover:bg-indigo-700 px-8">
        {submitting ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> 提交中...</> : "提交质检"}
      </Button>

      {result && st && (
        <div className={`rounded-xl border p-5 space-y-4 ${st.style}`}>
          <div className="flex items-center gap-2">
            <st.icon className={`w-5 h-5 ${runningStatuses.has(result.status) ? "animate-spin" : ""}`} />
            <span className="font-bold text-base">{st.text}</span>
            {polling && <Badge variant="outline" className="bg-white/70">实时更新</Badge>}
            <span className="text-xs ml-auto opacity-70">任务 #{result.id}</span>
          </div>
          <div className="grid grid-cols-2 gap-3 text-sm">
            <div className="bg-white/60 rounded-lg p-3">
              <span className="text-xs opacity-60">置信度</span>
              <p className="font-mono font-bold text-lg">{(result.confidence * 100).toFixed(1)}%</p>
            </div>
            <div className="bg-white/60 rounded-lg p-3">
              <span className="text-xs opacity-60">缺陷类别</span>
              <p className="font-medium">{formatDefectTypes(result.violation_types)}</p>
            </div>
          </div>
          {!runningStatuses.has(result.status) ? (
            <div className="bg-white/90 rounded-xl p-4 text-sm border-2 border-violet-200/80 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-sm font-semibold text-violet-900">品种分类（MobileNet / EfficientNet）</span>
              </div>
              <VarietySummary variety={parseVarietyResult(result.variety_result || "")} />
            </div>
          ) : (
            <div className="bg-white/60 rounded-lg p-3 text-sm text-slate-500 flex items-center gap-2">
              <Loader2 className="w-4 h-4 animate-spin text-violet-600" />
              品种分类与质检流水线执行中…
            </div>
          )}
          {!runningStatuses.has(result.status) && (
            <div className="bg-white/90 rounded-xl p-4 text-sm border-2 border-teal-200/80 shadow-sm">
              <div className="flex items-center gap-2 mb-3">
                <span className="text-sm font-semibold text-teal-900">6D 位姿（机器人抓取 / 分拣）</span>
                {polling && !result.pose_result?.trim() && (
                  <Badge variant="outline" className="text-teal-700 border-teal-200">
                    <Loader2 className="w-3 h-3 mr-1 animate-spin" /> 位姿计算中
                  </Badge>
                )}
              </div>
              <PoseSummary pose={parsePoseResult(result.pose_result)} />
            </div>
          )}
          <div className="bg-white/60 rounded-lg p-3 text-sm">
            <div className="flex items-center gap-2 text-xs opacity-70 mb-3">
              <Sparkles className="w-3.5 h-3.5" /> 质检流水线日志
            </div>
            {logs.length === 0 ? (
              <p className="text-slate-500">暂无过程日志</p>
            ) : (
              <div className="space-y-0.5">
                {logs.map((log, i) => {
                  const meta = stageMeta[log.stage] || { label: log.stage || "未知阶段", dot: "bg-slate-400" }
                  const isLast = i === logs.length - 1
                  return (
                    <div key={i} className="relative pl-8 pb-4">
                      {!isLast && <span className="absolute left-[10px] top-5 h-full w-px bg-slate-200" />}
                      <span className={`absolute left-0 top-1.5 h-5 w-5 rounded-full border-2 border-white shadow-sm ${meta.dot}`} />
                      <div className="rounded-md border border-white/70 bg-white/75 p-2.5">
                        <div className="flex items-center gap-2">
                          <span className="text-xs font-medium text-slate-700">{meta.label}</span>
                          <span className="text-xs text-slate-500">{log.time}</span>
                        </div>
                        <p className="mt-1 text-slate-700">{log.message}</p>
                        {log.payload ? (
                          <pre className="mt-2 text-xs bg-slate-900 text-slate-100 rounded p-2 overflow-x-auto">{JSON.stringify(log.payload, null, 2)}</pre>
                        ) : null}
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
          </div>
          {result.risk_description && (
            <div className="bg-white/60 rounded-lg p-3 text-sm">
              <span className="text-xs opacity-60">检测说明</span>
              <p className="mt-1">{result.risk_description}</p>
            </div>
          )}
          {(() => {
            const rules = parseRules(result.matched_rules)
            if (rules.length === 0) return null
            return (
              <div className="bg-white/60 rounded-lg p-3 text-sm">
                <span className="text-xs opacity-60">命中分拣规则</span>
                <div className="mt-1 space-y-1">
                  {rules.map((r, i) => (
                    <div key={i} className="flex items-center gap-2">
                      <Badge variant="outline" className="text-xs">{r.action === "block" ? "拦截" : r.action === "pass" ? "放行" : "复检"}</Badge>
                      <span>{r.name}</span>
                      <span className="ml-auto text-xs opacity-60">相似度 {(r.similarity * 100).toFixed(0)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )
          })()}
        </div>
      )}
    </div>
  )
}

function BatchSubmit() {
  const [files, setFiles] = useState<File[]>([])
  const [submitting, setSubmitting] = useState(false)
  const [batchResult, setBatchResult] = useState<Batch | null>(null)
  const [polling, setPolling] = useState(false)
  const inputRef = useRef<HTMLInputElement>(null)
  const pollTimerRef = useRef<number | null>(null)

  const stopPolling = () => {
    if (pollTimerRef.current) {
      window.clearInterval(pollTimerRef.current)
      pollTimerRef.current = null
    }
    setPolling(false)
  }
  useEffect(() => () => stopPolling(), [])

  const startPolling = (batchId: number) => {
    stopPolling()
    setPolling(true)
    pollTimerRef.current = window.setInterval(async () => {
      try {
        const latest = await fetchBatch(batchId)
        setBatchResult(latest)
        if (latest.pending_count === 0) stopPolling()
      } catch { stopPolling() }
    }, 1500)
  }

  const handleFiles = (fileList: FileList | null) => {
    if (!fileList) return
    setFiles((prev) => [...prev, ...Array.from(fileList)])
  }

  const removeFile = (idx: number) => {
    setFiles((prev) => prev.filter((_, i) => i !== idx))
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    handleFiles(e.dataTransfer.files)
  }

  const handleSubmit = async () => {
    if (files.length === 0) return
    setSubmitting(true)
    stopPolling()
    setBatchResult(null)
    const fd = new FormData()
    files.forEach((f) => fd.append("files", f))
    try {
      const batch = await createBatch(fd)
      setBatchResult(batch)
      if (batch.pending_count > 0) startPolling(batch.id)
    } finally {
      setSubmitting(false)
    }
  }

  const batchStatusStyle = (status: string) =>
    statusLabel[status]?.style || "bg-slate-50 border-slate-200 text-slate-600"
  const batchStatusText = (status: string) =>
    statusLabel[status]?.text || status

  const progress = batchResult
    ? batchResult.total_count > 0
      ? Math.round((batchResult.done_count / batchResult.total_count) * 100)
      : 0
    : 0

  return (
    <div className="space-y-6">
      <div
        className="border-2 border-dashed border-slate-200 rounded-xl p-8 text-center hover:border-indigo-300 hover:bg-indigo-50/30 transition-colors cursor-pointer"
        onClick={() => inputRef.current?.click()}
        onDragOver={(e) => e.preventDefault()}
        onDrop={handleDrop}
      >
        <input
          ref={inputRef}
          type="file"
          multiple
          accept={IMAGE_EXTS}
          className="hidden"
          onChange={(e) => { handleFiles(e.target.files); e.target.value = "" }}
        />
        <FolderUp className="w-10 h-10 mx-auto text-slate-300" />
        <p className="text-slate-500 mt-2">批量上传螺丝图片（多选）</p>
        <p className="text-xs text-slate-400 mt-1">{IMAGE_EXTS}</p>
      </div>

      {files.length > 0 && (
        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <span className="text-sm font-medium text-slate-700">已选择 {files.length} 张图片</span>
            <Button variant="ghost" size="sm" onClick={() => setFiles([])} className="text-slate-400 hover:text-red-500">
              <Trash2 className="w-3.5 h-3.5 mr-1" /> 清空
            </Button>
          </div>
          <div className="bg-white rounded-xl border border-slate-100 divide-y divide-slate-50 max-h-[240px] overflow-y-auto">
            {files.map((f, i) => (
              <div key={i} className="flex items-center gap-3 px-4 py-2.5 text-sm">
                <Image className="w-4 h-4 text-indigo-500 shrink-0" />
                <span className="flex-1 truncate text-slate-700">{f.name}</span>
                <span className="text-xs text-slate-400 shrink-0">{(f.size / 1024 / 1024).toFixed(2)} MB</span>
                <button onClick={() => removeFile(i)} className="text-slate-300 hover:text-red-500 transition-colors">
                  <XCircle className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      <Button onClick={handleSubmit} disabled={submitting || files.length === 0} className="bg-indigo-600 hover:bg-indigo-700 px-8">
        {submitting ? <><Loader2 className="w-4 h-4 mr-2 animate-spin" /> 批量提交中...</> : `批量质检 (${files.length} 张)`}
      </Button>

      {batchResult && (
        <div className="rounded-xl border border-slate-200 bg-white p-5 space-y-4">
          <div className="flex items-center gap-3">
            <FolderUp className="w-5 h-5 text-indigo-500" />
            <span className="font-bold text-slate-800">{batchResult.name}</span>
            {polling && <Badge variant="outline" className="bg-indigo-50 text-indigo-600 border-indigo-200">处理中</Badge>}
            <span className="text-xs text-slate-400 ml-auto">批次 #{batchResult.id}</span>
          </div>

          <div>
            <div className="flex items-center justify-between text-xs text-slate-500 mb-1.5">
              <span>处理进度</span>
              <span>{batchResult.done_count} / {batchResult.total_count} ({progress}%)</span>
            </div>
            <div className="w-full bg-slate-100 rounded-full h-2">
              <div className="bg-indigo-500 h-2 rounded-full transition-all duration-500" style={{ width: `${progress}%` }} />
            </div>
          </div>

          <div className="flex flex-wrap gap-2">
            {batchResult.pass_count > 0 && (
              <Badge className="bg-emerald-50 text-emerald-700 border-emerald-200 hover:bg-emerald-50">
                <CheckCircle2 className="w-3 h-3 mr-1" /> 良品 {batchResult.pass_count}
              </Badge>
            )}
            {batchResult.block_count > 0 && (
              <Badge className="bg-red-50 text-red-700 border-red-200 hover:bg-red-50">
                <XCircle className="w-3 h-3 mr-1" /> 不合格 {batchResult.block_count}
              </Badge>
            )}
            {batchResult.review_count > 0 && (
              <Badge className="bg-amber-50 text-amber-700 border-amber-200 hover:bg-amber-50">
                <Eye className="w-3 h-3 mr-1" /> 待复检 {batchResult.review_count}
              </Badge>
            )}
          </div>

          <div className="bg-slate-50/50 rounded-lg border border-slate-100 overflow-hidden">
            <table className="w-full text-sm">
              <thead className="bg-slate-50">
                <tr>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium text-xs">文件名</th>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium text-xs">状态</th>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium text-xs">置信度</th>
                  <th className="text-left py-2 px-3 text-slate-500 font-medium text-xs">缺陷类别</th>
                </tr>
              </thead>
              <tbody>
                {batchResult.tasks.map((t) => (
                  <tr key={t.id} className="border-t border-slate-100 hover:bg-white/80">
                    <td className="py-2 px-3 text-slate-700 max-w-[200px] truncate">{t.original_filename || `#${t.id}`}</td>
                    <td className="py-2 px-3">
                      <Badge variant="outline" className={`text-[10px] ${batchStatusStyle(t.status)}`}>
                        {batchStatusText(t.status)}
                      </Badge>
                    </td>
                    <td className="py-2 px-3 font-mono text-xs">{(t.confidence * 100).toFixed(0)}%</td>
                    <td className="py-2 px-3 text-slate-500 text-xs truncate max-w-[160px]">{formatDefectTypes(t.violation_types)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default function SubmitPage() {
  const [mode, setMode] = useState<"single" | "batch">("single")
  const [classes, setClasses] = useState<DefectClassMeta[]>([])

  useEffect(() => {
    fetchDefectClasses().then((r) => setClasses(r.classes)).catch(() => {})
  }, [])

  return (
    <div className="p-6 space-y-6 max-w-4xl">
      <div>
        <h1 className="text-2xl font-bold text-slate-900">质检提交</h1>
        <p className="text-sm text-slate-500 mt-1">上传螺丝图片，查看质检流水线与 6D 位姿结果</p>
      </div>

      <Tabs value={mode} onValueChange={(v) => setMode(v as "single" | "batch")}>
        <TabsList>
          <TabsTrigger value="single" className="gap-1.5">
            <FileIcon className="w-3.5 h-3.5" /> 单件质检
          </TabsTrigger>
          <TabsTrigger value="batch" className="gap-1.5">
            <FolderUp className="w-3.5 h-3.5" /> 批量质检
          </TabsTrigger>
        </TabsList>
      </Tabs>

      {mode === "single" ? <SingleSubmit classes={classes} /> : <BatchSubmit />}
    </div>
  )
}
