import { useNavigate } from "react-router-dom"
import {
  ShieldCheck,
  Scan,
  Zap,
  GitBranch,
  BarChart3,
  ArrowRight,
  Sparkles,
  ChevronRight,
} from "lucide-react"

const defectClasses = [
  { id: "good", zh: "良好" },
  { id: "manipulated_front", zh: "正面被篡改" },
  { id: "scratch_head", zh: "头部划痕" },
  { id: "scratch_neck", zh: "颈部划痕" },
  { id: "thread_side", zh: "侧面螺纹缺陷" },
  { id: "thread_top", zh: "顶部螺纹缺陷" },
]

const pipeline = [
  { step: "01", title: "输入采集", desc: "产线相机采集螺丝图像，支持单件与批量上传" },
  { step: "02", title: "品种分类", desc: "MobileNetV3-Small 输出品类标签与分拣策略（可换 EfficientNet）" },
  { step: "03", title: "YOLOv8 粗筛", desc: "双阈值（0.15 召回 / 0.5 触发精查），明显良品毫秒级放行" },
  { step: "04", title: "VLM 精查", desc: "多模态对可疑件做六类缺陷确认，超时可回退粗筛结果" },
  { step: "05", title: "策略分拣", desc: "FAISS + 规则引擎决策：良品放行 / 不合格拦截 / 人工复检" },
  { step: "06", title: "6D 位姿", desc: "YOLOv8-Pose+solvePnP / TinyDETR-Pose；手眼标定输出机器人位姿" },
]

export default function HomePage() {
  const navigate = useNavigate()

  return (
    <div className="min-h-screen bg-[#f0f4f8]">
      <div className="fixed inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-32 -right-32 w-[500px] h-[500px] bg-cyan-200/30 rounded-full blur-[120px]" />
        <div className="absolute top-1/3 -left-40 w-[400px] h-[400px] bg-blue-200/25 rounded-full blur-[100px]" />
      </div>

      <header className="fixed top-0 left-0 right-0 z-50 bg-[#f0f4f8]/80 backdrop-blur-xl border-b border-slate-200/60">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-[#1a5276] to-cyan-600 flex items-center justify-center shadow-lg">
              <ShieldCheck className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl text-slate-800 tracking-tight">ScrewInspect</span>
          </div>
          <button
            onClick={() => navigate("/login")}
            className="px-5 py-2 text-sm font-medium text-white bg-gradient-to-r from-[#1a5276] to-cyan-600 rounded-full hover:shadow-lg transition-all"
          >
            进入控制台
          </button>
        </div>
      </header>

      <section className="relative pt-32 pb-16">
        <div className="max-w-7xl mx-auto px-6 text-center">
          <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-white/70 border border-cyan-200/60 mb-8">
            <Sparkles className="w-4 h-4 text-cyan-600" />
            <span className="text-sm font-medium text-slate-700">基于 shrish23/screw-dataset 六类缺陷</span>
          </div>
          <h1 className="text-5xl sm:text-6xl font-extrabold text-slate-900 leading-tight">
            螺丝缺陷
            <span className="bg-gradient-to-r from-[#1a5276] via-cyan-600 to-teal-500 bg-clip-text text-transparent"> 智能质检 </span>
            与分拣
          </h1>
          <p className="mt-6 text-lg text-slate-500 max-w-2xl mx-auto">
            面向产线混线场景的混检架构：YOLOv8 粗筛 + VLM 精查 + 可配置分拣策略，对齐 Hugging Face 螺丝分割数据集类别体系。
          </p>
          <button
            onClick={() => navigate("/login")}
            className="mt-10 inline-flex items-center gap-2 px-8 py-3.5 font-semibold text-white bg-gradient-to-r from-[#1a5276] to-cyan-600 rounded-full hover:shadow-xl transition-all"
          >
            开始质检 <ArrowRight className="w-4 h-4" />
          </button>
        </div>
      </section>

      <section className="relative py-12 max-w-5xl mx-auto px-6">
        <h2 className="text-2xl font-bold text-slate-900 text-center mb-8">缺陷类别</h2>
        <div className="grid sm:grid-cols-3 gap-3">
          {defectClasses.map((c) => (
            <div key={c.id} className="rounded-xl bg-white/80 border border-slate-100 p-4 text-left shadow-sm">
              <p className="font-semibold text-slate-800">{c.zh}</p>
              <p className="text-xs font-mono text-slate-400 mt-1">{c.id}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="relative py-16 max-w-4xl mx-auto px-6">
        <h2 className="text-2xl font-bold text-slate-900 text-center mb-10">质检流水线</h2>
        <div className="space-y-6">
          {pipeline.map((item) => (
            <div key={item.step} className="flex gap-5 items-start bg-white/70 rounded-2xl p-5 border border-white shadow-sm">
              <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[#1a5276] to-cyan-500 flex items-center justify-center text-white font-bold shrink-0">
                {item.step}
              </div>
              <div>
                <h3 className="text-lg font-bold text-slate-900">{item.title}</h3>
                <p className="text-slate-500 mt-1 text-sm leading-relaxed">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="relative py-12 max-w-5xl mx-auto px-6 grid sm:grid-cols-3 gap-4">
        {[
          { icon: Scan, title: "数据集对齐", desc: "类别与 screw-dataset 一致，便于训练 YOLO 分割/检测头" },
          { icon: Zap, title: "分层加速", desc: "粗筛放行正常件，仅可疑件进入 VLM，平衡速度与精度" },
          { icon: GitBranch, title: "分拣闭环", desc: "pass / block / review 三态输出，对接机械臂分拣扩展" },
        ].map((f) => (
          <div key={f.title} className="p-6 rounded-2xl bg-white/70 border border-slate-100">
            <f.icon className="w-8 h-8 text-cyan-600 mb-3" />
            <h3 className="font-bold text-slate-900">{f.title}</h3>
            <p className="text-sm text-slate-500 mt-2">{f.desc}</p>
          </div>
        ))}
      </section>

      <section className="relative py-16 px-6">
        <div className="max-w-3xl mx-auto text-center rounded-3xl bg-gradient-to-br from-white to-cyan-50 border p-12">
          <BarChart3 className="w-10 h-10 text-cyan-600 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-slate-800">登录控制台上传螺丝图片</h2>
          <p className="text-slate-500 mt-3">查看缺陷分布、批次进度与人工复检队列</p>
          <button
            onClick={() => navigate("/login")}
            className="mt-6 inline-flex items-center gap-1 px-6 py-3 text-white bg-[#1a5276] rounded-full"
          >
            立即登录 <ChevronRight className="w-4 h-4" />
          </button>
        </div>
      </section>

      <footer className="py-8 text-center text-sm text-slate-400 border-t border-slate-200/60">
        ScrewInspect · 螺丝缺陷智能质检系统
      </footer>
    </div>
  )
}
