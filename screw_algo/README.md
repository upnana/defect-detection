# screw_algo — 离线算法流水线

与 `backend/`、`frontend/` **完全独立**：推理、配置、权重与标定资源均在 `screw_algo/` 内，不 import `backend`、不 `chdir` 到 backend。

## 流程

```
图像
  ├─[0] MobileNet / EfficientNet 品种分类 → 品类标签、分拣策略 (models/screw_variety_*.pt)
  ├─[1] YOLOv8-OBB/检测 粗筛（按品种路由权重，0.15/0.5 双阈值）
  ├─[2] VLM 精查（画框图 + Qwen2.5-VL，JSON Schema，3s 超时回退；可 --no-vlm）
  ├─[3] 规则分拣 pass | block | review（内存规则，无 DB）
  └─[4] 6D 位姿 auto: YOLO-Pose+PnP → TinyDETR-Pose → 轮廓回退(若开启)
```

输出 JSON 含 `variety_classification`、`sorting_strategy`、`pose_6d`（平移 XYZ mm + 欧拉/四元数/yaw-pitch-roll）。

## 模型与权重

训练数据均来自 Hugging Face [shrish23/screw-dataset](https://huggingface.co/datasets/shrish23/screw-dataset)（或由其转换的 YOLO 目录 `data/yolo-screw`、`data/yolo-screw-pose`）。**交付包内 `models/` 下的 `.pt` 均为在本项目数据上训练/微调后的权重，不是未微调的官方预训练包直接推理。**

| 阶段 | 推理模型 | 默认权重文件 | 预训练来源 | 本项目微调 | 训练脚本（复现用） |
|------|----------|--------------|------------|------------|-------------------|
| [0] 品种分类 | **MobileNetV3-Small**（`torchvision`，可选 EfficientNet-B0） | `models/screw_variety_mobilenet.pt` | ImageNet 分类 backbone | 在 screw-dataset 六类文件夹上微调全连接头（替换 classifier） | `screw_algo/scripts/train_variety_classifier.py` |
| [1] 粗筛 | **YOLOv8-OBB**（`auto` 有 OBB 权重则用 OBB，否则 detect/seg） | `screw_yolov8.pt` / `screw_yolov8_obb.pt` | COCO `yolov8n-obb.pt` 等 | mask→OBB 标签微调；按品种见 `assets/variety_yolo_models.json` | `scripts/train_screw_yolo_obb.py` |
| [2] VLM 精查 | **Qwen2.5-VL**（默认，`core/vlm_qwen.py`） | — | 厂商预训练 | 画框输入 + **`assets/vlm_screw_inspect_prompt.txt`** + Pydantic Schema | `VLM_PROVIDER=qwen`，`QWEN_API_KEY` |
| [3] 规则分拣 | 内存规则引擎 | — | — | 与 backend 种子策略一致的 pass / block / review 阈值 | — |
| [4a] 6D 路径 1 | **YOLOv8n-pose** + 3D 角点 **solvePnP** | `models/screw_yolov8_pose.pt` | COCO 预训练 `yolov8n-pose.pt` | 在 `data/yolo-screw-pose` 上微调关键点检测 | `backend/scripts/train_screw_yolo_pose.py` |
| [4b] 6D 路径 2 | **TinyDETR-Pose**（MobileNetV3-Small backbone + 回归头，输出 t + rot6d） | `models/screw_tinydetr_pose.pt` | ImageNet 预训练 backbone（见 `ml/tinydetr_pose_net.py`） | 在 screw-dataset 图像上用轮廓 PnP **伪标签**监督训练回归头 | `backend/scripts/train_tinydetr_pose.py` |
| [4c] 轮廓回退（可选） | OpenCV 阈值 + minAreaRect + PnP | — | — | 非深度学习，无权重 | — |

**课件对齐要点：** 品种低置信 → `variety_pending`（品种待确认）+ **通用检测头**；VLM 默认 **3s** 超时回退粗筛。

**VLM 默认：** `QWEN_MODEL=qwen2.5-vl-72b-instruct`（DashScope Key）；可改 DeepSeek / 豆包 Ark。

### 训练 YOLOv8-OBB 粗筛

```powershell
python screw_algo/scripts/convert_screw_masks_to_yolo_obb.py
python screw_algo/scripts/train_screw_yolo_obb.py --epochs 40
```

**几何先验（非神经网络）**：`assets/screw_keypoints_3d.json` 为螺丝头部/杆尖 3D 模型（mm）；`assets/default_camera_intrinsics.json`、`hand_eye_identity.json` 为默认标定占位，产线需替换。

### 训练品种分类器

数据目录与 [screw-dataset](https://huggingface.co/datasets/shrish23/screw-dataset) 一致：`<data>/<class_id>/images/*.png`。

```powershell
cd <项目根目录>
python screw_algo/scripts/train_variety_classifier.py ^
  --data data/screw-dataset/dataset ^
  --out screw_algo/models/screw_variety_mobilenet.pt ^
  --backbone mobilenet_v3_small --epochs 25

# EfficientNet-B0（稍大、略慢）
python screw_algo/scripts/train_variety_classifier.py ^
  --data data/screw-dataset/dataset ^
  --out screw_algo/models/screw_variety_efficientnet.pt ^
  --backbone efficientnet_b0 --epochs 25
```

训练完成后在 `.env` 中设置 `VARIETY_MODEL_PATH` 与 `VARIETY_BACKBONE`。分拣策略映射见 `assets/variety_sort_strategies.json`（产线多 SKU 时可扩展 `by_class_id`）。

## 运行

在项目根目录：

```powershell
$env:KMP_DUPLICATE_LIB_OK='TRUE'
cd "d:\project\Defect Detection"
python -m screw_algo "data\screw-dataset\dataset\scratch_neck\images\060.png" --json screw_algo\output\demo.json
```

仅粗筛（无 API）：

```powershell
python -m screw_algo "path\to\image.png" --no-vlm
```

可视化导出（与课件一致：VLM 画框 / 6D 绿框+RGB 轴）：

```powershell
python -m screw_algo "path\to\image.png" --no-vlm --save-vlm-preview --save-pose-preview
# 默认输出 screw_algo/runs/vlm_preview/<图名>.jpg 与 runs/pose_preview/<图名>.jpg
```

首次使用：复制 `screw_algo/.env.example` → `screw_algo/.env`，填入 VLM API Key（若需精查）。权重默认在 `screw_algo/models/`。

安装依赖（在任意虚拟环境中）：

```powershell
pip install -r screw_algo/requirements.txt
```

说明：`--no-vlm` 时仍需 YOLO 与 OpenCV；TinyDETR 位姿路径需要 `torch` / `torchvision`。Windows 若遇 OpenMP 冲突可设 `$env:KMP_DUPLICATE_LIB_OK='TRUE'`。

## 交付清单

复制 **`screw_algo` 整个目录**（或打成 zip）即可作为**离线算法模块**交付；解压后 **`screw_algo` 的上一级** 为工作根目录，在该目录执行 `python -m screw_algo ...`。

**打包与验收细则见 [`DELIVERY.md`](DELIVERY.md)。** 在项目根目录一键打包：

```powershell
powershell -ExecutionPolicy Bypass -File screw_algo\scripts\pack_offline_delivery.ps1
# 输出 dist/ScrewInspect-screw_algo-offline.zip（自动排除 .env、runs/obb/）
```

| 项 | 路径 / 动作 | 备注 |
|----|-------------|------|
| 源码 | `core/`、`stages/`、`pipeline.py`、`settings.py` 等 | 不依赖 `backend/` |
| 权重 | 上表四个 `.pt`（约 23MB） | 均为 screw-dataset 相关数据微调/训练结果，交付时应一并打包 |
| 标定资源 | `assets/*.json` | 产线需替换为现场相机内参与手眼标定 |
| 配置模板 | `.env.example` | 交付时附带；**勿**提交含密钥的 `.env` |
| 依赖说明 | `requirements.txt` | 接收方 `pip install -r screw_algo/requirements.txt` |
| 文档 | 本 README | 含流程、CLI、交付边界 |
| 可选 | `output/` 下脱敏 JSON 样例、1～2 张试跑图片 | 便于验收 |

**交付范围（建议在合同/说明中写清）**

- 包含：单张图 CLI 推理 → JSON（品种分类 + 粗筛、可选 VLM、分拣 `pass|block|review`、6D 位姿）。
- 不包含：Web 界面、完整数据集；YOLO/位姿训练脚本在 `backend/scripts/`（复现用）。**含** `screw_algo/scripts/train_variety_classifier.py`。
- 数据集：训练见 [shrish23/screw-dataset](https://huggingface.co/datasets/shrish23/screw-dataset)，使用须遵守其许可。
- VLM：接收方自备 API Key；纯离线验收可用 `--no-vlm`（仅用 YOLO + 规则 + 位姿）。

**验收示例**

```powershell
cd <交付根目录>
pip install -r screw_algo/requirements.txt
python -m screw_algo "<图片绝对或相对路径>" --no-vlm --json screw_algo/output/acceptance.json
```

期望：JSON 含 `variety_classification`、`coarse_screen`、`decision.action`、`pose.success` / `pose_6d`。

## 目录

| 路径 | 作用 |
|------|------|
| `settings.py` | `screw_algo/.env` + 默认路径 |
| `core/` | 品种分类、粗筛、VLM、规则、位姿 |
| `ml/variety_net.py` | MobileNetV3 / EfficientNet 分类头 |
| `scripts/train_variety_classifier.py` | 品种分类训练 |
| `models/` | 品种 / YOLO / TinyDETR 权重 |
| `assets/` | 内参、3D 关键点、手眼、分拣策略 JSON |
| `pipeline.py` | 五阶段编排与计时 |
| `stages/` | 对 `core` 的薄封装 |
| `requirements.txt` | 离线推理 Python 依赖 |
| `output/` | 建议存放 `--json` 结果 |
