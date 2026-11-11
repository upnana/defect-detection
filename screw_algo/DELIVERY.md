# screw_algo 离线包交付说明

面向接收方的 **算法离线包**（CLI + JSON），与 `backend/`、`frontend/` 无关。

## 交付范围

| 包含 | 说明 |
|------|------|
| 五段流水线 | 品种 → YOLO-OBB 粗筛 → VLM（可选）→ 规则 → 6D 位姿 |
| 权重 | `models/` 下 5 个 `.pt`（品种 / YOLO detect / OBB / YOLO-Pose / TinyDETR） |
| 标定占位 | `assets/*.json`（产线需替换相机内参、手眼、3D 关键点） |
| 可视化 | `--save-vlm-preview`、`--save-pose-preview` |
| VLM Prompt | `assets/vlm_screw_inspect_prompt.txt`（课件固定模板）+ 运行时附录 |
| Qwen-VL | `VLM_PROVIDER=qwen`，`core/vlm_qwen.py`（DashScope 兼容模式） |
| 文档 | `README.md`、本文件 |

| 不包含 | 说明 |
|--------|------|
| Web 控制台 | `frontend/`、`backend/` 任务 API |
| 完整数据集 | 训练数据需自行按 [screw-dataset](https://huggingface.co/datasets/shrish23/screw-dataset) 获取 |
| FAISS 策略库 | 离线为内存规则（`core/rules.py`） |
| 按品种多套 OBB | 路由已支持，当前各类共用 `screw_yolov8_obb.pt` |
| API 密钥 | 接收方复制 `.env.example` → `.env` 自行填写 |

## 打包前自检（交付方）

- [ ] `models/` 五个权重文件齐全（约 29MB）
- [ ] 未将 `screw_algo/.env` 打入压缩包
- [ ] 已附带 `.env.example`
- [ ] （建议）附带 `output/acceptance.json` 与 `runs/vlm_preview/`、`runs/pose_preview/` 样例图
- [ ] （建议）在 README 版本或 Git tag 上注明交付日期

## 一键打包（Windows）

在 **项目根目录**（`screw_algo` 的上一级）执行：

```powershell
powershell -ExecutionPolicy Bypass -File screw_algo\scripts\pack_offline_delivery.ps1
```

默认输出：`dist/ScrewInspect-screw_algo-offline.zip`  
可选参数：

```powershell
.\screw_algo\scripts\pack_offline_delivery.ps1 -OutZip "D:\交付\ScrewInspect-offline.zip"
```

脚本会 **排除**：`.env`、`runs/obb/`（训练产物）、`__pycache__`、`.pytest_cache` 等。

## 接收方环境

- Python **3.10–3.12**（3.13 若可用需自行验证）
- `pip install -r screw_algo/requirements.txt`
- GPU 可选；无 GPU 时 YOLO/位姿走 CPU
- Windows OpenMP 冲突：`$env:KMP_DUPLICATE_LIB_OK='TRUE'`

## 验收步骤

解压后目录结构应含 `screw_algo/pipeline.py`，且 **`screw_algo` 的父目录为工作根**。

```powershell
cd <工作根目录>
pip install -r screw_algo/requirements.txt

# 纯离线（无 VLM API）
python -m screw_algo "<图片路径>" --no-vlm ^
  --json screw_algo\output\acceptance.json ^
  --save-vlm-preview --save-pose-preview
```

**期望：**

| 检查项 | JSON / 文件 |
|--------|-------------|
| 品种 | `variety_classification.class_id`、`confidence` |
| 粗筛 | `coarse_screen.enabled`、`detector_mode`（OBB 时为 `yolov8_obb`） |
| 决策 | `decision.action` ∈ `pass` / `block` / `review` |
| 位姿 | `pose.success` 为 true，`pose_6d` 含平移与旋转 |
| 可视化 | `runs/vlm_preview/<图名>.jpg`、`runs/pose_preview/<图名>.jpg` |

### 含 VLM 验收（可选）

```powershell
copy screw_algo\.env.example screw_algo\.env
# 编辑 VLM_PROVIDER、QWEN_API_KEY 或 DEEPSEEK_* 
python -m screw_algo "<图片路径>" --json screw_algo\output\acceptance_vlm.json
```

VLM 默认超时 3s，超时回退粗筛结果。

## 产线替换项

1. `assets/default_camera_intrinsics.json` — 现场相机标定  
2. `assets/hand_eye_identity.json` — 手眼标定 4×4  
3. `assets/screw_keypoints_3d.json` — 与真实螺丝 CAD 一致时 6D 更准确  
4. `.env` — `VLM_*`、`YOLO_*` 阈值、 `POSE_MODE` 等  

## 常见问题

**Q: 提示找不到权重？**  
A: 确认 `screw_algo/models/*.pt` 已解压；路径相对 `screw_algo/`。

**Q: `--no-vlm` 仍很慢？**  
A: 首次加载 torch/ultralytics 较慢；品种 + YOLO + 位姿在 CPU 上约数秒/张。

**Q: 6D 叠图未生成？**  
A: 需 `pose.success=true`；检查 `YOLO_POSE_MODEL_PATH` 或 TinyDETR / 轮廓回退配置。

## 交付清单文件对照

| 路径 | 必需 |
|------|------|
| `core/`、`stages/`、`pipeline.py`、`settings.py`、`config.py` | 是 |
| `ml/`、`defect_types.py`、`__main__.py` | 是 |
| `models/*.pt` | 是 |
| `assets/` | 是 |
| `requirements.txt`、`.env.example`、`README.md`、`DELIVERY.md` | 是 |
| `scripts/train_*.py`、`convert_*.py` | 建议（复现 OBB/品种） |
| `scripts/pack_offline_delivery.ps1` | 建议 |
| `.env` | **否** |
| `runs/obb/` | **否** |
