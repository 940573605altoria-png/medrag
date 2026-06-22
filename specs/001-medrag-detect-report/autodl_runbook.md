# AutoDL 开工 Runbook —— 环境 + 模型冒烟

目标：在 AutoDL GPU 实例上把链路跑通——**装环境 → 下权重/数据 → 加载 Qwen3-VL 基座 →
一次推理 + 骨架管线在 GPU 上跑通**。这是 US1 训练前的最小闭环。

> 工作流：本地（Windows）写代码 + 推到 git；AutoDL 实例 `git pull` 后按本手册执行；
> 输出/报错贴回给我调。我**不能直接连**你的 AutoDL 实例。

---

## 0. 两种实例档位

**A. 真跑 30B（最终）**：基座 Qwen3-VL 30B-A3B（MoE），bf16 权重约 **60GB**。
- GPU：单卡 **A100/H100 80GB** bf16 直接跑（推理/LoRA 都够）；48GB 卡用 `--quant 4bit`/FP8；多卡 `device_map=auto` 分片、训练自动 ZeRO-2。
- 数据盘 `/root/autodl-tmp` ≥ **120GB**（权重 60G + 数据 + 缓存）。

**B. 当前 4090 验证代码路径（先做这个）**：实测 **RTX 4090 24GB / 数据盘 50GB / 裸镜像无 torch**。
30B 在此放不下，所以**用小号 dense 模型 `Qwen/Qwen3-VL-4B-Instruct`（~8GB）验证加载器/管线/脚本全通**，
锁定的 30B 基座不改，留给升级后的实例真跑。4B 在 24GB/50GB 上很宽裕。
> 镜像无 torch 也没关系——`autodl_setup.sh` 会自动装匹配 CUDA 的 torch（默认 `cu124`，适配 Ada/4090）。

## 1. 探测当前实例（先跑这个，把输出贴回）

```bash
echo "=== GPU ==="; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo "=== CUDA/torch ==="; python -c "import torch;print('torch',torch.__version__,'cuda',torch.version.cuda,'avail',torch.cuda.is_available(),'n_gpu',torch.cuda.device_count())" 2>&1
echo "=== transformers ==="; python -c "import transformers;print('transformers',transformers.__version__)" 2>&1
echo "=== python/conda ==="; python --version; which conda; conda --version 2>&1
echo "=== disk ==="; df -h /root/autodl-tmp 2>/dev/null; df -h /root
```

> 据此确认：torch 是否预装、CUDA 版本、显存够不够 bf16、transformers 是否够新
> （需含 `Qwen3VLMoeForConditionalGeneration`，约 transformers ≥ 4.46；不够则 setup 会升级）。

## 2. 拉代码

```bash
cd /root/autodl-tmp
git clone <你的仓库地址> medrag && cd medrag
# 或已 clone： git pull
```

## 3. 装环境

```bash
bash scripts/autodl_setup.sh
# 4090 裸镜像默认装 cu124 torch；如驱动/卡需要别的 CUDA 版：
#   TORCH_CUDA=cu121 bash scripts/autodl_setup.sh
```

做了：学术加速 + HF 镜像、建数据盘目录、把缓存引到数据盘、**校验/自动安装 torch**（裸镜像）、
装 medrag 包 + transformers/accelerate/peft/modelscope/pillow，最后跑 **L0 冒烟**（环境 + 骨架管线，
不下大模型）。看到 `L0 PASS` 即环境 OK。

把环境变量写进 `~/.bashrc` 长期生效（可选）：
```bash
cat >> ~/.bashrc <<'EOF'
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export MODELSCOPE_CACHE=/root/autodl-tmp/.cache/modelscope
EOF
```

## 4. 下权重

**B 档（当前 4090，下小模型 ~8GB）：**
```bash
MEDRAG_BASE_MODEL_MS=Qwen/Qwen3-VL-4B-Instruct bash scripts/download_assets.sh weights
export MEDRAG_BASE_MODEL=/root/autodl-tmp/weights/Qwen3-VL-4B-Instruct
```
**A 档（升级实例后，下 30B ~60GB，可后台跑）：**
```bash
bash scripts/download_assets.sh weights
export MEDRAG_BASE_MODEL=/root/autodl-tmp/weights/Qwen3-VL-30B-A3B-Instruct
```

## 5. 模型冒烟（L1）

```bash
# B 档（4090，4B）——也可不下权重直接在线拉：
python scripts/smoke_gpu.py --with-model --model-id Qwen/Qwen3-VL-4B-Instruct
# A 档（80GB 卡，30B bf16）：
python scripts/smoke_gpu.py --with-model
# 显存紧（48GB 卡跑 30B）：
python scripts/smoke_gpu.py --with-model --quant 4bit
```

看到 `L1 PASS` + 一段推理输出 + `peak VRAM` 即基座加载/推理通。**本批目标达成**
（4B 验证了加载器/管线/脚本链路；30B 真跑在升级实例后同命令、换 model-id 即可）。

## 6.（可选，US1 预备）下数据集

```bash
bash scripts/download_assets.sh data
```
- b（lavita 医学QA）、c（MedTrinity-25M demo）经 HF datasets 下载。
- a（kaggle 药品）需先放 `~/.kaggle/kaggle.json`（kaggle API token），否则自动跳过。

> ⚠️ **US1 前必查**：下完 c 后看几条样本，确认病灶标注是"图上绿框"还是"bbox 坐标"。
> 若是坐标，创新点 C 的提框入口（T024）改为直接读坐标，其余（inpaint/热图/防泄露）不变。

---

## 故障速查

| 现象 | 处理 |
|---|---|
| `Qwen3VLMoeForConditionalGeneration` 不存在 | transformers 太旧 → `pip install -U transformers`（loader 已回退到通用类，但 MoE 类最稳） |
| OOM 加载失败 | 换 `--quant 4bit`，或用 FP8 权重 `MEDRAG_BASE_MODEL=Qwen/Qwen3-VL-30B-A3B-Instruct-FP8`，或多卡 |
| 下载龟速/断 | 确认 `source /etc/network_turbo`；权重用 ModelScope、数据用 hf-mirror；断点重跑（modelscope 支持续传） |
| `trust_remote_code` 提示 | 已默认开启；如仍报错升级 transformers |
| flash-attn 报错 | 默认不开；需要时 `pip install flash-attn` 后传 `attn_implementation="flash_attention_2"` |
