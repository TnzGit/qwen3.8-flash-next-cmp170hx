# Strata 借鉴评审与本地测试交接

评审日期：2026-10-02。状态：**实验代码，默认关闭；没有 CMP170HX 性能实测，不应直接用于生产。**

## 结论与边界

保留用户已验证的单张 SM80 CMP170HX 64 GiB、约 48 GB 系统内存、主体权重全驻显存、BF16 PLE 在 NVMe、单并发 MTP=3、解码超过 150 tok/s 的部署。用户的数字是本轮基线，不用仓库默认 MTP=1 或 Strata 的异机数据替换它。

优先借鉴 **MTP 草稿专用的小词表 head**，其次借鉴 **按“最终接受 token / 整轮耗时”选择投机策略** 的评测方式。没有证据表明移植后一定有收益，更没有依据承诺提升几十个百分点。不要迁移 Strata 的整套引擎、GGUF 格式或 CPU 专家卸载路径。

本分支只新增文件，不修改原始 `patches/qwen38-ple-ssd.patch`、`scripts/serve.sh`、模型权重、现有结果或生产默认配置。安装器仅在显式 `--apply` 后修改另一个本地 vLLM checkout 的两处：新增实验模块；在 MTP 加载完并完成 head 共享后增加 4 行 hook。关闭时不导入实验模块，也不包装 logits processor。

## 审计基线

| 项目 | 固定版本 |
| --- | --- |
| 本 fork / iIIusi0n 原仓库 | `c7fe762c415c9f53fabccdadc25be949eb92052f`；本轮比较时 main 一致 |
| vLLM | `a5a30471ff2bb7f0824f2da10e358af98d304472` |
| Strata | `9259cad4cfa3543cd3b8decab5962672b968c649` |
| vLLM MTP speculator 原始 Git blob | `d94702612d9ba01f033e751592f93881dc991f6d` |

安装器同时核对 vLLM HEAD 和被修改文件的原始 Git blob，不会自动 checkout、reset、升级或“修复”版本漂移。现有 PLE 补丁可以是未提交修改；只要没有改动本实验涉及的文件，不要求整个 checkout 干净。测试 fixture 的 Git blob 已与上述原始文件核对，但这不是一次完整 vLLM 安装或集成测试。

## 哪些值得借，哪些不值得

| 方向 | 当前判断 | 理由 / 下一步 |
| --- | --- | --- |
| 只缩小 MTP draft 的物理 LM head | 优先实验 | 每个 draft 步少读 head 权重；目标词表和验证器不变。需测中文/代码接受率、额外显存和整机速度。 |
| 投机深度/lookup 的“接受 token / 毫秒”成本模型 | 优先借鉴方法，不直接搬控制器 | 第一轮仍固定 MTP=3，隔离变量；之后再测静态深度 1/2/3/4。动态策略涉及调度、图形状和状态回退，不能算四行小补丁。 |
| SM80 的小 batch GEMM/GEMV 调优 | 待 profile，第二阶段 | 固定版本 Qwen4Exp 的专用低延迟 GEMM plans 只有 SM90/SM103，并非 SM80。先测实际 head 和其他投影占比，不代表 vLLM 的所有 SM80 路径都未优化。 |
| PLE 残余 host 等待 / 缓存预算 | 条件性考虑 | 当前方案已经有 native AIO、精确行去重/LRU、pinned 缓冲、CUDA stream overlap 和 prompt read-ahead。必须先看到未被隐藏的等待，不能把重复实现当新增收益。 |
| CPU/GPU 专家分层、专家 NVMe lookahead | 不移植 | 用户的主体权重已全部进显存；这解决的是另一种内存瓶颈。 |
| PLE/KV 再量化、speed projection | 不纳入本轮 | 引入数值或模型行为变化；与保持当前模型输出路径的目标不符。 |

源码依据（均为固定版本）：

- [当前 PLE 实现说明](https://github.com/TnzGit/qwen3.8-flash-next-cmp170hx/blob/c7fe762c415c9f53fabccdadc25be949eb92052f/docs/IMPLEMENTATION.md)：95.37 GiB BF16 表、320-byte 行、native AIO、去重/LRU、传输重叠和两个 host break。
- [Strata 的 draft 子词表工具](https://github.com/Niko1221/Strata/blob/9259cad4cfa3543cd3b8decab5962672b968c649/tools/draft_vocab.py)及 [MTP 实现](https://github.com/Niko1221/Strata/blob/9259cad4cfa3543cd3b8decab5962672b968c649/src/core/mtp.cpp)：实际选取 head 的行，而非全词表 GEMM 后再 mask。
- [Strata DETAILS](https://github.com/Niko1221/Strata/blob/9259cad4cfa3543cd3b8decab5962672b968c649/docs/DETAILS.md)：旧 en/code 子集 40,525 IDs，仅包含 55,328 个 Han tokens 中的 27 个；补齐 CJK 后子集 106,299 IDs。其 15–38% 是 Q2_0 / RTX5070 上修复自己旧子集后的 CJK 增益，**不是相对完整词表 vLLM 的增益**。
- [Strata 成本策略](https://github.com/Niko1221/Strata/blob/9259cad4cfa3543cd3b8decab5962672b968c649/src/spec/draft_policy.cpp)：观察整轮耗时与接受量，决定 lookup 与 MTP；初始成本曲线含特定硬件先验，不能原样套到 CMP170HX。
- [固定版本 Qwen4ExpMTP](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/models/qwen4_exp/nvidia/mtp.py)：完整词表 ParallelLMHead / LogitsProcessor。
- [固定版本 head 共享](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/v1/worker/gpu/spec_decode/eagle/utils.py)：加载 draft 后替换为目标模型 head；实验必须在其后准备副本。
- [固定版本 draft 采样](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/v1/worker/gpu/spec_decode/speculator.py)与 [logits processor](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/model_executor/layers/logits_processor.py)：原型保留完整输出宽度和全局 ID，不改采样器/验证器。
- [固定版本 low_latency_gemm.py](https://github.com/vllm-project/vllm/blob/a5a30471ff2bb7f0824f2da10e358af98d304472/vllm/models/qwen4_exp/nvidia/low_latency_gemm.py)：只选择 SM90/SM103 专用 plans。

Strata 的速度表涉及不同 GPU、CPU、权重量化和 prompt；不做跨引擎 tok/s 排名。上述代码是借鉴思路的独立实现，没有移植 Strata kernel 或分发其词表文件。

## 实验如何工作

在新 GPU runner 的 `MTPSpeculator.load_draft_model()` 中，等待 `load_eagle_model()` 返回，再在 **draft 模型对象** 上包装 `logits_processor`：

1. 从已经共享好的 BF16 head 中，一次性复制选中行到连续 GPU buffer。
2. 每个 draft 步使用原有 unquantized method 和 logits processor 计算较短 head；保留原 soft-cap / scale 处理。
3. 将紧凑 logits 散射回完整词表宽度，未选中 ID 为 `-inf`，让现有 argmax 直接产生原始全局 token ID。

目标 head、目标 logits processor、输入 embedding、验证器、PLE、注意力/GDN 状态回退均不修改。子集之外的 token **仍可由目标模型输出**，但不能由这份 draft 提议，因此接受率可能下降。

这里保留 full-width scatter 是为了减少接线风险，并非最后的最优实现。先确定净收益，再考虑 compact argmax + ID 映射。没有引入新 CUDA kernel；实际 GEMM 形状改变仍可能改变浮点舍入，不能把“目标路径未改”表述成“已证明端到端 token 完全一致”。

### 显存并不减少

目标模型仍需要完整 head，当前 vLLM 又在目标/draft 间共享它，所以实验的子集是**额外副本**，不能释放原始 head。

若实际配置为 V=248,320、H=2,560，BF16 完整 head 约 1.184 GiB；106,299 行副本约 519 MiB。这里的 106,299 仅作容量示例，本工具生成的行数不保证与 Strata 相同。还要计入 ID、logits、CUDA graph pool 开销。Strata 原生量化 head 的显存数字不能照搬到 BF16。

full-vocab 控制组额外复制约 1.184 GiB，因此也要先检查显存余量；不要为跑实验驱逐主体权重、改量化或强行改变原来的 GPU memory utilization。若当前余量不足，停止实验并记录这一限制。测试缓存容量和生产长上下文是否受到影响。

### 有意限制的支持范围

仅支持本轮审计的 pinned vLLM **新 GPU runner / NVIDIA Qwen4ExpMTP / SM80 / TP=PP=DP=PCP=DCP=1 / BF16 非量化 head / greedy draft sampling**。不支持 probabilistic draft、local argmax reduction、adaptive verification、LoRA、watermark、运行中换权重或换 tokenizer。拒绝这些组合是保护，不应删除 guard 后直接生产运行。

`greedy draft` 与目标请求的 temperature 是两件事。第一轮用目标 temperature=0 做回归；真实随机采样仍需独立质量/分布与性能回归，不能要求不同投机执行的随机样本逐 token 一致。

**必须看到日志 `EXPERIMENTAL qwen4 MTP draft vocabulary ACTIVE`。** 本补丁只接新 runner；旧 runner 或不同安装路径可能根本没经过 hook。没有激活日志的运行是“未生效”，不是性能通过。启用/禁用均需要重启测试进程、重新捕获 CUDA graphs；不支持热切换。

## 给本地 agent 的操作步骤

只在维护窗口或独立测试实例运行；本分支不会停止现有服务。不要同时运行两个挤满同一 GPU 的服务，也不要在生产服务还占满显存时运行 head 微基准。

以下从本 fork 分支的根目录执行。将三个路径替换为现有部署路径，不安装/升级 vLLM、PyTorch、驱动或模型。

```bash
export VLLM_DIR=/path/to/the/pinned/vllm-checkout
export MODEL_DIR=/path/to/the/current/local/model
export PY=/path/to/the/current/vllm-venv/bin/python
export WORK=/tmp/qwen-strata-review
mkdir -p "$WORK"

# 1. CPU 回归；有 CUDA 时同一文件还会跑小型 graph replay 测试。
"$PY" -m pytest -q tests/test_strata_draft_vocab.py

# 2. 只检查，默认不写入；HEAD 或原始文件不匹配则停止。
"$PY" scripts/apply-strata-draft-vocab.py --vllm-dir "$VLLM_DIR" --check

# 3. 本地 tokenizer 绑定的两份 manifest，不下载模型或词表。
"$PY" experiments/strata_draft_vocab.py build \
  --model-dir "$MODEL_DIR" --mode full --out "$WORK/full.json"
"$PY" experiments/strata_draft_vocab.py build \
  --model-dir "$MODEL_DIR" --mode cjk-code --out "$WORK/cjk-code.json"
```

生成器以独占创建方式写文件；重复执行请使用新输出名，不要忽略报错。`cjk-code` 需要现有环境中的 `tokenizers`。它保留低 ID 的保守基集（默认前 65,536 个，**不是实测词频排序**）、ASCII、CJK、单 token 解码不完整的字节片段、special tokens；可用 `--corpus /path/to/local/calibration.txt` 增加常用 token。校准语料只在本地读取，不上传；校准集与最终评测集分开。生成策略实际保留多少行以命令输出为准，接近全词表时可能不值得继续。

manifest 包含完整 ID 列表、vocab size、tokenizer.json SHA256；启用配置还必须给出 manifest 自身 SHA256。禁止拿另一个模型/分词器的 map 直接使用。

### 先微基准，再决定是否安装

仅在 GPU 空闲且内存足够时运行：

```bash
"$PY" experiments/strata_draft_vocab.py bench \
  --model-dir "$MODEL_DIR" --manifest "$WORK/cjk-code.json" \
  --rows 1 4 --iters 100 > "$WORK/head-microbench.json"
```

这只分配配置同形状的**随机 BF16 head 和 hidden states**，测 F.linear 与 subset+scatter 的 CUDA graph replay，交替顺序、多轮输出原始样本。没有读完整模型权重，也没有覆盖真实 draft hidden states、MTP 接受率、PLE、目标模型、调度器或端到端请求。`saved_ms_per_3_draft_heads` 只是 3 次相同 head 调用的算术差，**不是每个输出 token 的节省时间**；rows=4 也不等于该请求所有 head 都运行 4 行。若微基准无稳定收益，不必继续接入生产模型。

```bash
# 安装只改两处，不重建 CUDA，不改原启动器，不自动启动服务。
"$PY" scripts/apply-strata-draft-vocab.py --vllm-dir "$VLLM_DIR" --apply
```

### 启动配置：合并，而不是替换

使用当前已经验证 >150 tok/s 的启动命令，保留 MTP=3、PLE、缓存、上下文、GPU memory utilization、attention backend 等所有参数。在那一个现有 `--additional-config` JSON 内，**合并**生成器打印的 `additional_config_to_MERGE` 两个键：

```json
{
  "qwen4_mtp_draft_vocab": "/tmp/qwen-strata-review/cjk-code.json",
  "qwen4_mtp_draft_vocab_sha256": "使用生成器打印的实际SHA256"
}
```

上面只是新增键，**不是完整 additional-config**。不能覆盖原来的 `ple_ssd_offload`、`ple_ssd_native_library` 等；不要靠传第二个 `--additional-config` 期待自动合并。启动时从日志核对真实行数、额外 MiB 和 SHA256。

现有 speculative-config 保持 `method=mtp`、`num_speculative_tokens=3`，本原型要求 `draft_sample_method=greedy`。仓库 `scripts/serve.sh` 默认 MTP=1 且没有这些实验键的入口，因此不要把“直接执行原启动脚本”当作已启用实验；使用本地 agent 维护的测试启动配置并显式设置 MTP=3。本分支故意不改原启动脚本。

### 测试顺序和判定

先做相同配置的 A0/A0 重复，确认平台噪声；再做 A1 完整词表控制，最后做 B 缩词表：

| 组 | 设置 | 目的 |
| --- | --- | --- |
| A0 | 实验键不存在，MTP=3 | 当前生产路径，重复运行估计噪声 |
| A1 | full.json，MTP=3 | 检查 hook/副本/scatter、共享权重和 token 映射，不期待加速 |
| B | cjk-code.json，MTP=3 | 测减少 head 工作后的**净收益** |

每组使用相同 prompt、seed、目标采样参数、输出 token 上限、上下文长度、CUDA graph 配置和 GPU 时钟策略。先 warmup，再至少 5 轮；按 A/B/B/A 或交替测试避免温度漂移。短输出极易受 TTFT 和 MTP 接受率波动影响，建议每条输出至少 512 token，速度组一致处理 EOS；质量组不要为了凑长度强行 ignore EOS。

至少覆盖中文长回答、英文、中文混合代码、JSON/tool call，以及实际经常使用的长上下文。长上下文范围以当前生产配置和显存能稳定支持的长度为限，不自动扩容。既测单并发主目标，也做一次现有多请求/取消请求回归。

不要将 streaming chunk 数当 token 数；MTP 一次可能吐多个 token。用服务端真实 token 数；分开报告 TTFT、解码窗口时间与端到端时间，明确第一个 chunk 含多个 token 时的计数口径。正式 decode tok/s 最好使用同一套服务端计时/基准，不能把 `completion_tokens / 包含 prefill 的总耗时` 直接当成与用户 150 tok/s 相同的指标。

收集实际可得的 draft proposed/accepted 数、各 draft 位置接受率、每轮接受长度、目标 verify 时间、draft 时间、未重叠的 PLE host 等待，以及峰值 VRAM/RAM。counter 名称与 profile 标注以本地固定版本为准，不假设未核对的 Prometheus 指标存在。完整 profile 单独采集，不用 profile 模式的速度替代正常服务速度。

如果 A1 与 A0 的 greedy 输出不同，或 B 出现 token ID 错乱、非法 JSON/tool call、新增拒答/截断、崩溃、长上下文回归，先停下定位，**不要以“理论上 speculative 无损”跳过验证**。目标路径未改不代表端到端的浮点舍入、batch 形状与状态已实测等价。

建议预先约定的性能门槛：中位解码增益至少高于 A0/A0 的噪声，并可暂用 3% 作为继续投入的最低门槛；这不是预测。中文、代码和真实任务均不能明显倒退，不增加不可接受的长上下文/显存压力。若只赢一个 prompt、只改善微基准、接受率下降抵消收益，记录负结果并关闭即可。

整轮成本判断应是：

`净速度 ≈ 每轮最终接受/输出的 token 数 ÷ (目标验证 + draft + 未被隐藏的 I/O/调度时间)`

不是只看 head kernel 倍速，也不是只看 MTP acceptance。先完成固定 MTP=3 的 A/B，之后才单独扫 1/2/3/4，避免把两个变量的收益混在一起。

### 撤回

最简单的关闭方式：移除两个实验键，重新启动测试进程。彻底撤回安装：

```bash
"$PY" scripts/apply-strata-draft-vocab.py --vllm-dir "$VLLM_DIR" --reverse
```

安装器发现实验文件被手工改动会拒绝删除，避免覆盖 agent 的后续工作。不要使用 `git reset --hard` 撤回，以免丢掉原来运行良好的 PLE 补丁。

## 本轮实际验证与未验证项

本轮运行环境：PyTorch 2.10.0+cpu、无 CUDA、未安装完整 vLLM 或模型 tokenizer。`python -m pytest -q tests/test_strata_draft_vocab.py`：**35 passed, 1 skipped**。跳过的是 CUDA graph replay 测试。

已经验证：选中 logits 与 CPU 参考一致、全局 ID 映射、目标 head 不被修改、完整词表控制、soft-cap 后排除值仍为 -inf、非法 ID/配置/hash 拒绝、默认关闭、生成文件不覆盖、安装/重复安装/撤回、拒绝错误 HEAD 和未知文件修改。词表选择测试使用 mock tokenizer；安装测试使用与上游 Git blob 相同的文件 fixture，并在临时 git repo 上做真实 git apply/反向操作。

**未验证：真实 tokenizer 生成结果、真实 vLLM 加载与运行集成、实际完整 CUDA graphs、CMP170HX kernel 性能、MTP 接受率、端到端 token/质量、150 tok/s 以上的增益。** 必须由本地 agent 完成；当前状态保持 draft。

建议本地结果报告包含：代码 commit、vLLM commit、实际启动命令（删掉密钥）、GPU/时钟/温度、模型/tokenizer/map hashes、各组激活日志、prompt token 数、输出 token 数、TTFT、明确口径的 decode tok/s、接受率、峰值内存、逐轮数据及正确性差异。不要上传私有 prompt、API key 或整份校准语料。
