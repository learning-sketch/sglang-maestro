# sgl-agent

在 H20 类加速卡（NVIDIA H20，或者自定义的 H20 级别树外设备）上，围绕 SGLang 做**模型适配、性能分析、框架特性调优和部署规划**的 agent。

它回答四类问题：

| 场景 | 问题 | 命令 |
| --- | --- | --- |
| 1 模型适配 | 模型在这块卡上怎么跑起来？框架要额外补哪些东西（平台插件 hook、算子、精度路线、import 时的编译）？改动原则是什么？上手顺序？ | `sgl-agent adapt` |
| 2 性能分析 | 算子性能如何、瓶颈在哪（roofline）？有没有算子异常？哪里能做融合、通算重叠、多流并行、权重预取？ | `sgl-agent roofline` / `trace` / `logs` |
| 3 框架特性 | DP attention、prefill CP、DCP、LayerNorm SP、HiCache、DeepEP、TBO、EPLB、投机解码：要不要开、在这块卡上要适配什么、参数怎么调？ | `sgl-agent features` |
| 5 部署 | 给定负载和 SLO，实例开多大、几个副本、要不要 PD 分离、router 怎么配？最终的启动/压测命令是什么？ | `sgl-agent deploy` |

`sgl-agent run` 把四个场景串成一条流水线；给了实测日志（或用 `--demo` 的合成数据）时，还会**用实测校准成本模型、重新规划部署**，形成“预测 → 实测 → 校准 → 再规划”的闭环。

## 安装

需要 Python ≥ 3.10。不需要 GPU，也不需要安装 SGLang：所有分析基于内置的 SGLang 能力快照和成本模型离线完成。

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
sgl-agent --help
```

## 快速开始

```bash
# 内置的模型、硬件预设和负载
sgl-agent list

# 一键跑完全部场景（没有设备时用 --demo 的合成 trace/日志/压测演示闭环）
sgl-agent run --model qwen3-32b --hardware h20 --workload chat --demo -o out/qwen3-32b-chat
# 汇总在 out/qwen3-32b-chat/report.md，各场景的详细报告在 01-adapt.md … 06-deploy-calibrated.md
```

单独跑某个场景：

```bash
# 场景 1：树外 H20 类设备上适配 DeepSeek-V3（hook 清单、算子缺口、精度路线、上手步骤）
sgl-agent adapt --model deepseek-v3 --hardware template-h20-class -o adapt.md

# 场景 2：roofline；分析真实 profiler trace（/start_profile 采集的目录）和 server 日志
sgl-agent roofline --model qwen3-32b --hardware h20 --tp 8
sgl-agent trace /tmp/sgl-profile --model qwen3-32b --hardware h20 --tp 8
sgl-agent logs server.log bench.txt --model qwen3-32b --hardware h20 --tp 8

# 场景 3：针对负载评估各框架特性，给出组合后的启动命令
sgl-agent features --model deepseek-v3 --hardware h20 --workload rag-long-context

# 场景 5：部署规划（--json 可以留给 calibrate --plan 对比预测）
sgl-agent deploy --model qwen3-235b-a22b --hardware h20 --workload chat --json -o plan.json
```

所有分析命令默认输出 Markdown；`--json` 或 `-o 文件.json` 输出 JSON。

## 在真实设备上闭环

1. `sgl-agent deploy ...` 拿到部署方案，按“部署命令”一节启动实例和 router。
2. 用方案里的压测命令（`sglang.benchmark.serving --output-file bench_result.jsonl`）压测。
3. 校准成本模型，并和方案的预测对比：

   ```bash
   sgl-agent calibrate --log server.log --bench bench_result.jsonl --plan plan.json \
     --model qwen3-235b-a22b --hardware h20 --tp 8 -o calibration.json
   ```

4. 带上校准结果重新规划：`sgl-agent deploy ... --calibration calibration.json`（`features`、`trace`、`logs` 也接受 `--calibration`）。

`sgl-agent run --log server.log --bench bench_result.jsonl ...` 会自动完成第 3、4 步。

## 自定义设备

复制 `src/sgl_agent/data/hardware/template-h20-class.yaml`，填上芯片的规格（显存容量/带宽、各精度算力、机内/机间互联）和**实际的软件栈**（平台类型、device_type、dispatch key、支持的 GEMM/KV 精度、注意力和 MoE 后端、通信库、已有的融合算子……），然后：

```bash
sgl-agent adapt --model deepseek-v3 --hardware ./my-chip.yaml
```

软件栈里没列出的能力，会被当作“需要你的 SGLang 平台插件提供”，出现在场景 1 的缺口清单里。`efficiency` 一节是成本模型的效率假设，拿到实测后用 `sgl-agent calibrate` 校准。

## 依据与假设

- **SGLang 行为来自源码**：启动参数、取值、模型架构、平台接口（hook、能力位、编译点、树外平台会抛错的调用）等由 `sgl-agent sync-knowledge` 从 SGLang 源码静态分析生成（内置快照对应 SGLang `72730d6`）；调度、PD、router、压测工具等行为规则记录在 `src/sgl_agent/knowledge/facts.py`，报告里的每条结论都带源码出处。部署其它版本时：

  ```bash
  sgl-agent sync-knowledge --sglang-src /path/to/sglang -o snapshot.json
  export SGL_AGENT_KNOWLEDGE=$PWD/snapshot.json   # 或每条命令加 --knowledge snapshot.json
  ```

  生成的每条启动命令都会按快照校验参数名、取值和弃用状态，不合法的参数会直接标在报告里。
- **耗时来自成本模型**：逐算子 roofline（效率系数见硬件 YAML）+ 按 SGLang 调度规则的离散事件仿真（prefill 优先、chunked prefill、KV 预留与撤回、radix/HiCache、router 策略、PD 的 KV 传输）。这是**估计**，上线前请用实测校准。
- **合成数据只用于演示和回归测试**：`synth-trace`、`synth-log` 和 `run --demo` 生成的文件都带 synthetic 标记，基于它们的报告会明确写出“不是实测”。

## 开发

```bash
pytest            # 测试
ruff check . && ruff format --check .
```

代码结构：

| 目录 | 内容 |
| --- | --- |
| `src/sgl_agent/knowledge/` | SGLang 能力快照（`sync.py` 从源码生成）和行为规则 `facts.py` |
| `src/sgl_agent/hardware/`、`model/`、`workload.py` | 硬件规格、模型结构/参数/并行与显存、负载与 SLO |
| `src/sgl_agent/adapt/` | 场景 1：平台画像、hook 清单、算子清单、精度路线、上手步骤 |
| `src/sgl_agent/perf/` | 场景 2：roofline、trace 解析与检测器（异常/融合/重叠/多流/预取）、日志分析、校准 |
| `src/sgl_agent/features/` | 场景 3：各特性的 advisor 和组合 |
| `src/sgl_agent/deploy/` | 场景 5：实例模型、集群仿真、规模/副本/PD 搜索、放置与命令渲染 |
| `src/sgl_agent/pipeline.py`、`cli.py` | 一键流水线和命令行 |
