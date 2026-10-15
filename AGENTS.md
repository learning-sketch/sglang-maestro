# sglang-maestro 开发约定

以后的 agent 先读这份文件，再改代码。用法和场景说明见 [README.md](README.md)。

## 公共约定

### 不编造 SGLang 的行为

1. 代码依赖的每一条 SGLang 行为（参数、默认值、日志格式、HTTP 接口、调度规则、插件机制）都必须在 SGLang 源码里核对过。基准版本是内置快照 `src/sglang_maestro/knowledge/sglang_capabilities.json` 的 `sglang_commit`。用 `scripts/fetch_sglang.sh` 把该 commit 检出到本地（默认 `/tmp/sglang-src`），不要在脚本或代码里硬编码这个 hash。目录不存在时按 SHA 浅取，服务器不支持则回退到完整 fetch 并打印一行说明；目录已是 git 仓库且 commit 已在本地时直接 checkout，不联网，也不要加 `--depth`。仓库地址可用 `SGLANG_REPO_URL` 覆盖。
2. 核对过的行为写进 `src/sglang_maestro/knowledge/facts.py`：数值写成带 docstring 的模块级常量；出处写进 `SOURCES`（key → `相对 SGLang 仓库根的路径::函数或类（必要说明）`），代码里用 `source(key)` 引用。报告里会显示成“源码依据”。
3. 核对不了的（硬件数字、效率系数、外部库的行为）在报告里明确标成“假设”，不要写成事实。
4. 不 import SGLang，不需要 GPU。分析全部离线完成。SGLang 源码只用来核对和静态解析（`knowledge/sync.py`）。
5. 生成的命令要通过现有的快照校验（`commands.py` 的 `launch_server_command` / `tool_command` 会检查参数名和取值，问题记在 `Command.problems` 里）。

### 代码

- 代码、注释、docstring 用英文；报告、提示、错误信息等面向用户的文字用中文。
- 注释只写代码本身表达不了的约束，不写叙述性注释。
- 和周围代码保持一致：dataclass + 纯函数；报告用 `markdown.py` 里的 `table` / `bullets` 等；结构化的发现用 `findings.py` 的 `Finding`；对外的结构都提供 `to_dict()`。
- ruff：line-length 100，规则 E、F、W、I、B、UP、SIM。`ruff check .` 和 `ruff format --check .` 必须通过。

### 测试

- 默认测试要快。慢测试加 `@pytest.mark.slow`；`pytest` 默认带 `-m "not slow"`。
- 决策基线在 `tests/baselines/`（`adapt.json`、`roofline.json`、`features.json`、`deploy.json`）。改动让基线变了，就用 `sglang-maestro sweep --write-baselines tests/baselines` 重新生成，并在 PR 描述里逐条解释每处变化为什么是对的；解释不了的变化就是回归。
- 部署仿真用 `tests/conftest.py` 里的 `SMALL_DEPLOY` 和 `dense_chat`、`oot_moe` fixture 缩短时间。
- 快照可复现：`SGLANG_SRC` 指向 SGLang 源码时，`pytest -m slow` 运行 `sglang-maestro sync-knowledge`，结果必须和内置 `sglang_capabilities.json` 逐字节一致。没设 `SGLANG_SRC` 时该测试 skip。不一致时修 `sync.py` 的输出确定性（例如文件遍历顺序），不要放宽比较。
- 合成数据（`synth-trace`、`synth-log`、`run --demo`）只用于演示和测试，基于它的报告必须写明“不是实测”。

### 自检

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e '.[dev]'
pytest && ruff check . && ruff format --check . && shellcheck scripts/*.sh
scripts/fetch_sglang.sh
SGLANG_SRC=/tmp/sglang-src pytest -m slow
sglang-maestro run --model qwen3-32b --hardware h20 --workload chat --demo -o /tmp/demo
```

### 交付

- 从最新的 `main` 开新分支；每个逻辑改动一个提交，提交信息写清楚做了什么。
- 第一次推送前 rebase 到最新的 `main` 并解决冲突（常见冲突点是 `cli.py` 的 `build_parser` 和 `README.md`，保留双方的改动）。已经推送过的分支用 `git merge origin/main` 合入，不要强推：B、C、D、E 会和本分支并行，先合并的 PR 会让其他分支落后，强推会改写已经推送的历史。
- 能开 PR 就开 PR。描述里写：改了什么、怎么验证的（命令和结果）、哪些是假设、还有什么没做。
- 同步更新 README.md 里对应的用法。
- 不改和任务无关的代码；发现的其它问题写在 PR 描述里，不要顺手改。

## 目录地图

| 路径 | 内容 |
| --- | --- |
| `src/sglang_maestro/knowledge/` | SGLang 能力快照（`sync.py` 从源码生成）和行为规则 `facts.py` |
| `src/sglang_maestro/hardware/`、`model/`、`workload.py` | 硬件规格、模型结构/参数/并行与显存、负载与 SLO |
| `src/sglang_maestro/adapt/` | 场景 1：平台画像、hook 清单、算子清单、精度路线、上手步骤 |
| `src/sglang_maestro/perf/` | 场景 2：roofline、trace 解析与检测器（异常/融合/重叠/多流/预取）、日志分析、校准 |
| `src/sglang_maestro/features/` | 场景 3：各特性的 advisor 和组合 |
| `src/sglang_maestro/deploy/` | 场景 5：实例模型、集群仿真、规模/副本/PD 搜索、放置与命令渲染 |
| `src/sglang_maestro/pipeline.py`、`cli.py`、`sweep.py` | 一键流水线、命令行，以及模型 × 硬件 × 负载的回归 sweep |
| `tests/baselines/` | 场景 1、2、3、5 的决策基线（`sglang-maestro sweep --write-baselines` 生成） |
| `src/sglang_maestro/commands.py` | 启动、router、压测命令的渲染和快照校验 |
| `src/sglang_maestro/findings.py`、`markdown.py` | 结构化发现和 Markdown 报告片段 |
| `src/sglang_maestro/target.py`、`data/` | 模型/硬件/负载的加载入口，以及内置数据 |
| `tests/` | 场景、命令行、流水线和基线测试；`conftest.py` 里有缩短部署仿真的 fixture |
| `scripts/fetch_sglang.sh` | 按快照里的 commit 浅取 SGLang 源码；已有仓库且 commit 在本地则不联网 |
