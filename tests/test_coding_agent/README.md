# test_coding_agent — Slime Coding Agent 端到端测试套件

这个目录是 slime 仓库中 coding agent rollout 链路（trajectory_manager + AnthropicAdapter + E2B sandbox + claude-code CLI）的**集中测试位置**。它取代了过去散落在 `examples/coding_agent_rl/` 下的 `launch_swe.py / middleware.py / middleware_debug.py / trajectory_manager_debug.py` 四个调试/批跑脚本。

## 目录结构

```
tests/test_coding_agent/
├── README.md                              # 本文档
├── __init__.py                            # 空，让下划线模块可 import
├── test_coding_agent_swe_e2e.py           # 主 e2e 入口（CLI + pytest smoke）
├── _runner.py                             # adapter app 启动 + sandbox 调度 + amain
├── _dump_helpers.py                       # tree dumper + 6 个 debug hook + aiohttp dump middleware
├── _analysis.py                           # compute_tree_stats / write_summary / compare CLI
├── test_adapter_wire_scrubbing.py         # adapter wire-scrub + turn cap 单元测试
├── test_dump_helpers.py                   # dump_helpers 单元测试
├── test_runner.py                         # _runner 单元测试（含 in-process server）
├── test_analysis.py                       # _analysis 单元测试
├── test_trajectory_manager.py             # TrajectoryManager 核心算法单元测试（从 tests/ 搬来）
└── test_trajectory_manager_debug.py       # tree dump 单元测试（从 tests/ 搬来）
```

**命名约定**：下划线开头的模块（`_runner.py / _dump_helpers.py / _analysis.py`）不被 pytest 自动收集，是辅助库。pytest 收集 `test_*.py` 共 6 个文件。

## 跑法

### 1. 纯 unit test（无外部依赖，永远能跑）

```bash
cd /mnt/jingshenghang/code/slime_swe/slime
python -m pytest tests/test_coding_agent/ --no-header -q
```

预期：~67 passed + 1 skipped（其中 skipped 是默认 disable 的 SWE e2e smoke）。

如果还想跑 adapter 老 test 做回归：

```bash
python -m pytest tests/test_coding_agent/ tests/test_agent_adapters.py --no-header -q
```

预期 82 passed + 3 skipped。

### 2. SWE e2e smoke（需要 E2B + sglang + 模型 checkpoint）

```bash
SWE_E2E_SMOKE=1 pytest tests/test_coding_agent/test_coding_agent_swe_e2e.py -k smoke
```

会跑 2 个 SWE instance，验证端到端链路通畅。失败的常见原因：
- E2B_API_KEY 没设
- sglang upstream 不通（默认 `http://127.0.0.1:30000`）
- 模型 checkpoint 路径不存在
- host_ip 不对（4 节点上要用所在节点的 reverse-VIP）

### 3. 大批量跑出 dump 数据

跟旧 `launch_swe.py` 一样的工作流：

```bash
python -m tests.test_coding_agent.test_coding_agent_swe_e2e \
    --limit 100 --concurrency 16 \
    --runs-dir /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe_new
```

跑完后 `runs/swe_new/<时间戳>/` 下会有：

```
runs/swe_new/<batch_ts>/
├── batch_meta.json            # 这次跑的元数据（cmd / dataset / model 等）
├── summary.json               # 所有 instance 结果的 JSON 列表
├── summary.txt                # 人读表 + fork/drop 详情段
└── <idx>_<safe_instance_id>/
    ├── meta.json              # 实例元数据
    ├── PROBLEM_STATEMENT.md   # 任务描述
    ├── stdout.log/stderr.log  # claude -p 输出
    ├── summary.json           # 该实例汇总
    ├── trajectory_tree.json   # 完整树结构
    ├── trajectory_tree.txt    # 树的人读视图
    ├── trajectory.json        # 训练用 Sample 列表（含 prompt_text/response_text）
    ├── turn_0001_request.json
    ├── turn_0001_request_scrubbed.json
    ├── turn_0001_response.sse
    ├── turn_0001_sglang_request.json
    ├── turn_0001_sglang_response.json
    └── turn_0001_openai.json
```

### 4. CLI 参数表

| 参数 | 默认 | 说明 |
|---|---|---|
| `--dataset` | swe-train-1545-slime.jsonl | 数据集路径 |
| `--limit` | 20 | 跑前 N 个 instance |
| `--offset` | 0 | 跳过前 N 个 |
| `--concurrency` | 20 | asyncio.Semaphore 并发度 |
| `--max-turns-per-sid` | 100 | 每个 sid 最多多少 /v1/messages turn（超了返回 429） |
| `--model` | Qwen3.6-35B-A3B | tokenizer 路径 |
| `--tool-parser` | qwen3_coder | adapter 参数 |
| `--reasoning-parser` | qwen3 | adapter 参数 |
| `--sglang-url` | `http://127.0.0.1:30000` | sglang upstream |
| `--host-ip` | `$SLIME_HEAD_HOST` 或 172.27.14.123 | sandbox 反连本机的 IP |
| `--port` | 18080 | adapter app 监听端口（4 节点上唯一开放反向端口） |
| `--node-tgz` | node-v22.20.0 tar.xz | 上传到沙箱的 Node tarball |
| `--cc-tgz` | claude-code 2.1.143 | 上传到沙箱的 cc tgz |
| `--runs-dir` | `runs/swe_new` | dump 输出根目录 |
| `--sandbox-timeout` | 1800 | E2B sandbox 生命周期上限 (s) |
| `--claude-timeout` | 1500 | `claude -p` 单次预算 (s) |
| `--cc-prompt` | 默认 SWE prompt | claude 驱动指令 |
| `--tito-snapshot-min-loss-tokens` | None | TITO 救援阈值，None 关闭 |

## 与历史数据对比

跑完新 batch 后，跟参考数据（`runs/swe/<batch_ts>/`）对比 fork / TITO drop 量级：

```bash
python -m tests.test_coding_agent._analysis compare \
    --baseline /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe/<历史 batch> \
    --new      /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe_new/<本次 batch>
```

输出表格化对比 5 个 axis：
- `n_forks_total`
- `n_dropped_turns_total`
- `n_dropped_tokens_total`
- `n_with_fork`
- `n_with_drop`

判定：每个 axis 绝对差异 ≤ max(1, ⌈baseline * 10%⌉) 视为 pass，全 pass exit 0；任一 fail exit 1。

## 与 4 节点 RL 训练的关系

4 节点训练脚本 `/mnt/jingshenghang/code/slime_swe/0527-async/progress/4-nodes-training/coding-agent-rl-4-nodes.sh` 通过 `examples.coding_agent_rl.generate.generate` 进入，**复用同一份** `slime.agent.adapters.AnthropicAdapter` + `slime.agent.trajectory_manager.TrajectoryManager`。两条路径的本质差异只有：

| 路径 | adapter 启动方式 | dump 路径 |
|---|---|---|
| 本 e2e test | in-process aiohttp + dump middleware | `runs/swe_new/<batch>/<inst>/turn_*` 等 |
| 4 节点训练 | `generate.py` 用 `run_app_in_thread(adapter.app)` 起 in-process app | 训练侧不写 turn-level dump |

所以本 test 通过即代表 trajectory 生成基础设施（scrub / fold / turn cap / cc title-gen skip / TITO snapshot）对训练侧可用。

## 重构后约束

相较 `main` 分支，本次重构**只新增了两类文件**：

1. `slime/agent/trajectory_manager.py`（运行时代码，唯一允许的 `slime/` 下新增）
2. `tests/test_coding_agent/` 目录（本目录，所有 debug + test）

`slime/agent/adapters/anthropic.py` 和 `examples/coding_agent_rl/generate.py` 是 main 上已存在的文件，重构过程中被修改但不算"新增"。

历史上 `examples/coding_agent_rl/` 下的 `launch_swe.py / middleware.py / middleware_debug.py / trajectory_manager_debug.py` 在本次重构中被**删除**——它们的核心运行时逻辑（scrub / fold / turn cap）已被吸收进 adapter，debug / dump / 分析 / 大批跑能力全部搬到本目录。
