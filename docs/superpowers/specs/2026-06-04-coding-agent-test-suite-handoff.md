# Coding Agent 测试套件重构落地报告

**Date**: 2026-06-04
**Branch**: `trajectory-manager-migration-v2`
**Commit**: `529b5c41 refactor(coding_agent): collapse examples/* into adapter + tests/test_coding_agent/`（本地，未 push）

---

## 一句话

把 `examples/coding_agent_rl/` 下的 4 个 debug / launch 脚本拆掉，运行时部分（billing scrub / system fold / per-sid turn cap）吸收进 `AnthropicAdapter`，debug + e2e + 分析能力全部搬到 `tests/test_coding_agent/`，使"相较 main 净增运行时代码 = `slime/agent/trajectory_manager.py` 一个文件"。

---

## 1. 为什么做这件事

`trajectory-manager-migration-v2` 分支在落地 `TrajectoryManager`（Plan C：DFS 路由 + LCP 对齐 + TITO snapshot 救援）的同时，也在 `examples/coding_agent_rl/` 下零散新增了 4 个文件，让仓库走向两种割裂的协议适配路径：

| 文件 | 原行数 | 实际职责 |
|---|---|---|
| `examples/coding_agent_rl/middleware.py` | 544 | 包 AnthropicAdapter 的独立 aiohttp 进程，加 /get_trajectory + /register_sid + per-sid turn cap + billing/system 协议适配 |
| `examples/coding_agent_rl/middleware_debug.py` | 317 | 给 middleware 提供 6 个 debug hook（按 sid 写 turn dump） |
| `examples/coding_agent_rl/trajectory_manager_debug.py` | 197 | tree dumper（txt / json）|
| `examples/coding_agent_rl/launch_swe.py` | 808 | 批跑 N 个 E2B sandbox，按 sid 注册 dump 目录到 middleware |

问题：
- **协议适配不一致**：4 节点 RL 训练 (`generate.py` → in-process `AnthropicAdapter`) 直接用 adapter 的 aiohttp app，根本不经过 `middleware.py`。所以 billing header scrub / mid-list system fold / per-sid turn cap **只在 launch_swe 路径上生效**，训练路径没有。
- **debug 代码混进运行时目录**：`*_debug.py` / `launch_swe.py` 是开发者批跑+调试用的，跟训练时实际 import 的 `generate.py` 同目录，让后续 reviewer 看不清"哪些是运行时、哪些是调试"。
- **PR 净增过多**：相对 `origin/main`，本分支新增 4 个 `examples/` 文件 + 1 个 `slime/agent/trajectory_manager.py`，未来要合主时面更宽。

期望终态：
- **相较 `origin/main`，运行时净增 = `slime/agent/trajectory_manager.py` 一个文件**
- 测试净增 = `tests/test_coding_agent/` 整个目录
- 4 节点 RL 训练 和 e2e 调试**走同一份 adapter**，协议适配自动一致

---

## 2. 重构后的文件分布

### 运行时（被训练 / 推理 import）

```
slime/agent/
├── trajectory_manager.py        # Plan C TrajectoryManager（main 上不存在，本分支新增）
├── trajectory.py
├── parsing.py
├── sandbox.py                   # E2B 封装
└── adapters/
    ├── anthropic.py             # ★ 本次修改：吸收 scrub / fold / turn cap
    ├── openai.py
    └── common.py

examples/coding_agent_rl/        # ★ 本次砍掉 4 个文件后，只剩
├── generate.py                  # slime 训练侧入口（main 上已有）
├── sandbox.py
├── aiohttp_threaded.py
├── translators/
├── run_qwen36_35b_a3b_swe_8nodes.sh
└── README.md
```

### 测试 + 调试（开发者用、CI 用）

```
tests/test_coding_agent/         # ★ 本次新增整个目录
├── README.md                    # 中文使用指南
├── __init__.py
├── test_coding_agent_swe_e2e.py # CLI 入口 + pytest smoke（取代旧 launch_swe.py）
├── _runner.py                   # adapter app + E2B sandbox + amain
├── _dump_helpers.py             # 6 个 debug hook + aiohttp dump middleware + tree dumpers
├── _analysis.py                 # compute_tree_stats / write_summary / compare CLI
├── test_adapter_wire_scrubbing.py    # 9 个 adapter 协议适配 unit
├── test_dump_helpers.py              # 15 个 dump_helpers unit
├── test_runner.py                    # 10 个 runner unit（含 in-process server）
├── test_analysis.py                  # 6 个 analysis unit
├── test_trajectory_manager.py        # 23 个 TrajectoryManager 核心算法 unit（从 tests/ 搬来）
└── test_trajectory_manager_debug.py  # 4 个 tree dump unit（从 tests/ 搬来）
```

**命名约定**：下划线前缀的模块（`_runner.py / _dump_helpers.py / _analysis.py`）不被 pytest 自动收集，是辅助库。pytest 只收集 `test_*.py`。

---

## 3. 关键架构决策

### 决策 1：协议适配下沉到 adapter

`middleware.py` 里的 3 个 wire 适配特性全部搬进 `slime/agent/adapters/anthropic.py`，作为 `_handle_request` 的入口处理：

| 特性 | adapter 内位置 | 行为 |
|---|---|---|
| `_scrub_claude_code_billing_header_in_body` | `adapters/anthropic.py` | 入口对 body 就地清洗 `x-anthropic-billing-header:` sidechannel |
| `_fold_mid_list_system_into_user` | 同上 | mid-list `role: system` 折叠为前一个 user 消息的 `<system-reminder>` 文本块 |
| `max_turns_per_sid` (HTTP 429) | `AnthropicAdapter.__init__` 新增 kwarg + `_handle_request` 入口判 | 每 sid 满 N turn 后返回 429；`None`（默认）禁用 |

副作用：**4 节点训练路径 (`generate.py`) 自动获得这三个特性**，跟 launch_swe debug 路径行为一致。

### 决策 2：HTTP 服务复用 `adapter.app`，砍掉 middleware 进程

`AnthropicAdapter.app` 本来就是 aiohttp 应用（4 节点训练 `run_app_in_thread(adapter.app)` 一直在用）。e2e test 直接 `aiohttp.web.AppRunner(adapter.app)` 起 in-process 服务即可，没必要再起 subprocess。所以：

- ❌ 删除 `middleware.py` 整个 aiohttp 框架（parse_args / build_app / main / `/get_trajectory` / `/register_sid` / SSE 捕获）
- ✅ debug 钩子（SSE 捕获、turn 编号、per-turn dump）作为**单独的 aiohttp middleware** 套在 `adapter.app` 上，由 `_dump_helpers.install_dump_layer(adapter, ...)` 注入
- ✅ sid → inst_dir 映射改成 in-process dict (`debug_handle.sid_dump_dir[sid] = str(inst_dir)`)，取代 `/register_sid` HTTP 调用
- ✅ drain 改成直接调 `adapter.finish_session(sid, ...)`，取代 `/get_trajectory` HTTP 调用

### 决策 3：SSE 捕获用 `WeakKeyDictionary` keyed by `asyncio.Task`

`aiohttp` 在单 event loop 服务多并发请求时，对 `StreamResponse.write` 做实例 / 类级 monkey-patch 会跨请求互染（见 memory `aiohttp_per_task_monkeypatch_race`）。实现按以下约束：

- module-level `_SSE_CAPTURE: WeakKeyDictionary[asyncio.Task, list[bytes]]`
- `_install_sse_capture_patch()` 全 process 只 patch 一次（幂等）
- dump middleware 在 handler 调用**前**注册 captor `_SSE_CAPTURE[task] = []`，`finally` 块 `pop`
- patched `write` 只对**当前 task** 的 captor 写

### 决策 4：unit test 也搬进 `tests/test_coding_agent/`

`tests/test_trajectory_manager.py`（792L）和 `tests/test_trajectory_manager_debug.py`（183L）原本在 `tests/` 根目录。按约束"所有 coding agent 相关测试集中"，搬到 `tests/test_coding_agent/` 下：

- `test_trajectory_manager_debug.py` 的 import 从 `examples.coding_agent_rl.trajectory_manager_debug` 改成 `tests.test_coding_agent._dump_helpers`
- `test_trajectory_manager.py` 的 `test_debug_dump_shape` 同改

---

## 4. 测试覆盖

`pytest tests/test_coding_agent/ tests/test_agent_adapters.py` → **85 passed / 4 skipped / 0 fail**

按文件分布：

| 文件 | 用例数 | 覆盖范围 |
|---|---|---|
| `test_adapter_wire_scrubbing.py` | 9 | billing scrub（4）+ system fold（3）+ turn cap（2） |
| `test_dump_helpers.py` | 15 | 6 个 hook 各自 + dump middleware e2e 捕获 + sid override |
| `test_runner.py` | 10 | normalize_sample / load_dataset / ensure_e2b_env / start_adapter_app / drain_and_dump_sid |
| `test_analysis.py` | 6 | compute_tree_stats / write_summary / compare（pass + fail） |
| `test_trajectory_manager.py` | 23 | TrajectoryManager DFS 路由 + LCP 对齐 + TITO snapshot（搬迁） |
| `test_trajectory_manager_debug.py` | 4 | tree dump txt / json + reasoning/thinking（搬迁） |
| `test_coding_agent_swe_e2e.py` | 1 (skipped) | 真实 e2e smoke，opt-in via `SWE_E2E_SMOKE=1` |
| `tests/test_agent_adapters.py` | 17 + 3 skip | 已有的 adapter 单元测试（回归） |

整个套件 **3.5 秒**跑完，没有任何 GPU / sglang / 模型依赖。

---

## 5. 未完成的验证（接手开发要做）

### Step A：真实 SWE e2e 大批跑

```bash
cd /mnt/jingshenghang/code/slime_swe/slime
# 假设 sglang 上游已经在 127.0.0.1:30000 起好
python -m tests.test_coding_agent.test_coding_agent_swe_e2e \
    --limit 100 --concurrency 16 \
    --runs-dir /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe_new
```

环境要求：
- `E2B_API_KEY` 设了（值不重要，GLM 网关不在意，但 SDK 要走格式 check —— `ensure_e2b_env` 会替换 dummy）
- sglang upstream 可达（默认 `--sglang-url http://127.0.0.1:30000`）
- 模型 checkpoint 在 `--model` 指向路径（默认 `Qwen3.6-35B-A3B`）
- 节点 / cc tarball 在默认路径

预期：每个 inst_dir 下生成跟历史 `runs/swe/` 完全一致的文件结构（`trajectory_tree.{json,txt}` / `trajectory.json` / `turn_NNNN_*` / `summary.json`）。

### Step B：与历史数据对比

```bash
python -m tests.test_coding_agent._analysis compare \
    --baseline /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe/20260604-084045 \
    --new      /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe_new/<本次 batch>
```

判定逻辑：5 个 axis（`n_forks_total` / `n_dropped_turns_total` / `n_dropped_tokens_total` / `n_with_fork` / `n_with_drop`），每 axis 绝对差 ≤ `max(1, ⌈baseline*10%⌉)` 视为 pass。任一 fail 退出码 1。

**如果对比 fail**：先看哪个 axis 偏离最多。
- `n_forks_total` 大偏 → adapter 入口 scrub / fold 在新路径上行为变了，让某些 prompt prefix 不再 fork
- `n_dropped_*` 大偏 → TITO snapshot 阈值 (`tito_snapshot_min_loss_tokens`) 或 trajectory_manager 的 LCP 对齐改了
- `n_with_fork` 跟 `n_forks_total` 不成比例 → 一些 instance 出现 fork 爆炸（顶级 sub-agent 行为漂移）

### Step C：4 节点 RL 训练验证

```bash
bash /mnt/jingshenghang/code/slime_swe/0527-async/progress/4-nodes-training/coding-agent-rl-4-nodes.sh
```

Dry-run（只验证 import 链，不真启训练）已经过：
```bash
python -c "
import os; os.environ['SLIME_TITO_SNAPSHOT_MIN_LOSS_TOKENS'] = '1000'
from examples.coding_agent_rl.generate import generate
from slime.agent.adapters.anthropic import AnthropicAdapter
AnthropicAdapter(tokenizer=None, sglang_url='http://fake', max_turns_per_sid=100, tito_snapshot_min_loss_tokens=1000)
print('OK')
"
```

完整 RL 跑第一个 iteration 时观察：
- `SWE_LIST_TRAJECTORY=1` + `SWE_SAVE_TRAJECTORY_TREE=1` 输出正常
- rollout 内 fan-out 出 Sample 序列、`loss_mask` 正确
- grad_norm 不为 0（之前小模型 cold-start 真零的坑，见 memory `small_model_not_oom_workaround`）

---

## 6. 接手开发的常见操作

### "我要加一个新的 wire 适配特性"

例如：claude code 又出了新版本，body 里多了某个需要剥掉的字段。

1. 在 `tests/test_coding_agent/test_adapter_wire_scrubbing.py` 加 RED test（参考 `test_scrub_billing_header_from_system_string`），描述 `body_obj` 输入 / 期望 mutation。
2. 在 `slime/agent/adapters/anthropic.py` 加私有函数 `_scrub_<feature>_in_body(body_obj) -> bool`（参考 `_scrub_claude_code_billing_header_in_body`）。
3. 在 `_handle_request` 入口 line ~530 处插一行 `_scrub_<feature>_in_body(body)`。
4. RED → GREEN，跑 `pytest tests/test_coding_agent/test_adapter_wire_scrubbing.py`。

**关键**：不要再去 examples/ 下加文件。运行时适配一律走 adapter。

### "我要加一个新的 debug hook"

例如：想 dump 每个 turn 的 SGLang upstream 实际看到的 routing key。

1. 在 `tests/test_coding_agent/test_dump_helpers.py` 加 RED test，调用 `Debug.on_<new_hook>(...)`，assert 写出特定文件。
2. 在 `tests/test_coding_agent/_dump_helpers.py` 的 `Debug` 类加 method。
3. 如果需要中间件触发（不是 adapter 的 `on_turn_appended` 桥），改 `build_dump_middleware` 在 handler 前后调它。

**关键**：debug hook 不能放进 `slime/agent/`。这是测试目录专属。

### "compare 报 fail 怎么 debug"

`_analysis.compare()` 返回的 `report["text"]` 会列出每 axis 的 baseline / new / delta / 是否过阈。挑偏离最大的 axis → 去 `summary.json` 找 outlier instance（一两个 instance 撑大 total 是常见模式）→ 看那个 instance 的 `trajectory_tree.json` / `turn_NNNN_*.json` diff 找根因。

### "我要让 e2e test 也跑 CI"

当前 `test_coding_agent_swe_e2e::test_smoke_e2e` 是 opt-in（`SWE_E2E_SMOKE=1` 才跑）。CI 接入：

1. 加一个 CI job，环境里设 `E2B_API_KEY` + 准备好小 sglang 实例
2. `SWE_E2E_SMOKE=1 pytest tests/test_coding_agent/test_coding_agent_swe_e2e.py -k smoke`
3. 默认 2 instance + 2 concurrency，跑通即过

或者改 `test_smoke_e2e` 默认 `pytest.skip` 行为：把 `SWE_E2E_SMOKE` 反向（默认跑、`SWE_E2E_SMOKE=0` 才 skip）。改一行就行。

### "我要做大批跑的并行优化"

`_runner.amain` 用 `asyncio.Semaphore(args.concurrency)` 限并发，没有 worker pool / 真线程。如果想换 process pool（让 sandbox 启动并发到 N=100+）：

- 看 `amain` 的 `worker` 协程内除了 `run_one_instance` 还做了 incremental `write_summary`，结构清晰，但跨进程要 pickle `AdapterSession`（不可能）。所以并发只能在 asyncio 层，不能跨进程。
- 真正的瓶颈是 E2B sandbox 启动 + claude-code 单 instance 时延（约 30s + N 个 turn × 5s）。提升 N=100 的并发率主要靠 GLM E2B 网关的并发上限（见 memory `e2b_glm_concurrency_limits`）。

---

## 7. 重要约束（接手开发不要踩）

### 不要在 `slime/agent/` 下新增文件

除非新功能是真正的运行时基础设施（被 4 节点训练 import）。在 spec / plan 阶段就先问"4 节点训练用到吗？"，没有就放 `tests/test_coding_agent/`。

### 不要在 `examples/coding_agent_rl/` 下新增文件

那目录现在只剩**真实运行时**（`generate.py` 是训练入口、`sandbox.py` 是 E2B 封装）。任何"debug 一下"或"批跑试试"的脚本都应该是 `tests/test_coding_agent/test_*.py`。

### 不要破坏 adapter 的现有 ctor 签名

`AnthropicAdapter.__init__` 现有 kwarg：`tokenizer / sglang_url / tool_parser / reasoning_parser / tito_snapshot_min_loss_tokens / max_turns_per_sid / on_turn_appended`。`generate.py` 和 `_runner.start_adapter_app` 都按这个签名构造。新加 kwarg 要默认值，旧调用不变。

### 不要直接 commit 跳过 pre-commit

repo 有 `.pre-commit-config.yaml` 配置了 ruff / autoflake / isort / black。Memory `precommit_before_commit.md` 要求 commit 前先跑 `pre-commit run --files <...>`，否则可能误 `--amend` 丢工作。流程：

```bash
git add <files>
pre-commit run --files <files>  # 看是否有自动 fix
git add -u <files>              # re-stage 修改
git commit -m "..."             # pre-commit hook 也会再跑一次
```

### 不要 force push / 推到 main

按 user 全局规则，`main / master` 永远不能 push。当前分支 `trajectory-manager-migration-v2` 推到 `jsh` remote（个人 fork `github.com/jingshenghang/slime`），不要推 `origin`（THUDM/slime）。

---

## 8. 文件 quick reference

| 想做什么 | 改哪里 |
|---|---|
| 改 trajectory 节点 / 路由 / LCP 算法 | `slime/agent/trajectory_manager.py` |
| 改协议适配（scrub / fold / turn cap） | `slime/agent/adapters/anthropic.py` |
| 改 SGLang 调用 / tool parser 入口 | `slime/agent/adapters/anthropic.py:_handle_request` |
| 改训练侧 generate 入口 | `examples/coding_agent_rl/generate.py` |
| 加 debug hook / 改 dump 文件格式 | `tests/test_coding_agent/_dump_helpers.py` |
| 改 tree 树状统计 / summary 字段 | `tests/test_coding_agent/_analysis.py` |
| 改 e2e 入口 CLI / 批跑流程 | `tests/test_coding_agent/_runner.py` + `test_coding_agent_swe_e2e.py` |
| 改对比阈值 / compare axis | `tests/test_coding_agent/_analysis.py:COMPARE_TOLERANCE` / `COMPARE_AXES` |

---

## 9. 历史 commit 引用

本次重构落在单一 commit：

```
529b5c41 refactor(coding_agent): collapse examples/* into adapter + tests/test_coding_agent/
```

前序 10 个 commit（已经在 `trajectory-manager-migration-v2` 分支但还未合 main）：

```
06d8a6a4 docs: archive design specs and implementation plans for this iteration
bc4eedc2 chore(coding_agent_rl): import trajectory_manager_debug helper + test
9a6e084c feat(launch_swe): route per-sid debug dumps into inst_dir, drop mw/ layer
c7411e39 feat(anthropic-adapter): skip append_turn for cc title-generation requests
944d3767 feat(anthropic-adapter): add _is_cc_title_generation_request helper
710815bd feat(generate): wire SLIME_TITO_SNAPSHOT_MIN_LOSS_TOKENS env (1000)
bdbaa4f9 feat(middleware): expose --tito-snapshot-min-loss-tokens
3afda620 feat(anthropic-adapter): forward tito_snapshot_min_loss_tokens
07c685d0 feat(trajectory_manager): emit complementary snapshot Sample
abe03aa3 feat(trajectory_manager): add tito_snapshot_min_loss_tokens ctor arg
```

合主 PR 时建议把 11 个 commit squash 成 1-3 个有意义的 commit（`trajectory_manager` 落地 / adapter 集成 / 测试套件），不要每个 squash 提交都搬上 main。
