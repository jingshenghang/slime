# Design: src_v2 TrajectoryManager 迁移到 slime agent

**Date**: 2026-06-03
**Branch**: `trajectory-manager-migration-v2` (拉自 origin/main `d719f036`)
**Source**: `/mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/src_v2`
**Target**: `/mnt/jingshenghang/code/slime_swe/slime`

## 1. 背景与目标

src_v2 是一版精简的 trajectory 抓取实现，包含：

- `trajectory_manager.py` (Plan C, 2026-06-03)：**per-sid 树状轨迹管理**，DFS 按
  `(role, node_match_key)` 路由，per-turn assistant leaf 存 sglang 端的
  `prompt_ids / response_ids / response_logprobs / finish_reason / turn_index`；
  `get_trajectory` 走 LCP drop-and-replace 处理 TITO drift，直接产出 `list[Sample]`。
- `middleware.py`：独立进程，挂 :18080，嵌套 slime 的 `AnthropicAdapter` 处理
  `/v1/messages`，加 `/get_trajectory` route 让外部 drain 一个 sid。包含
  Claude Code wire 兼容性（billing-header scrub、mid-list system fold、SSE
  per-task capture）。
- `middleware_debug.py` + `trajectory_manager_debug.py`：可选 debug 工具，
  写 per-turn 的 `turn_NNNN_*` artifact 和 `trajectory_tree.{txt,json}`、
  `trajectory.json`（带解码后的 prompt_text / response_text）。
- `launch_swe.py`：起 1 个共享 middleware + N 个 E2B sandbox worker，
  并发跑 SWE 数据集；每个 worker 用独立 Bearer-token sid 把轨迹挂到同一个
  TrajectoryManager。

**目标**：把这套实现移植到 slime（基于 origin/main 拉的新分支
`trajectory-manager-migration-v2`），并改造 slime 现有 Anthropic adapter 接入
`TrajectoryManager`。最终交付标准是：**slime 里的 `launch_swe.py` 能 20
并发拉起 SWE 实验**。

`trajectory_manager.py` 核心不动；adapter 来适配它。debug 文件和 launch
脚本是调试用，最终交付不强依赖（但本次迁移要带上）。

## 2. 现状对照

### slime origin/main 现有 agent 结构

```
slime/agent/
├── __init__.py
├── parsing.py            # ParsedModelOutput / parse_model_output
├── sandbox.py            # E2BSandbox + Sandbox protocol
├── trajectory.py         # TurnRecord / TurnSegment / TokenSegment / merge_turns / fan_out_sample_segments
└── adapters/
    ├── __init__.py
    ├── common.py         # BaseAdapter / AdapterChain / call_sglang_generate / render_token_ids
    ├── anthropic.py      # AnthropicAdapter (main+active_sub chain, _select_chain, finish_session→list[TokenSegment])
    └── openai.py         # OpenAIAdapter (同样基于 TurnRecord/TurnSegment)

examples/coding_agent_rl/
├── generate.py           # _State 单例 + in-process AnthropicAdapter（thread）+ fan_out_sample_segments
├── sandbox.py            # boot_agent_sandbox / install_node22 / install_claude_code / run_claude_code / evaluate
└── aiohttp_threaded.py   # run_app_in_thread
```

数据结构差异：
- 现有 `TurnRecord`/`TurnSegment`/`TokenSegment` + `merge_turns` 是 chain-merge
  路径，由 `_select_chain` 在 main / active_sub 间分流。
- src_v2 `TrajectoryManager` + `Node` 是 per-sid 树，append_turn 不分 main/sub
  ——同一棵树通过 DFS 把 sub-agent / compaction 分支自动 fork。

### src_v2 设计要点

- TrajectoryManager 只接收 OpenAI-shape `prompt_messages` + `tools`，配合
  sglang 端的 `prompt_ids/response_ids/logprobs/finish_reason`，存到
  assistant leaf 节点上。
- `get_trajectory(sid, base_sample, reward, drop=True)` 直接返回 `list[Sample]`
  ——一个 leaf 一个 Sample，reward 按 leaves 数均分。
- TITO drift 由 LCP 在 linearize 时处理（每个 turn 取 prompt 的 LCP 长度，
  drop 之前 emit 的 drift 区域 + previous-turn response，emit 本 turn
  prompt[LCP:] 作 loss_mask=0、response 作 loss_mask=1）。

## 3. 架构方案

### 3.1 目标文件布局

```
slime/agent/
├── trajectory_manager.py        # 新增：src_v2 复制 + 微调（保留核心算法）
├── trajectory.py                # 保留：OpenAI adapter 仍依赖 TurnRecord/TurnSegment
├── adapters/
│   ├── common.py                # 微调：可选共享 manager 给 anthropic
│   ├── anthropic.py             # 改造：使用 TrajectoryManager
│   └── openai.py                # 不动
└── ...

examples/coding_agent_rl/
├── generate.py                  # 改造：adapter.finish_session → list[Sample]，移除 fan_out_sample_segments
├── sandbox.py                   # 不动
├── middleware.py                # 新增：src_v2 复制 + 简化（去 sglang proxy / future，去 SLIME_ROOT hack）
├── middleware_debug.py          # 新增：src_v2 复制
├── trajectory_manager_debug.py  # 新增：src_v2 复制
└── launch_swe.py                # 新增：src_v2 复制 + 路径适配

tests/
└── test_trajectory_manager.py   # 新增：src_v2 复制
```

> 关键决策：**保留 `slime/agent/trajectory.py`**——它被 OpenAI adapter
> 用着。这次只改造 Anthropic 路径（SWE 工作流唯一走的路径）。两条 chain
> 并存，OpenAI adapter 不动。

### 3.2 in-process vs external middleware

slime 当前在 `examples/coding_agent_rl/generate.py` 用 in-process 模式
（`run_app_in_thread`）；src_v2 是 external middleware 进程 +
`launch_swe.py` spawn 之。**两条路径都保留**：

- **训练时**：generate.py in-process adapter，每个 sample 一个 sid，
  `finish_session(sid)` 直接拿 `list[Sample]`。
- **debug/swe-launch 时**：`launch_swe.py` 起独立 middleware 进程（`:18080`）+
  N 个 sandbox worker；外部 `POST /get_trajectory?sid=...` 拉样本。

两条路径**复用同一个 `TrajectoryManager` 和 `AnthropicAdapter`**——adapter
内部维护一个 TrajectoryManager，无论 in-process 还是 external middleware
模式下都通过 `adapter.finish_session(sid)` 同一接口 drain。

### 3.3 关键简化：取消 src_v2 middleware 的 sglang proxy

src_v2 middleware.py 用了一个绕弯路：让嵌入的 AnthropicAdapter 把
sglang_url 指向 middleware 自己的 `self_url`，再加 `/generate` proxy
转发到真实 sglang，靠 `asyncio.Future` + `(sid, turn_n)` 配对来在 handler
外面拼装 turn。这是因为 src_v2 没改 adapter，adapter 内部看不到 prompt_ids /
response_ids。

**slime 迁移版**：adapter 内部直接调 `manager.append_turn(sid, ...)`，所以
middleware 不再需要 sglang proxy / future / X-SMG-Routing-Key 协调。
middleware.py 退化为：

1. 起 aiohttp server，嵌 `AnthropicAdapter.app`，把 `--sglang-url` 直接
   传给 adapter（adapter 直连真实 sglang）。
2. 装一个 `dump_mw` middleware：scrub Claude Code billing-header、fold
   mid-list system。
3. 加 `/get_trajectory` route：调 `adapter.finish_session(sid)` 然后
   render 成 JSON。
4. 加 `/healthz`、`/v1/models`、`/v1/messages/count_tokens`（adapter 已有，复用）。
5. 可选 debug install（`--run-dir`）：注入 hook 到 adapter，
   per-turn dump artifact + drain 前后 dump tree/trajectory。

SSE capture（per-task isolation）继续保留，因为 debug 模式要还原 wire SSE。

### 3.4 Adapter 改造

`slime/agent/adapters/anthropic.py` 改造后的 `_handle_request` 流程：

```
body = await request.json()
sid = _request_session_id(request)
# 1. translate Anthropic → OpenAI shape
translated = _translate_anthropic(body.messages, body.system)
tools = _anthropic_tools_to_chat_tools(body.tools)
# 2. render prompt_ids（chain memo 可保留以减少重复 render，初版不带 memo 也行）
prompt_ids = render_token_ids(translated, tokenizer, tools=tools, add_generation_prompt=True)
# 3. call sglang
gen = await call_sglang_generate(prompt_ids, session, body, app, session_id=sid)
# 4. parse output → blocks
blocks, stop, _ = _build_reply(gen.output_text or tok.decode(gen.output_ids), gen.finish_reason, tools, app)
# 5. assemble response_message (OpenAI shape) for trajectory_manager
response_message = _build_response_message(parsed, gen.output_text)
# 6. append_turn
adapter.manager.append_turn(
    sid,
    prompt_messages=translated,
    tools=tools,
    prompt_ids=prompt_ids,
    response_ids=gen.output_ids,
    response_logprobs=gen.output_log_probs,
    response_message=response_message,
    finish_reason=stop_or_finish,
)
# 7. write SSE / json response（不变）
```

`AnthropicAdapter` 增加属性：
```python
self.manager = TrajectoryManager(tokenizer=tokenizer)
```

`finish_session(sid)` 改为：
```python
async def finish_session(self, sid, *, base_sample=None, reward=0.0, wait_timeout=5.0):
    await self.shutdown_session(sid, wait_timeout=wait_timeout)
    return self.manager.get_trajectory(sid, base_sample=base_sample, reward=reward)
```

返回直接是 `list[Sample]`（已经填好 tokens/loss_mask/logprobs/reward/group_id/
metadata），调用方不再走 `fan_out_sample_segments`。

旧的 `Session` 里的 `main` / `active_sub` / `pending_dispatch_id` / `segments`
/ `_select_chain` / `_replace_chat_messages` / `_extend_chat_messages` / `Chain`
（in adapter.py 的旧逻辑）整体删除——它们的语义被 TrajectoryManager 的 DFS
自动取代。

`AdapterChain` 在 `common.py` 中保留供 OpenAI adapter 使用（不动）。

### 3.5 OpenAI adapter

不动。继续用 `TurnRecord` / `TurnSegment` / `merge_turn_segments`。OpenAI
路径不在本次 SWE 工作流中。

### 3.6 examples/coding_agent_rl/generate.py 改造

```python
# 旧
segments = await state.adapter.finish_session(session_id)
fanned = fan_out_sample_segments(sample, segments, reward, tokenizer, metadata=trajectory_metadata)
return fanned

# 新
base = sample  # 已有 index/group_id/prompt/label/metadata
samples = await state.adapter.finish_session(
    session_id,
    base_sample=base,
    reward=reward,
)
if not samples:
    return _abort_result(sample, "adapter_session_empty")
# trajectory_metadata 注入（is_solved / applied_cleanly / elapsed_sec）
for s in samples:
    s.metadata = {**(s.metadata or {}), **trajectory_metadata}
    # 训练侧需要 response 文本：从 tokens[-response_length:] 解出
    rlen = s.response_length or 0
    s.response = tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False) if rlen else ""
return samples
```

> 注意：`TrajectoryManager.get_trajectory` 不填 `sample.response`（它只
> 填 `tokens / response_length / loss_mask / rollout_log_probs`）。slime
> 训练管线读 `sample.response` 做 logging，所以 generate.py 这里补一句解码。

### 3.7 middleware.py 简化与适配

迁移自 src_v2 + 以下改动：

- 删 `SLIME_ROOT = "/mnt/jingshenghang/code/slime_opensource"` 的 sys.path
  hack（已在 slime 里）。
- 不再起 sglang_proxy / sglang_abort_proxy / turn_futures —— adapter 直连
  真实 sglang。
- 不再做 SSE → OpenAI message 反解析（adapter 内部已有 OpenAI-shape 数据）。
- `dump_mw` 退化为：billing-header scrub + mid-list system fold + per-turn
  debug dump（可选）。
- SSE capture 仍保留（per-task `_SSE_CAPTURE` 字典 + 全局
  `web.StreamResponse.write` patch），供 debug 模式 dump wire SSE 用。
- `/get_trajectory` route：调 `adapter.finish_session(sid, base_sample=Sample(...),
  reward=...)`，render JSON 返回。
- 保留 max_turns_per_sid 429 限流。
- 保留 conditional debug install（`--run-dir`）+ `middleware_debug.install(...)`
  注册 hook。

### 3.8 middleware_debug.py / trajectory_manager_debug.py

逐字搬过去，不改逻辑。中间一些 import 路径：

- middleware_debug.py `from trajectory_manager_debug import ...` ——
  保持同目录 import，需要 examples/coding_agent_rl 在 `sys.path`（launch_swe
  / middleware 起动时已经加进去）。

### 3.9 launch_swe.py 适配

迁移自 src_v2，改动：

- 删 SLIME_ROOT hack。
- DEFAULT 路径常量保留（用户那台机器上的路径），但都暴露 CLI flag 可改。
- `mw_path = Path(__file__).resolve().parent / "middleware.py"`——同目录。
- 默认 model / dataset / tarball 路径以参考训练脚本
  (`coding-agent-rl-4-nodes.sh`) 为准。

## 4. 数据流

### 4.1 in-process（训练）

```
slime train loop
  ↳ generate(args, sample, sp)
        ↳ _State (singleton): tokenizer + AnthropicAdapter + run_app_in_thread
        ↳ adapter.open_session(sid)
        ↳ sandbox.boot_agent_sandbox(image) → run_claude_code(adapter_url, sid)
              ↳ E2B sandbox → claude-code CLI → POST /v1/messages → adapter._handle_request
                    ↳ render prompt_ids → call_sglang_generate → manager.append_turn(sid, ...)
                    ↳ stream SSE response back
              ↳ ...N turns... → /done marker
        ↳ git_diff → evaluate → reward
        ↳ samples = adapter.finish_session(sid, base_sample=sample, reward=reward) → list[Sample]
        ↳ return samples
```

### 4.2 external middleware（launch_swe.py）

```
launch_swe.py
  ↳ start_shared_middleware (subprocess: python middleware.py --port 18080 --sglang-url ...)
        ↳ middleware: AnthropicAdapter + scrub middleware + /get_trajectory + debug
  ↳ N × asyncio worker
        ↳ E2B sandbox → install Node + cc → POST /v1/messages
              ↳ middleware → AnthropicAdapter._handle_request → manager.append_turn(sid)
        ↳ POST /get_trajectory?sid=<bearer> → middleware → adapter.finish_session(sid)
              → JSON sample list
        ↳ stash mw_dir/<sid>/trajectory.json / trajectory_tree.json
```

## 5. 错误处理

- adapter 不变的：sglang 失败抛 RuntimeError；client cancel 走 fire-and-forget
  /abort_request。
- append_turn 失败：log 但不挂掉 handler（保证 SSE 返回 claude-code）。
  src_v2 已是这种行为。
- get_trajectory 失败：HTTP 500 + JSON error；launch_swe.py per-worker
  捕获。
- max_turns_per_sid：middleware 返 429，cc 客户端在 sandbox 内退出。
- E2B 失败：sandbox.py 已有 retry + `_is_transient_rpc_error`，不动。

## 6. 测试

- `tests/test_trajectory_manager.py`：从 src_v2 复制，验证：
  - append_turn / get_trajectory 单 leaf 行为
  - DFS 合并（相同 prefix 路由到同节点）
  - 多 leaf fork（compaction / sub-agent）
  - LCP drop-and-replace 在 TITO drift 下的行为
- 单元测试不依赖 sglang，只对 manager 行为做端到端断言。

集成验证：
- 跑 `launch_swe.py --limit 20 --concurrency 20`，要求 20 个 sid 都
  `num_samples >= 1`，trajectory_tree.json 非空。

## 7. 不在本次 scope

- OpenAI adapter 改造（继续走 TurnRecord/TurnSegment）。
- src_v2 的 `prefix_merging.py` / `merge_tree.py` / `launch_demo.py` / `probe_host_ip.py`
  / `test_middleware_inproc.py`——不迁移（前两个是旧版 trajectory；
  launch_demo 是更轻的 demo，没要求；probe_host_ip 是一次性工具；
  test_middleware_inproc.py 测的是 src_v2 那套 middleware 完整 wire 协议，
  改 adapter 后协议不一样了，迁过去会大改）。
- 训练脚本（`coding-agent-rl-4-nodes.sh`）不动——只要 generate.py /
  adapter.finish_session 接口保持兼容即可。
- E2B sandbox 实现（slime/agent/sandbox.py）不动。
