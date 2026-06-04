# Implementation Plan: src_v2 TrajectoryManager 迁移

**Reference**: `docs/superpowers/specs/2026-06-03-trajectory-manager-migration-design.md`
**Branch**: `trajectory-manager-migration-v2` from origin/main `d719f036`

## 步骤总览（按依赖排序）

1. **复制 src_v2 三个核心 .py 到 slime**（trajectory_manager + 两个 debug 文件）
2. **改造 `slime/agent/adapters/anthropic.py`**：嵌入 TrajectoryManager
3. **改造 `examples/coding_agent_rl/generate.py`**：用新 `finish_session` 返回 `list[Sample]`
4. **复制 + 简化 `middleware.py`** 放到 `examples/coding_agent_rl/`
5. **复制 + 适配 `launch_swe.py`** 放到 `examples/coding_agent_rl/`
6. **写测试 `tests/test_trajectory_manager.py`**（从 src_v2 复制）
7. **验证**：跑测试 + 跑 `launch_swe.py --limit 1` 起单 sandbox 看通路 + 跑 `--limit 20 --concurrency 20` 验交付标准

## Step 1: 复制 src_v2 三个核心 .py

`cp` 三份：

```
src_v2/trajectory_manager.py        → slime/slime/agent/trajectory_manager.py
src_v2/middleware_debug.py          → slime/examples/coding_agent_rl/middleware_debug.py
src_v2/trajectory_manager_debug.py  → slime/examples/coding_agent_rl/trajectory_manager_debug.py
```

调整：

- `trajectory_manager.py`：保持 src_v2 原文，只确保 `from slime.utils.types import Sample` 在 slime 仓里 work（已 work，slime 自身的 trajectory_manager 也用 `slime.utils.types.Sample`，路径一致）。
- `middleware_debug.py`：保持原文。`from trajectory_manager_debug import ...` 在 install 处，只要 examples/coding_agent_rl 加进 sys.path，import 就能解。
- `trajectory_manager_debug.py`：保持原文。

## Step 2: 改造 `slime/agent/adapters/anthropic.py`

替换关键部分：

### 2.1 移除旧 segment/chain 机制

删除：
- `_SUBAGENT_TOOLS`
- `Session` dataclass（adapter.py 内的版本）
- `_select_chain`、`_start_sub_chain`、`_replace_chat_messages`、`_extend_chat_messages`、`_build_prompt`
- `AnthropicAdapter.finish_session`（旧的 `merge_turn_segments`）

### 2.2 新增/改写

```python
from slime.agent.trajectory_manager import TrajectoryManager

@dataclasses.dataclass
class Session:
    sampling_defaults: dict = dataclasses.field(default_factory=dict)
    max_context_tokens: int = 0
    lock: asyncio.Lock = dataclasses.field(default_factory=asyncio.Lock)


class AnthropicAdapter(BaseAdapter):
    session_cls = Session

    def __init__(self, *, tokenizer, sglang_url, tool_parser=None, reasoning_parser=None) -> None:
        super().__init__(
            tokenizer=tokenizer, sglang_url=sglang_url,
            tool_parser=tool_parser, reasoning_parser=reasoning_parser,
        )
        # ONE manager shared across all sids — per-sid trees inside.
        self.manager = TrajectoryManager(tokenizer=tokenizer)
        self.app.router.add_post("/v1/messages", _handle_request)
        self.app.router.add_post("/v1/messages/count_tokens", _count_tokens)
        self.app.router.add_get("/healthz", _ok)
        self.app.router.add_get("/v1/models", _ok)

    async def finish_session(
        self,
        sid: str,
        *,
        base_sample=None,
        reward: float = 0.0,
        wait_timeout: float = 5.0,
    ):
        await self.shutdown_session(sid, wait_timeout=wait_timeout)
        # store.pop ensures no dangling per-sid Session
        self.store.pop(sid, None)
        return self.manager.get_trajectory(sid, base_sample=base_sample, reward=reward)
```

### 2.3 改写 `_handle_request`

```python
async def _handle_request(request):
    body = await request.json()
    sid = _request_session_id(request)
    adapter = request.app[ADAPTER_KEY]
    if sid in adapter.closed:
        return web.Response(status=503, text="session closed")
    app = request.app
    tok = app[TOKENIZER_KEY]
    s = adapter.store.setdefault(sid, Session())
    task = asyncio.current_task()
    adapter.inflight.setdefault(sid, set()).add(task)
    try:
        async with s.lock:
            translated = _translate_anthropic(body.get("messages") or [], body.get("system"))
            tools_schema = _anthropic_tools_to_chat_tools(body.get("tools"))
            prompt_ids = render_token_ids(translated, tok, tools=tools_schema, add_generation_prompt=True)
            turn = await call_sglang_generate(
                prompt_ids, s, body, app,
                max_token_keys=("max_tokens",), stop_keys=("stop_sequences",),
                log_prefix="anthropic_adapter", logger=logger, session_id=sid,
            )
            # parse + build reply
            raw_output = tok.decode(turn.output_ids, skip_special_tokens=False) if turn.output_ids else ""
            parsed = parse_model_output(
                raw_output, tools_schema=tools_schema,
                tool_parser_name=app[TOOL_PARSER_KEY],
                reasoning_parser_name=app[REASONING_PARSER_KEY],
            )
            blocks, stop_reason, response_message = _build_blocks_and_response_message(parsed, turn.finish_reason)

            # append to trajectory manager
            try:
                adapter.manager.append_turn(
                    sid,
                    prompt_messages=translated,
                    tools=tools_schema,
                    prompt_ids=prompt_ids,
                    response_ids=turn.output_ids,
                    response_logprobs=turn.output_log_probs,
                    response_message=response_message,
                    finish_reason=_finish_reason_from_turn(turn.finish_reason, parsed.tool_uses),
                    metadata={"sid": sid},
                )
            except Exception:
                logger.exception("append_turn(%s) failed", sid)

            in_tok, out_tok = len(prompt_ids), len(turn.output_ids)
        # SSE / json response (existing helpers, signature unchanged)
        if body.get("stream") is True or "text/event-stream" in request.headers.get("Accept", ""):
            return await _stream_response(request, blocks, stop_reason, in_tok, out_tok)
        return web.json_response(_message_response(body, blocks, stop_reason, in_tok, out_tok))
    finally:
        adapter.inflight.get(sid, set()).discard(task)
```

辅助：

```python
def _build_blocks_and_response_message(parsed, finish):
    blocks = []
    if parsed.reasoning:
        blocks.append({"type": "thinking", "thinking": parsed.reasoning})
    if parsed.text:
        blocks.append({"type": "text", "text": parsed.text})
    tcs_for_msg = []
    for tu in parsed.tool_uses:
        tu_id = f"toolu_{secrets.token_hex(8)}"
        blocks.append({"type": "tool_use", "id": tu_id, "name": tu["name"], "input": tu["input"]})
        tcs_for_msg.append({
            "id": tu_id,
            "type": "function",
            "function": {
                "name": tu["name"],
                # canonicalize arguments to JSON string with sorted keys for stable matching
                "arguments": json.dumps(tu.get("input") or {}, sort_keys=True, ensure_ascii=False),
            },
        })
    if not blocks:
        blocks.append({"type": "text", "text": ""})
    if parsed.tool_uses:
        stop_reason = "tool_use"
    elif finish == "length":
        stop_reason = "max_tokens"
    else:
        stop_reason = "end_turn"
    response_message = {"role": "assistant", "content": parsed.text or ""}
    if parsed.reasoning:
        response_message["reasoning_content"] = parsed.reasoning
    if tcs_for_msg:
        response_message["tool_calls"] = tcs_for_msg
    return blocks, stop_reason, response_message


def _finish_reason_from_turn(finish, tool_uses):
    if tool_uses:
        return "tool_calls"
    return finish or "stop"
```

> `render_token_ids` 的签名在 main 上是 `render_token_ids(chain, tokenizer)` ——
> 接 chain object。adapter 里要么自己组个 chain，要么改成调
> `tokenizer.apply_chat_template`。为了不动 common.py，写一个 thin helper：
>
> ```python
> def render_token_ids(messages, tokenizer, *, tools=None, add_generation_prompt=True):
>     enc = tokenizer.apply_chat_template(
>         messages, tools=tools, tokenize=True, add_generation_prompt=add_generation_prompt,
>     )
>     ids = enc["input_ids"] if hasattr(enc, "__getitem__") and "input_ids" in enc else enc
>     return list(ids)
> ```
>
> 这个 helper 放到 anthropic.py 内部，不冲击 common.py 的同名函数（它服务 OpenAI adapter）。

### 2.4 keep helpers
保留：`_flatten`, `_translate_anthropic`, `_anthropic_tools_to_chat_tools`,
`_request_session_id`, `_message_response`, `_stream_response`,
`_count_tokens`, `_ok`.

## Step 3: 改造 `examples/coding_agent_rl/generate.py`

`_merge_samples` 改造：

```python
def _merge_samples(
    *,
    sample: Sample,
    state: _State,
    samples: list[Sample],
    reward_result: RewardResult,
    elapsed_sec: float,
    instance_id: str,
):
    if not samples:
        return _abort_result(sample, "adapter_session_empty")
    trajectory_metadata = {
        "instance_id": instance_id,
        "is_solved": reward_result.is_solved,
        "applied_cleanly": reward_result.applied_cleanly,
        "elapsed_sec": elapsed_sec,
    }
    for i, s in enumerate(samples):
        s.metadata = {**(s.metadata or {}), **trajectory_metadata, "segment_idx": i, "num_segments": len(samples)}
        # populate sample.response from response tokens (training pipeline reads it)
        rlen = int(s.response_length or 0)
        if rlen and s.tokens:
            s.response = state.tokenizer.decode(s.tokens[-rlen:], skip_special_tokens=False)
        else:
            s.response = ""
    return samples
```

`generate(...)` 调用：

```python
samples = await state.adapter.finish_session(
    session_id,
    base_sample=sample,
    reward=float(reward_result.reward),
)
return _merge_samples(sample=sample, state=state, samples=samples, ...)
```

移除 `from slime.agent.trajectory_manager import TokenSegment, fan_out_sample_segments`
导入；改成 `from slime.utils.types import Sample`（仅用于类型 hint）。

## Step 4: 复制 + 简化 `middleware.py`

放 `examples/coding_agent_rl/middleware.py`。基于 src_v2 改：

1. 删 `SLIME_ROOT` sys.path 注入。
2. 删 sglang_proxy / sglang_abort_proxy / turn_futures / X-SMG-Routing-Key
   配对逻辑。
3. 嵌入 AnthropicAdapter 时直接传 `sglang_url=args.sglang_url`（不是
   self_url）。
4. `dump_mw`：保留 sid 抽取、429 限流、billing-header scrub、mid-list
   system fold；保留 SSE per-task capture（debug 用）；删除 sglang future
   等待 + manager.append_turn 调用——这两件 adapter 内部已经做了。
5. 改用 `adapter.manager`（共享同一个 manager 实例）做 DEBUG hook 的 manager 引用。
6. 仍然 `on_request / on_request_scrubbed / on_sse / on_turn_appended`
   hook 调用——但 on_turn_appended 现在只是 dump 后置事件（adapter 内部已 append），
   不再传 prompt_ids/response_ids 参数（已经存在 manager 里），改 dump
   sid + n + body shape。简化方案：保留 on_request/scrubbed/sse/drain_*；
   去掉 on_sglang_pair 和 on_turn_appended（reduce 复杂度，把 turn 元
   数据放到 on_sse 的 dump 里）。
7. `/get_trajectory`：调 `adapter.finish_session(sid, base_sample=..., reward=...)`；
   `DEBUG.on_drain_start(sid, adapter.manager)` 在 finish 之前 dump tree（注意
   `finish_session` 内会 drop session，所以 dump 必须在调用 finish_session 之前）。

middleware_debug.py 微调：`on_turn_appended` 这个 hook 入参变（或干脆去掉）；
对应 src_v2 的实现里 prompt_messages/tools/response_message/prompt_ids/response_ids/finish_reason
还想要的话，先在 dump_mw 里组装好再调 hook。**简化路径**：保留 src_v2 原版
middleware_debug 不动，由 middleware.py 在 _handle_request 处理后调
on_turn_appended 并构造参数（需要在 adapter 之外重做 translate / render，但
我们正好已经有这些）。

> 决策：**middleware.py 自己接管 trajectory dump 的部分**：因为现在 adapter
> 内部直接 append_turn，middleware 想 dump 需要在外面 re-translate + re-render
> 一遍——开销不小。**简化**：不再单独 `turn_NNNN_openai.json` dump；通过
> `on_drain_start` 把整棵 tree dump 出来，turn-级别细节通过 sse + scrubbed
> request 还原。middleware_debug.py 里 `on_turn_appended` 简化成可选 hook
> （middleware.py 不再 call），其他 hook 不变。

## Step 5: 复制 + 适配 `launch_swe.py`

放 `examples/coding_agent_rl/launch_swe.py`。基于 src_v2 改：

1. 删 `SLIME_ROOT` sys.path hack。
2. `mw_path = Path(__file__).resolve().parent / "middleware.py"` —— 同目录。
3. DEFAULT_NODE_TGZ / DEFAULT_CC_TGZ / DEFAULT_DATASET / DEFAULT_MODEL：保留
   src_v2 的默认值（用户机器上的路径），但参考训练脚本，CC tarball 改为
   `/mnt/jingshenghang/storage/claude_code/anthropic-ai-claude-code-2.1.143-local-linux-x64.tgz`
   （训练脚本用的），夹带 fallback 到 src_v2 default。
4. `host_ip` default：保留 src_v2 default（用户用 CLI 改）。
5. `ensure_e2b_env`：保留——设 dummy E2B_API_KEY + 默认 metadata file。
6. `from slime.agent.sandbox import E2BSandbox`：已经在 slime 里能 import。
7. drain 部分：HTTP POST 不变（middleware 暴露的接口）。
8. 仍然按 sid 写 `runs/swe/<ts>/inst_<idx>/summary.json` + `mw/<sid>/...`。

## Step 6: 测试

`tests/test_trajectory_manager.py`：从 src_v2 复制全部测试，只调整 import：

```python
from slime.agent.trajectory_manager import TrajectoryManager, Node, node_match_key
```

跑：`pytest tests/test_trajectory_manager.py -x -v`。

> src_v2 的 `test_trajectory_manager.py` 640 行，应该覆盖 manager 核心
> 行为（append / DFS / get_trajectory / TITO drift）。复制过来直接跑。
>
> 注意 src_v2 还有 `test_manager.py`（344 行）和
> `test_trajectory_manager_debug.py`（96 行）。两个也带上：放到
> `tests/test_trajectory_manager_debug.py` 和 `tests/test_trajectory_tree.py`
> （前者纯 debug 模块；后者基于 manager 的更高层断言）。

## Step 7: 验证

### 7.1 单测
```bash
cd /mnt/jingshenghang/code/slime_swe/slime
python -m pytest tests/test_trajectory_manager.py -x -q
python -m pytest tests/test_trajectory_manager_debug.py -x -q
```

### 7.2 小规模 smoke（middleware + 1 sandbox + 1 sid）

```bash
# 需要本地或某台机器有 sglang router 起着（端口 30000，model 与 launch_swe.py --model 一致）
# 起 middleware 不依赖 sglang 是否在；但 reachability check 需要 sandbox 能 curl 通 middleware
python examples/coding_agent_rl/launch_swe.py \
    --limit 1 \
    --concurrency 1 \
    --sglang-url http://127.0.0.1:30000 \
    --host-ip <local host ip>
```

通过条件：
- middleware healthz 200
- E2B sandbox 启动、Node + cc 安装成功
- /v1/messages 至少 1 个 turn
- /get_trajectory 返回 num_samples >= 1
- summary.json `tree.found == True`、tree.turns > 0

### 7.3 20 并发（交付标准）

```bash
python examples/coding_agent_rl/launch_swe.py \
    --limit 20 \
    --concurrency 20 \
    --sglang-url http://127.0.0.1:30000 \
    --host-ip <local host ip>
```

通过条件：
- batch summary.txt 显示 20 instance，`ok >= ~16`（允许 ~20% sandbox 抽风）
- `with_tito_drop` 在合理范围（小数字是正常的，多数 trajectory 不会 drift）
- runs/swe/<ts>/mw/<sid>/trajectory.json + trajectory_tree.json 都生成

如果 sglang 在本地起不起来：把 sglang_url 指到训练集群上正在运行的实例
（或者跑一个 GLM-mini sglang）。如果 sandbox dial-back IP 不通：
prove_host_ip.py 等价的逻辑——sandbox 内 `curl http://<HOST_IP>:18080/healthz`
应该 200。

## 风险 / 已知 trade-off

- **render 性能**：src_v2 middleware 不做 render memo（因为它本来就用
  adapter render）；slime 新 anthropic adapter 也没 memo 一行行 chain.
  Per-turn render 成本 = `O(prompt_len)`，accept。
- **OpenAI adapter 不动**：它继续走 TurnRecord/TurnSegment。未来如果要
  统一到 TrajectoryManager，单独的 PR。
- **finish_session 签名变了**：从 `→ list[TokenSegment]` 改为
  `→ list[Sample]`。OpenAI adapter 的 finish_session 仍返回 list[TokenSegment]
  ——签名不兼容。这是 anthropic-only 改造，acceptable。
- **训练脚本 SWE_LIST_TRAJECTORY**：脚本里有 `export SWE_LIST_TRAJECTORY=1`
  这种环境变量是历史遗留；新 generate.py 直接 return list[Sample]，
  不再读这个 flag。等于 SWE_LIST_TRAJECTORY=1 现在是默认行为。
