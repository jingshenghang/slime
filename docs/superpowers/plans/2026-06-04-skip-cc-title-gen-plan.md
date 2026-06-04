# Skip cc title-generation from trajectory Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `AnthropicAdapter._handle_request` 识别并跳过 Claude Code 的 title-generation 请求的 `manager.append_turn(...)`，使这类 meta 请求不进 trajectory tree / 不产出训练 sample，但保留所有 dump（含 `turn_NNNN_openai.json`，因为 `on_turn_appended` hook 仍调）。

**Architecture:** 仅 adapter 层一处 guard：识别条件 AND 合取 — `tools_schema` 为 falsy AND `translated` 开头若干 `role=system` 块文本中至少一处含 `"Generate a concise, sentence-case title"`。识别命中则跳过 append_turn 那一段 try 块（hook 仍调用）；其他路径完全不动。

**Tech Stack:** Python 3.12 / aiohttp / pytest / 项目内部 `slime.agent.adapters.anthropic`。

**Spec:** [`docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md`](../specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md)

---

## File Structure

- `slime/agent/adapters/anthropic.py` — 加 module-level 常量 `_CC_TITLE_GEN_MARKER` + helper `_is_cc_title_generation_request(translated, tools_schema)` + `_handle_request` 中一处 if-guard 包裹现有 `append_turn` try 块。约 +30 行。
- `tests/test_agent_adapters.py` — 加 3 个识别函数 unit test + 1 个 `_handle_request` integration test（验证 title-gen 不进 manager 但 hook 仍调）。约 +120 行。

任何其他文件都不动（含 `TrajectoryManager`、其他 adapter、middleware）。

---

### Task 1: 识别函数 — 写失败测试

**Files:**
- Test: `tests/test_agent_adapters.py`（追加到文件末尾）

- [ ] **Step 1: 在 test 文件末尾追加 3 个识别函数 unit test**

把下面整段追加到 `tests/test_agent_adapters.py` 末尾（不要替换已有内容）：

```python


# ---------------------------------------------------------------------------
# _is_cc_title_generation_request — detect cc per-session title-gen requests
# (spec: docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md)
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_is_cc_title_generation_request_detects_title_gen_body():
    # cc title-gen request: NO tools + system block contains the marker string.
    translated = [
        {
            "role": "system",
            "content": (
                "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n"
                "Generate a concise, sentence-case title (3-7 words) that captures the user's task."
            ),
        },
        {"role": "user", "content": "Read PROBLEM_STATEMENT.md and fix the bug."},
    ]
    tools_schema = None  # cc sends tools=[] -> _anthropic_tools_to_chat_tools returns None
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is True


@pytest.mark.unit
def test_is_cc_title_generation_request_rejects_main_conversation():
    # cc main agent request: has tools AND no title-gen marker. Must NOT match.
    translated = [
        {
            "role": "system",
            "content": (
                "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n"
                "You are an interactive agent that helps users with software engineering tasks."
            ),
        },
        {"role": "user", "content": "<system-reminder>...</system-reminder>"},
    ]
    tools_schema = [
        {"type": "function", "function": {"name": "Read", "description": "", "parameters": {}}},
        {"type": "function", "function": {"name": "Edit", "description": "", "parameters": {}}},
    ]
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is False


@pytest.mark.unit
def test_is_cc_title_generation_request_rejects_subagent():
    # cc sub-agent request: has tools AND no title-gen marker. Must NOT match.
    translated = [
        {
            "role": "system",
            "content": (
                "You are a Claude agent, built on Anthropic's Claude Agent SDK.\n"
                "You are a file search specialist for Claude Code, Anthropic's official CLI for Claude."
            ),
        },
        {"role": "user", "content": "Find files related to authentication."},
    ]
    tools_schema = [
        {"type": "function", "function": {"name": "Grep", "description": "", "parameters": {}}},
    ]
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is False


@pytest.mark.unit
def test_is_cc_title_generation_request_handles_block_list_system_content():
    # cc may send system as a list of blocks (anthropic SDK shape).
    # The marker may live inside one of the blocks; helper must scan all of them.
    translated = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "x-anthropic-billing-header: cc_version=2.1.143"},
                {"type": "text", "text": "You are a Claude agent."},
                {
                    "type": "text",
                    "text": "Generate a concise, sentence-case title (3-7 words) ...",
                },
            ],
        },
        {"role": "user", "content": "hi"},
    ]
    assert anthropic._is_cc_title_generation_request(translated, None) is True


@pytest.mark.unit
def test_is_cc_title_generation_request_rejects_when_tools_present_even_if_marker():
    # AND-conjunction: tools present -> never title-gen, even if marker text leaks in.
    translated = [
        {
            "role": "system",
            "content": "Generate a concise, sentence-case title (3-7 words) ...",
        },
        {"role": "user", "content": "hi"},
    ]
    tools_schema = [{"type": "function", "function": {"name": "Read", "description": "", "parameters": {}}}]
    assert anthropic._is_cc_title_generation_request(translated, tools_schema) is False
```

- [ ] **Step 2: 跑测试确认 fail**

Run:
```bash
cd /mnt/jingshenghang/code/slime_swe/slime
pytest tests/test_agent_adapters.py -k _is_cc_title_generation_request -v
```

Expected: `AttributeError: module 'slime.agent.adapters.anthropic' has no attribute '_is_cc_title_generation_request'` 5 个 ERROR。

- [ ] **Step 3: 不 commit**（等实现一起 commit，TDD red-green-refactor 的 red 阶段）。

---

### Task 2: 识别函数 — 写实现使测试通过

**Files:**
- Modify: `slime/agent/adapters/anthropic.py`（在 line 148 `_translate_anthropic` 之前加常量 + 函数）

- [ ] **Step 1: 在 `_translate_anthropic` 函数定义之前插入常量和 helper**

打开 `slime/agent/adapters/anthropic.py`，找到这行（约 line 148）：

```python
def _translate_anthropic(msgs: list[dict], system: Any) -> list[dict]:
```

在它**之前**插入以下代码：

```python
# Marker string Claude Code embeds in the system prompt of its per-session
# title-generation request (a meta request that asks the LLM to produce a
# short conversation title). Title-gen requests should NOT enter the RL
# trajectory — they aren't agent work. See spec
# docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md.
_CC_TITLE_GEN_MARKER = "Generate a concise, sentence-case title"


def _is_cc_title_generation_request(
    translated: list[dict],
    tools_schema: list[dict] | None,
) -> bool:
    """Return True iff this is a Claude Code per-session title-generation request.

    Detection is AND-conjunction:
      (1) ``tools_schema`` is falsy (cc sends tools=[]; converter returns None).
      (2) one of the leading ``role=system`` messages' content contains
          ``_CC_TITLE_GEN_MARKER``.

    Scanning stops at the first non-system message — title-gen system blocks
    always sit at the head of the request.
    """
    if tools_schema:
        return False
    for msg in translated:
        if msg.get("role") != "system":
            break
        content = msg.get("content")
        if isinstance(content, str):
            if _CC_TITLE_GEN_MARKER in content:
                return True
        elif isinstance(content, list):
            for block in content:
                if (
                    isinstance(block, dict)
                    and isinstance(block.get("text"), str)
                    and _CC_TITLE_GEN_MARKER in block["text"]
                ):
                    return True
    return False
```

- [ ] **Step 2: 跑测试确认 5 个 pass**

Run:
```bash
pytest tests/test_agent_adapters.py -k _is_cc_title_generation_request -v
```

Expected: 5 passed。

- [ ] **Step 3: 跑回归确认没破其他**

Run:
```bash
pytest tests/test_agent_adapters.py tests/test_trajectory_manager.py -q
```

Expected: 全部既有 + 5 个新 → all pass / 部分既有 skip 不变。

- [ ] **Step 4: pre-commit + commit**

```bash
pre-commit run --files slime/agent/adapters/anthropic.py tests/test_agent_adapters.py 2>&1 || true
git add slime/agent/adapters/anthropic.py tests/test_agent_adapters.py
git commit -m "$(cat <<'EOF'
feat(anthropic-adapter): add _is_cc_title_generation_request helper

cc fires a per-session title-generation request on startup (tools=[], system
contains "Generate a concise, sentence-case title..."). Those are meta requests
that should not enter the RL trajectory. This commit only adds the detector +
unit tests; the call-site guard lands in the next commit.
EOF
)"
```

(若 pre-commit 自动改了文件，重新 git add 再 commit；不要 --amend，按 CLAUDE.md 全局规则。)

---

### Task 3: `_handle_request` 接入 guard — 写失败 integration 测试

**Files:**
- Test: `tests/test_agent_adapters.py`（追加到文件末尾）

- [ ] **Step 1: 追加 integration test 验证 `_handle_request` 跳过 append_turn 但调 hook**

把以下整段追加到 `tests/test_agent_adapters.py` 末尾：

```python


# ---------------------------------------------------------------------------
# _handle_request: title-gen requests skip append_turn but still fire hook
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_anthropic_handle_request_skips_append_turn_for_title_gen(monkeypatch):
    """A cc title-gen request must NOT enter manager._trees, but the
    on_turn_appended hook MUST still fire so per-turn dumps (openai.json,
    etc.) keep being written."""

    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=[42, 43, 44],
            finish_reason="stop",
            output_log_probs=[-0.1, -0.2, -0.3],
        )

    monkeypatch.setattr(anthropic, "call_sglang_generate", fake_generate)

    hook_calls: list[tuple] = []

    def hook(sid, prompt_messages, tools, response_message,
             prompt_ids, response_ids, finish_reason):
        hook_calls.append((sid, len(prompt_ids), len(response_ids), finish_reason))

    tok = ToyTokenizer(outputs={(42, 43, 44): '{"title": "Fix the bug"}<|im_end|>'})

    adapter = anthropic.AnthropicAdapter(
        tokenizer=tok,
        sglang_url="http://127.0.0.1:1",
        on_turn_appended=hook,
    )

    # Build a cc-style title-gen request body.
    title_gen_body = {
        "messages": [{"role": "user", "content": "Read PROBLEM_STATEMENT.md and fix."}],
        "system": [
            {"type": "text", "text": "You are a Claude agent."},
            {
                "type": "text",
                "text": "Generate a concise, sentence-case title (3-7 words) ...",
            },
        ],
        "tools": [],
        "max_tokens": 512,
        "stream": False,
    }

    async def run():
        app = adapter.app
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            r = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer test-title-gen-sid"},
                json=title_gen_body,
            )
            assert r.status == 200, await r.text()
        finally:
            await client.close()

    asyncio.run(run())

    # The manager MUST NOT have learned about this sid (no tree was opened).
    assert "test-title-gen-sid" not in adapter.manager._trees, (
        f"title-gen request leaked into manager: trees={list(adapter.manager._trees)}"
    )

    # The hook MUST have fired exactly once with this sid.
    assert len(hook_calls) == 1, f"expected 1 hook call, got {hook_calls}"
    assert hook_calls[0][0] == "test-title-gen-sid"
    assert hook_calls[0][2] == 3  # response_ids length matches fake_generate output


@pytest.mark.unit
def test_anthropic_handle_request_still_appends_main_conversation(monkeypatch):
    """Sanity counter-test: a normal (non-title-gen) request MUST still
    populate manager._trees and fire the hook. Prevents an over-broad guard
    from silently dropping real conversation turns."""

    async def fake_generate(prompt_ids, session, body, app, **kwargs):
        return TurnRecord(
            prompt_ids=list(prompt_ids),
            output_ids=[7, 8, 9],
            finish_reason="stop",
            output_log_probs=[-0.1, -0.2, -0.3],
        )

    monkeypatch.setattr(anthropic, "call_sglang_generate", fake_generate)

    hook_calls: list[tuple] = []

    def hook(sid, *rest):
        hook_calls.append(sid)

    tok = ToyTokenizer(outputs={(7, 8, 9): "hello"})

    adapter = anthropic.AnthropicAdapter(
        tokenizer=tok,
        sglang_url="http://127.0.0.1:1",
        on_turn_appended=hook,
    )

    main_body = {
        "messages": [{"role": "user", "content": "Run the tests."}],
        "system": [
            {"type": "text", "text": "You are an interactive agent."},
        ],
        "tools": [
            {
                "name": "Read",
                "description": "Read a file",
                "input_schema": {"type": "object", "properties": {}},
            }
        ],
        "max_tokens": 8000,
        "stream": False,
    }

    async def run():
        app = adapter.app
        client = TestClient(TestServer(app))
        await client.start_server()
        try:
            r = await client.post(
                "/v1/messages",
                headers={"Authorization": "Bearer test-main-sid"},
                json=main_body,
            )
            assert r.status == 200, await r.text()
        finally:
            await client.close()

    asyncio.run(run())

    assert "test-main-sid" in adapter.manager._trees, (
        f"main conversation did NOT reach manager: trees={list(adapter.manager._trees)}"
    )
    assert hook_calls == ["test-main-sid"], hook_calls
```

- [ ] **Step 2: 跑测试确认 fail**

Run:
```bash
pytest tests/test_agent_adapters.py::test_anthropic_handle_request_skips_append_turn_for_title_gen -v
```

Expected: FAIL — `AssertionError: title-gen request leaked into manager: trees=['test-title-gen-sid']`
（实现还没加 guard，title-gen 也会被 append。）

Sanity counter 那个可能直接 pass（因为它就是当前行为），先不强求 fail。

- [ ] **Step 3: 不 commit**（实现一起 commit）。

---

### Task 4: `_handle_request` 接入 guard — 写实现使测试通过

**Files:**
- Modify: `slime/agent/adapters/anthropic.py`（约 line 380-395，`_handle_request` 中 append_turn try 块）

- [ ] **Step 1: 用 guard 包裹 append_turn try 块**

打开 `slime/agent/adapters/anthropic.py`，找到 `_handle_request` 里这段（约 line 378-395）：

```python
            blocks, stop_reason, response_message = _build_blocks_and_response_message(parsed, turn.finish_reason)

            try:
                adapter.manager.append_turn(
                    sid,
                    prompt_messages=translated,
                    tools=tools_schema,
                    prompt_ids=prompt_ids,
                    response_ids=list(turn.output_ids),
                    response_logprobs=list(turn.output_log_probs)
                    if turn.output_log_probs and len(turn.output_log_probs) == len(turn.output_ids)
                    else None,
                    response_message=response_message,
                    finish_reason=_finish_reason_for_manager(turn.finish_reason, parsed.tool_uses),
                    metadata={"sid": sid},
                )
            except Exception:
                logger.exception("append_turn(sid=%s) failed", sid)
```

改成：

```python
            blocks, stop_reason, response_message = _build_blocks_and_response_message(parsed, turn.finish_reason)

            if _is_cc_title_generation_request(translated, tools_schema):
                # Claude Code meta request (per-session title generation).
                # Skip the trajectory so it doesn't pollute the tree / become
                # an RL sample. The on_turn_appended hook below still fires,
                # so per-turn dumps (request, sse, openai.json) keep landing
                # on disk for debugging. See spec
                # docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md.
                logger.info(
                    "skipping append_turn for cc title-generation request (sid=%s)",
                    sid,
                )
            else:
                try:
                    adapter.manager.append_turn(
                        sid,
                        prompt_messages=translated,
                        tools=tools_schema,
                        prompt_ids=prompt_ids,
                        response_ids=list(turn.output_ids),
                        response_logprobs=list(turn.output_log_probs)
                        if turn.output_log_probs and len(turn.output_log_probs) == len(turn.output_ids)
                        else None,
                        response_message=response_message,
                        finish_reason=_finish_reason_for_manager(turn.finish_reason, parsed.tool_uses),
                        metadata={"sid": sid},
                    )
                except Exception:
                    logger.exception("append_turn(sid=%s) failed", sid)
```

**不要动** `hook = adapter.on_turn_appended ...` 那段（紧跟在后面，需保持在 if/else 外面，让两条路径都触发 hook）。

- [ ] **Step 2: 跑 2 个 integration 测试确认 pass**

Run:
```bash
pytest tests/test_agent_adapters.py::test_anthropic_handle_request_skips_append_turn_for_title_gen tests/test_agent_adapters.py::test_anthropic_handle_request_still_appends_main_conversation -v
```

Expected: 2 passed。

- [ ] **Step 3: 全量回归**

Run:
```bash
pytest tests/test_agent_adapters.py tests/test_trajectory_manager.py -q
```

Expected: 全 pass（既有 34 passed + 3 skipped + 新增 5 个 helper test + 2 个 integration test = 41 passed + 3 skipped）。

- [ ] **Step 4: pre-commit + commit**

```bash
pre-commit run --files slime/agent/adapters/anthropic.py tests/test_agent_adapters.py 2>&1 || true
git add slime/agent/adapters/anthropic.py tests/test_agent_adapters.py
git commit -m "$(cat <<'EOF'
feat(anthropic-adapter): skip append_turn for cc title-generation requests

cc's per-session title-gen request (tools=[] + system contains the marker
"Generate a concise, sentence-case title") was being fed to TrajectoryManager
as a separate leaf, producing a spurious short Sample in the RL batch and
adding a phantom system child to trajectory_tree.json.

Guard the append_turn call site with _is_cc_title_generation_request added in
the previous commit. The on_turn_appended hook still fires for title-gen so
turn_NNNN_openai.json / request.json / response.sse dumps continue to land
on disk (debug visibility preserved).

Out of scope: sub-agent requests (have tools, treated as real work).
Spec: docs/superpowers/specs/2026-06-04-skip-cc-title-gen-from-trajectory-design.md
EOF
)"
```

---

### Task 5: 端到端 smoke 验证

**Files:** 无代码改动；纯验证。

- [ ] **Step 1: 起一条 20 并发 batch**

(prerequisite: sglang 在 30000 端口跑；middleware 在 18080；同上轮命令。)

Run:
```bash
cd /mnt/jingshenghang/code/slime_swe/slime
python -m examples.coding_agent_rl.launch_swe --offset 0 --limit 20 --concurrency 20
```

Expected: 跟上轮一样 ok=20，elapsed 接近上轮（不应明显变慢）。

- [ ] **Step 2: 找最新 batch dir**

Run:
```bash
BATCH=$(ls -td /mnt/jingshenghang/code/slime_swe/0603-trajectory-manager/runs/swe/* | head -1)
echo "$BATCH"
```

- [ ] **Step 3: 验 "skipping append_turn" 日志 ≈ 20 条（每个 inst 1 次）**

Run:
```bash
grep -c "skipping append_turn for cc title-generation request" "$BATCH/middleware.log"
```

Expected: 20（或略多，若某个 cc 因 sandbox 重启再发了一次 title-gen）。

- [ ] **Step 4: 验 trajectory_tree.json 不再有 title-gen system 子节点**

Run:
```bash
for d in $BATCH/0*; do
  python3 -c "
import json
t = json.load(open('$d/trajectory_tree.json'))
root = t.get('root') or t
for c in root.get('children') or []:
    msgs = c.get('messages') or []
    if msgs:
        c0 = msgs[0].get('content')
        if isinstance(c0, list) and c0:
            text = str((c0[0] or {}).get('text',''))
        elif isinstance(c0, str):
            text = c0
        else:
            text = ''
        if 'Generate a concise, sentence-case title' in text:
            print(f'BAD $(basename $d): title-gen leaked into tree')
            break
else:
    pass
"
done
```

Expected: 没有 "BAD" 输出（所有 inst 都干净）。

- [ ] **Step 5: 验 trajectory.json 每条 sample 的 response_length 都 >= 一定阈值（title-gen sample 通常 <100）**

Run:
```bash
for d in $BATCH/0*; do
  python3 -c "
import json
samples = json.load(open('$d/trajectory.json'))
for i, s in enumerate(samples):
    rl = s.get('response_length') or 0
    pt = (s.get('prompt_text') or '')[:80]
    if 'Generate a concise, sentence-case title' in pt:
        print(f'BAD $(basename $d) sample[{i}]: title-gen sample leaked, rl={rl}')
        break
"
done
```

Expected: 没有 "BAD" 输出。

- [ ] **Step 6: 验 turn_0001_openai.json 现在出现（hook 仍调）**

Run:
```bash
missing=0
for d in $BATCH/0*; do
  [ -f "$d/turn_0001_openai.json" ] || { echo "MISSING: $(basename $d)/turn_0001_openai.json"; missing=$((missing+1)); }
done
echo "total missing: $missing"
```

Expected: `total missing: 0`（每个 inst 都有 turn_0001_openai.json，证明 hook 跳过 guard 走到了）。

- [ ] **Step 7: 验 sub-agent 仍然进了 trajectory（按设计保留）**

Run（找一个有 sub-agent 的 inst，例如 0004 那种 file_search specialist 出现过的）：
```bash
python3 -c "
import json, glob
for inst_dir in sorted(glob.glob('$BATCH/00*')):
    samples = json.load(open(inst_dir + '/trajectory.json'))
    for s in samples:
        pt = (s.get('prompt_text') or '')[:200]
        if 'file search specialist' in pt:
            print(f'OK sub-agent preserved in {inst_dir.rsplit(\"/\",1)[-1]}')
            break
"
```

Expected: 至少 1 行 "OK sub-agent preserved" —— 证明 sub-agent 没被误伤。

---

## Self-Review

**1. Spec coverage：**
- "识别口径 AND 合取 tools=[] + marker" → Task 1 + Task 2（helper + 5 个 unit test 覆盖 tools=空、有 tools、block-list content、AND 优先级）。✓
- "只跳 append_turn，hook 仍调，dump 保留" → Task 3 integration test 双向断言：`adapter.manager._trees` 不含 sid + `hook_calls` 长度=1。Task 4 实现保留 hook 在 if/else 之外。✓
- "不动 sub-agent" → Task 1 第 3 个 test 显式断言 sub-agent body 不命中。Task 5 step 7 端到端验证。✓
- "不动 TrajectoryManager / 其他 adapter" → Plan 全文未提其他文件。✓
- "验证：trajectory_tree 不含 title-gen、trajectory.json 少一条、openai.json 仍出现" → Task 5 step 3/4/5/6 一一对应。✓
- "Verification 段提到的 `skipping append_turn for cc title-generation request` log" → Task 4 实现里有该 logger.info；Task 5 step 3 grep。✓

**2. Placeholder scan：** 无 TBD/TODO；所有代码 step 都给完整代码；commands 都可直接复制运行。✓

**3. Type consistency：** helper 签名一致（`translated: list[dict], tools_schema: list[dict] | None`）；test 里 `anthropic._is_cc_title_generation_request` 调用与实现命名一致；`TurnRecord` import 已在 test 文件顶部既有（line 16）；`ToyTokenizer` 已在 line 22 既有。✓

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-06-04-skip-cc-title-gen-plan.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — 每个 task 派一个新 subagent，spec-reviewer + code-quality-reviewer 两段 review，task 之间快速迭代。

**2. Inline Execution** — 直接在当前会话执行 Task 1-5，每个 task 后停一下让我 review。

**Which approach?**
