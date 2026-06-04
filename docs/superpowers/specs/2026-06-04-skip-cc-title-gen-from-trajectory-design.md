# Spec: 在 AnthropicAdapter 里跳过 cc title-generation 请求的 append_turn

## 背景

Claude Code（cc）每开一个新会话，启动时会向 `/v1/messages` 发一次额外请求让 LLM 生成一个简短会话标题，特征：

- `tools=[]`（没工具）
- `system` 块里某一块文本包含 `"Generate a concise, sentence-case title (3-7 words)..."`
- `max_tokens` 通常很小（512）
- response 体也是短 JSON（如 `{"title": "Fix issue described in PROBLEM_STATEMENT.md"}`）

这种请求**不是 agent 的真实工作行为**，纯粹是 cc 内部 UI/会话管理用的 meta 请求。**它不应该进强化学习训练**。

当前 `slime/agent/adapters/anthropic.py:373-389` `_handle_request` 把所有走完的请求一律 `adapter.manager.append_turn(...)`。后果（在 `runs/swe/20260604-075235/` 实测）：

1. `trajectory_tree.json` 的 root 多一个 `system` 子节点（title-gen 的 system prompt），变成多叉。
2. `trajectory.json` 多一条 sample（`response_length` 几十到一百），它会被 RL 当作一条独立轨迹。
3. 配套 turn dump 缺 `turn_0001_openai.json` 的副作用。**这个 spec 不解决"为什么 turn 1 没出 openai dump"——它保留 hook 调用，问题如真存在仍需独立排查（见"不在范围内的事"）**。

## 目标 & 非目标

**目标**：
- 识别 cc 的 title-gen 请求并跳过 `adapter.manager.append_turn(...)`。
- 其他 dump 全部保留（request.json / request_scrubbed.json / response.sse / **包括 openai.json**），便于排查 cc 实际发了什么。
- 不破坏任何现有路径；不影响 OpenAI adapter / 其他 client。

**非目标**：
- **不处理** sub-agent（"You are a file search specialist..."）。sub-agent 跟主对话一样有 tools、是真实工作行为，按现状保留。是否训练 sub-agent 是另一个独立话题。
- 不动 `TrajectoryManager`——manager 是纯数据结构，不该知道 cc 协议细节。

## 识别口径

**AND 合取**两个条件同时成立才视为 title-gen：

1. **`tools` 为空**：`tools_schema` 是 `None` 或空 list。
2. **`system` 含 title-gen marker**：translated message list 的开头若干 `role: system` 消息中，任一 `content`（str 或 list-of-blocks 形式）包含子串 `"Generate a concise, sentence-case title"`。

理由：
- 仅靠 marker 字符串 → 太脆弱（cc 改 prompt 就失效）。
- 仅靠 tools=[] → 将来可能误伤合法的"无工具"请求。
- AND 后，**误伤合法工作请求**的概率几乎为零（合法工作请求必带 tools），**漏判 title-gen** 只在 cc 同时换 marker 文本 + 加 tools 时才发生（看不出有什么动机这么做）。

`max_tokens` 阈值放弃：cc 改默认值很常见，引入不必要的脆弱性。

## 实现 Sketch

**仅改 1 个文件**：`slime/agent/adapters/anthropic.py`

```python
_CC_TITLE_GEN_MARKER = "Generate a concise, sentence-case title"


def _is_cc_title_generation_request(
    translated: list[dict],
    tools_schema: list[dict] | None,
) -> bool:
    """Detect Claude Code's per-session title-generation request.

    These are meta requests cc fires once per session to pick a short title;
    they shouldn't enter the RL trajectory. Match requires BOTH: no tools AND
    a leading system message containing the title-gen marker string.
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

`_handle_request`（约 line 380-395）：在 `try: adapter.manager.append_turn(...)` 之前加 guard：

```python
if _is_cc_title_generation_request(translated, tools_schema):
    logger.info(
        "skipping append_turn for cc title-generation request (sid=%s)", sid
    )
else:
    try:
        adapter.manager.append_turn(...)
    except Exception:
        logger.exception("append_turn(sid=%s) failed", sid)
```

`hook = adapter.on_turn_appended` 那段**保持不变**——title-gen 仍会触发 hook，仍会写 `turn_NNNN_openai.json`（用户决定：dump 全保留）。

## 测试

`tests/test_agent_adapters.py` 加 3 个 unit test：

1. `test_is_cc_title_generation_request_detects_title_gen_body`
   - 输入：translated 含一个 system 块文本含 marker，tools_schema=None / []
   - 期望：True

2. `test_is_cc_title_generation_request_rejects_main_conversation`
   - 输入：translated 含正常 cc agent system（无 marker），tools_schema 有 28 个工具
   - 期望：False

3. `test_is_cc_title_generation_request_rejects_subagent`
   - 输入：translated 含 "You are a file search specialist..." system，tools_schema 有 19 个工具
   - 期望：False

4.（可选 integration）`test_anthropic_handle_request_skips_append_turn_for_title_gen`
   - monkey-patch `call_sglang_generate` 返回 dummy turn
   - 用 mocked request 发一个 title-gen body
   - 断言：`adapter.manager._trees` 仍为空（append 没调）；但 `on_turn_appended` 闭包记录被调一次

## 验证（端到端）

跑同一条命令再次 batch：

```bash
python -m examples.coding_agent_rl.launch_swe --offset 0 --limit 20 --concurrency 20
```

期望：

- 每个 `inst_dir/trajectory_tree.json` 的 root 子节点中**不再出现** title-gen system 节点（root 只有"主对话 system" + 可能的"sub-agent system"）。
- 每个 `inst_dir/trajectory.json` 的 sample 总数比当前少 1（无 title-gen sample）。
- 每个 inst 的 `turn_0001_openai.json` 仍然存在（hook 没被跳过）。
- middleware.log 出现 `"skipping append_turn for cc title-generation request (sid=swe-XXXX-...)"` 共 20 行（每 inst 一次）。

## 不在范围内的事

- sub-agent 处理（保留现状）。
- turn 5/6 等其他场景缺 openai.json 的独立 bug（如果存在）—— 这个 spec 不解决，等独立调查。
- 任何 RL 训练侧改动；这里只改输入到训练流水线之前的过滤。

## 改动文件清单

- `slime/agent/adapters/anthropic.py` — 加 1 个常量 + 1 个 helper + `_handle_request` 里 1 个 if-guard（约 +30 行）
- `tests/test_agent_adapters.py` — 加 3-4 个 unit test（约 +80 行）
