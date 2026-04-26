"""Regression test for mixed-role one-step rollback in the session path."""

from __future__ import annotations

import requests
from tests.fast.router.session_pretokenized_test_utils import (
    ScriptedBackendTurn,
    ScriptedChatBackend,
    compute_local_session_mismatch,
    fetch_session_payload,
    forbidden_mismatches,
    load_test_tokenizer,
    make_router_env,
    teardown_router_env,
)

from miles.utils.test_utils.mock_trajectories import WEATHER_TOOLS


def test_mixed_role_one_step_rollback_preserves_token_correctness():
    hf_checkpoint = "zai-org/GLM-4.7-Flash"
    tito_model = "glm47"
    allowed_append_roles = ["tool", "user", "system"]
    tokenizer = load_test_tokenizer(hf_checkpoint, None)

    assistant_a = {
        "role": "assistant",
        "content": "",
        "reasoning_content": " ",
        "tool_calls": [
            {
                "id": "call_weather_1",
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "arguments": {"city": "Beijing"},
                },
            }
        ],
    }
    assistant_b = {
        "role": "assistant",
        "content": "Brief answer: tool branch B completed.",
        "reasoning_content": " ",
    }
    assistant_c = {
        "role": "assistant",
        "content": '{"summary":"tool/user/system rollback branch C completed"}',
        "reasoning_content": " ",
    }

    scripted_turns = [
        ScriptedBackendTurn(
            response_message=assistant_a,
            render_message={**assistant_a, "content": None},
        ),
        ScriptedBackendTurn(
            response_message=assistant_b,
            render_message=assistant_b,
        ),
        ScriptedBackendTurn(
            response_message=assistant_c,
            render_message=assistant_c,
        ),
    ]
    backend = ScriptedChatBackend(tokenizer, scripted_turns)
    backend.start()
    env = make_router_env(
        backend,
        hf_checkpoint=hf_checkpoint,
        chat_template_path=None,
        tito_model=tito_model,
        allowed_append_roles=allowed_append_roles,
    )

    system_msg = {"role": "system", "content": "You are a careful assistant."}
    user_msg = {"role": "user", "content": "Check Beijing weather with the tool."}
    branch_b_tool = {
        "role": "tool",
        "content": '{"temperature": 25, "condition": "sunny"}',
        "tool_call_id": "call_weather_1",
        "name": "get_weather",
    }
    branch_b_user = {"role": "user", "content": "Now summarize it in one sentence."}
    branch_b_system = {"role": "system", "content": "Use a brief style."}
    branch_c_tool = {
        "role": "tool",
        "content": '{"temperature": 12, "condition": "cloudy"}',
        "tool_call_id": "call_weather_1",
        "name": "get_weather",
    }
    branch_c_user = {"role": "user", "content": "Actually, include a structured recap."}
    branch_c_system = {"role": "system", "content": "Return valid JSON."}

    turn1_messages = [system_msg, user_msg]
    turn2_messages = turn1_messages + [assistant_a, branch_b_tool, branch_b_user, branch_b_system]
    turn3_messages = turn1_messages + [assistant_a, branch_c_tool, branch_c_user, branch_c_system]

    try:
        backend.reset_stats()
        session_id = requests.post(f"{env.url}/sessions", timeout=5.0).json()["session_id"]

        def _run_turn(turn_index: int, messages: list[dict], expected_assistant: dict):
            response = requests.post(
                f"{env.url}/sessions/{session_id}/v1/chat/completions",
                json={"messages": messages, "tools": WEATHER_TOOLS},
                timeout=10.0,
            )
            assert response.status_code == 200, f"turn {turn_index} failed: {response.text}"
            body = response.json()
            assert len(body["choices"]) == 1
            assert body["choices"][0]["message"] == expected_assistant
            assert "prompt_token_ids" in body["choices"][0]
            if turn_index > 0:
                assert "input_ids" in backend.request_log[turn_index], f"turn {turn_index} missing input_ids"

            session_payload = fetch_session_payload(env.url, session_id)
            metadata = session_payload["metadata"]
            session_messages = list(messages) + [expected_assistant]
            remote_mismatch = metadata.get("tito_session_mismatch", [])
            local_mismatch = compute_local_session_mismatch(
                tokenizer,
                tito_model=tito_model,
                allowed_append_roles=allowed_append_roles,
                messages=session_messages,
                accumulated_token_ids=metadata["accumulated_token_ids"],
                tools=WEATHER_TOOLS,
            )
            assert remote_mismatch == local_mismatch
            assert (
                forbidden_mismatches(remote_mismatch) == []
            ), f"turn {turn_index} has forbidden mismatch types: {remote_mismatch}"
            return session_payload

        _run_turn(0, turn1_messages, assistant_a)
        _run_turn(1, turn2_messages, assistant_b)
        final_session_payload = _run_turn(2, turn3_messages, assistant_c)

        records = final_session_payload["records"]
        assert len(records) == 2, f"rollback should replace checkpoint B, got {len(records)} records"
        assert records[0]["request"]["messages"] == turn1_messages
        assert records[1]["request"]["messages"] == turn3_messages
    finally:
        teardown_router_env(env)
