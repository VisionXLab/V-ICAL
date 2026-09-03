import asyncio
import base64
import hashlib
import json

import ai_backends.kimi as kimi_backend
from src.ai.ai_player import AIPlayer


class _Response:
    def __init__(self, status_code, data):
        self.status_code = status_code
        self._data = data
        self.text = str(data)

    def json(self):
        return self._data


def test_call_kimi_uploads_rewrites_and_deletes_video(monkeypatch):
    calls = []

    class FakeClient:
        def __init__(self, **_kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *_args):
            return None

        async def post(self, url, **kwargs):
            calls.append(("POST", url, kwargs))
            if url.endswith("/files"):
                assert kwargs["data"] == {"purpose": "video"}
                assert kwargs["files"]["file"][1] == b"fake-mp4"
                return _Response(200, {"id": "file-123"})
            payload = kwargs["json"]
            video_part = payload["messages"][0]["content"][0]
            assert video_part == {
                "type": "video_url",
                "video_url": {"url": "ms://file-123"},
            }
            assert "temperature" not in payload
            assert payload["reasoning_effort"] == "low"
            return _Response(200, {
                "choices": [{"message": {
                    "role": "assistant",
                    "content": "[Action: Left]",
                    "reasoning_content": "看见障碍",
                }}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
            })

        async def delete(self, url, **kwargs):
            calls.append(("DELETE", url, kwargs))
            return _Response(200, {"deleted": True})

    monkeypatch.setattr(kimi_backend.httpx, "AsyncClient", FakeClient)
    monkeypatch.setattr(kimi_backend, "_log_api_response", lambda *_args: None)
    video = base64.b64encode(b"fake-mp4").decode()
    text, usage = asyncio.run(kimi_backend.call_kimi(
        messages=[{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{video}"}},
            {"type": "text", "text": "play"},
        ]}],
        api_key="test-key",
        base_url="https://api.moonshot.cn/v1",
        model="kimi-k3",
        reasoning_effort="low",
    ))

    assert text == "[Action: Left]"
    assert usage["thinking"] == "看见障碍"
    assert usage["_assistant_message"]["reasoning_content"] == "看见障碍"
    assert usage["input_video_sha256"] == [hashlib.sha256(b"fake-mp4").hexdigest()]
    assert usage["temporary_files_uploaded"] == 1
    assert usage["temporary_files_deleted"] == 1
    assert usage["temporary_file_cleanup_errors"] == []
    assert calls[-1][0:2] == ("DELETE", "https://api.moonshot.cn/v1/files/file-123")


def test_kimi_history_preserves_complete_assistant_message():
    player = object.__new__(AIPlayer)
    player.api_mode = "kimi"
    player._last_assistant_message = {
        "role": "assistant",
        "content": "[Action: Left]",
        "reasoning_content": "reasoning",
    }
    message = player._assistant_history_message("[Action: Left]")
    assert message["reasoning_content"] == "reasoning"
    assert player._last_assistant_message is None


def test_kimi_rejects_unsupported_reasoning_effort():
    try:
        asyncio.run(kimi_backend.call_kimi(
            messages=[], api_key="test", base_url="https://api.moonshot.cn/v1",
            model="kimi-k3", reasoning_effort="medium",
        ))
    except ValueError as exc:
        assert "low, high, max" in str(exc)
    else:
        raise AssertionError("unsupported reasoning_effort should fail before HTTP")


def test_batch_loop_keeps_kimi_video_hash_in_round_usage():
    from tools.batch import batch_run_loop

    expected_hash = "a" * 64

    class FakeAI:
        frame_window = 0

        def _create_initial_prompt_section(self, section):
            return section

        def _build_initial_message_content(self, *_args):
            return ([{"type": "text", "text": "prompt"}], [], ["video"])

        async def _call_model(self, _messages):
            return "[Action: Left]", {
                "input": 123, "output": 4, "total": 127,
                "input_video_count": 1,
                "input_video_sha256": [expected_hash],
            }

        def _assistant_history_message(self, text):
            return {"role": "assistant", "content": text}

        def _parse_action(self, _text):
            return 0

    class FakeSession:
        action_info = {0: "Left"}
        frames = [b"frame"]
        is_game_over = False
        last_reward = 0
        last_info = {}

        def get_frame_base64(self, _index):
            return "frame"

        async def step(self, _action):
            self.is_game_over = True
            self.last_reward = -1
            self.last_info = {"terminated": True}
            return "next-frame"

    result = asyncio.run(batch_run_loop(FakeAI(), FakeSession()))
    assert result["status"] == "completed"
    assert result["token_usage"]["per_round"][0]["input_video_sha256"] == [expected_hash]


def test_video_probe_is_independent_and_writes_artifacts(monkeypatch, tmp_path):
    import sys
    import tools.kimi_video_probe as probe_tool

    video_bytes = b"real-batch-shaped-video"
    video_b64 = base64.b64encode(video_bytes).decode()
    video_hash = hashlib.sha256(video_bytes).hexdigest()
    config_dir = tmp_path / "source_config"
    config_dir.mkdir()
    (config_dir / "config.json").write_text("{}", encoding="utf-8")
    entry = {"game_id": "CliffWalking-v1", "config_name": "0", "config_dir": config_dir, "config_data": {}}

    async def fake_call_kimi(**_kwargs):
        content = _kwargs["messages"][0]["content"]
        assert len(content) == 2
        assert content[1] == {"type": "text", "text": probe_tool.BLIND_PROMPT}
        return "一个角色在悬崖旁的网格中移动，最后到达饼干目标。", {
            "input": 100,
            "output": 10,
            "total": 110,
            "input_video_count": 1,
            "input_video_sha256": [video_hash],
            "temporary_files_uploaded": 1,
            "temporary_files_deleted": 1,
            "temporary_file_cleanup_errors": [],
            "_assistant_message": {"role": "assistant", "content": "ok"},
        }

    monkeypatch.setenv("TEST_MOONSHOT_API_KEY", "secret-not-for-output")
    monkeypatch.setattr(probe_tool, "call_kimi", fake_call_kimi)
    monkeypatch.setattr(probe_tool, "load_probe_input", lambda: (entry, video_b64, video_hash, 10))
    monkeypatch.setattr(sys, "argv", [
        "vcl kimi-video-probe", "--api-key-env", "TEST_MOONSHOT_API_KEY",
        "--out", str(tmp_path / "reports"),
    ])

    assert asyncio.run(probe_tool.main()) == 0
    reports = list((tmp_path / "reports").glob("*/probe.md"))
    assert len(reports) == 1
    report = reports[0]
    assert (report.parent / "probe_video.mp4").read_bytes() == video_bytes
    assert "secret-not-for-output" not in report.read_text(encoding="utf-8")
    result = (report.parent / "probe.json").read_text(encoding="utf-8")
    assert "secret-not-for-output" not in result
    assert '"status": "PASS"' in result


def test_verify_gate_reads_standard_batch_artifacts_only(monkeypatch, tmp_path):
    import sys
    from tools._paths import ROOT
    import tools.kimi_verify_gate as verifier

    config_hash = hashlib.sha256(
        (ROOT / "ai_configs" / "CliffWalking-v1" / "0" / "config.json").read_bytes()
    ).hexdigest()
    video_hash = "b" * 64
    probe_dir = tmp_path / "probe"
    batch_run = tmp_path / "batch" / "CliffWalking-v1" / "0" / "run_0"
    probe_dir.mkdir()
    batch_run.mkdir(parents=True)
    probe = {
        "status": "PASS", "model": "kimi-k3", "base_url": "https://api.moonshot.cn/v1",
        "video_sha256": video_hash, "source_config_sha256": config_hash,
    }
    result = {
        "game_id": "CliffWalking-v1", "config_name": "0", "run_index": 0,
        "status": "completed", "total_steps": 8, "total_reward": -8, "total_rounds": 8,
        "model": "kimi-k3", "error": None,
        "api_config": {"api_mode": "kimi", "base_url": "https://api.moonshot.cn/v1"},
        "token_usage": {"per_round": [{"round": 1, "input": 1200, "output": 20,
                                          "total": 1220, "input_video_count": 1,
                                          "input_video_sha256": [video_hash]}]},
    }
    (probe_dir / "probe.json").write_text(json.dumps(probe), encoding="utf-8")
    (batch_run / "result.json").write_text(json.dumps(result), encoding="utf-8")
    (batch_run / "conversation.json").write_text(
        json.dumps({"rounds": [{"is_valid": True}]}), encoding="utf-8"
    )
    monkeypatch.setattr(sys, "argv", [
        "vcl kimi-verify-gate", "--probe", str(probe_dir / "probe.json"),
        "--batch", str(tmp_path / "batch"), "--out", str(tmp_path / "verified"),
    ])
    assert verifier.main() == 0
    reports = list((tmp_path / "verified").glob("*/verification.md"))
    assert len(reports) == 1
    assert "状态：**PASS**" in reports[0].read_text(encoding="utf-8")
