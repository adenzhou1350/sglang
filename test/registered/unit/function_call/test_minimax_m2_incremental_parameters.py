"""CPU-only tests for incremental MiniMax M2 parameter parsing."""

import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _make_tools():
    return [
        Tool(
            type="function",
            function=Function(
                name="write_file",
                description="Write text to a file",
                parameters={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string"},
                        "content": {"type": "string"},
                        "mode": {"type": "string"},
                    },
                },
            ),
        ),
        Tool(
            type="function",
            function=Function(
                name="search",
                description="Search for text",
                parameters={
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                },
            ),
        ),
    ]


def _collect(detector, chunks, tools):
    normal_text = ""
    calls = {}
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        normal_text += result.normal_text
        for call in result.calls:
            state = calls.setdefault(call.tool_index, {"name": None, "parameters": ""})
            if call.name is not None:
                state["name"] = call.name
            state["parameters"] += call.parameters
    return normal_text, [calls[index] for index in sorted(calls)]


class TestMinimaxM2IncrementalParameters(unittest.TestCase):
    def setUp(self):
        self.tools = _make_tools()

    def test_plain_text_fast_path_preserves_content_and_state(self):
        detector = MinimaxM2Detector()

        result = detector.parse_streaming_increment(
            "ordinary assistant text", self.tools
        )

        self.assertEqual(result.normal_text, "ordinary assistant text")
        self.assertEqual(result.calls, [])
        self.assertEqual(detector._buf, "")
        self.assertFalse(detector._in_tool_call)

    def test_long_value_keeps_only_delimiter_suffix_in_grammar_buffer(self):
        detector = MinimaxM2Detector()
        prefix = (
            '<minimax:tool_call><invoke name="write_file"><parameter name="content">'
        )
        chunks = [prefix]
        chunks.extend("x" * 32 for _ in range(512))
        chunks.append("</parameter></invoke></minimax:tool_call>")

        normal_text = ""
        calls = {}
        for index, chunk in enumerate(chunks):
            result = detector.parse_streaming_increment(chunk, self.tools)
            normal_text += result.normal_text
            for call in result.calls:
                state = calls.setdefault(
                    call.tool_index, {"name": None, "parameters": ""}
                )
                if call.name is not None:
                    state["name"] = call.name
                state["parameters"] += call.parameters
            if (
                0 < index < len(chunks) - 1
                and detector._pending_parameter_name is not None
            ):
                self.assertLess(
                    len(detector._buf),
                    max(
                        len(detector.tool_call_parameter_end_token),
                        len(detector.tool_call_function_end_token),
                    ),
                )

        self.assertEqual(normal_text, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertEqual(json.loads(calls[0]["parameters"]), {"content": "x" * 16384})

    def test_fragmented_parameter_markers_and_multiple_parameters(self):
        wire = (
            '<minimax:tool_call><invoke name="write_file">'
            '<parameter name="path">/tmp/out</parameter>'
            '<parameter name="content">hello 世界</parameter>'
            '<parameter name="mode">append</parameter>'
            "</invoke></minimax:tool_call>"
        )
        marker_end = wire.index('<parameter name="path">')
        chunks = [wire[:marker_end]] + list(wire[marker_end:])

        normal, calls = _collect(MinimaxM2Detector(), chunks, self.tools)

        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertEqual(
            json.loads(calls[0]["parameters"]),
            {"path": "/tmp/out", "content": "hello 世界", "mode": "append"},
        )

    def test_multiple_invokes_keep_parameter_state_request_local(self):
        chunks = [
            '<minimax:tool_call><invoke name="write_file">',
            '<parameter name="path">/tmp/a</parameter>',
            "</invoke>",
            '<invoke name="search"><parameter name="query">needle',
            " in haystack</parameter></invoke></minimax:tool_call>",
        ]

        normal, calls = _collect(MinimaxM2Detector(), chunks, self.tools)

        self.assertEqual(normal, "")
        self.assertEqual(len(calls), 2)
        self.assertEqual(calls[0]["name"], "write_file")
        self.assertEqual(json.loads(calls[0]["parameters"]), {"path": "/tmp/a"})
        self.assertEqual(calls[1]["name"], "search")
        self.assertEqual(
            json.loads(calls[1]["parameters"]), {"query": "needle in haystack"}
        )


if __name__ == "__main__":
    unittest.main()
