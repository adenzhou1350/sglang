import json
import logging
import re
from typing import Any, Dict, List, Tuple

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)
from sglang.srt.function_call.utils import get_schema_properties

logger = logging.getLogger(__name__)


class MinimaxM2Detector(BaseFormatDetector):
    """
    Detector for MiniMax M2 models.
    Assumes function call format:
        <minimax:tool_call>
        <invoke name="func1">
        <parameter name="param1">value1</parameter>
        <parameter name="param2">value2</parameter>
        </invoke>
        </minimax:tool_call>
    """

    _INCREMENTAL_PARAMETER_THRESHOLD = 256
    _STREAMING_PARAMETER_REGEX = re.compile(
        r"<parameter name=\"([^>]+)\">(.*?)</parameter>", re.DOTALL
    )
    tool_call_parameter_prefix = '<parameter name="'
    tool_call_parameter_end_token = "</parameter>"
    _pending_parameter_name: str | None
    _pending_parameter_value_parts: List[str]

    def __init__(self):
        super().__init__()
        self.tool_call_start_token: str = "<minimax:tool_call>"
        self.tool_call_end_token: str = "</minimax:tool_call>"
        self.tool_call_prefix: str = '<invoke name="'
        self.tool_call_function_end_token: str = "</invoke>"
        self.tool_call_regex = re.compile(
            r"<minimax:tool_call>(.*?)</minimax:tool_call>|<minimax:tool_call>(.*?)$",
            re.DOTALL,
        )
        self.tool_call_function_regex = re.compile(
            r"<invoke name=\"(.*?)</invoke>|<invoke name=\"(.*)$", re.DOTALL
        )
        self.tool_call_parameter_regex = re.compile(
            r"<parameter name=\"(.*?)</parameter>|<parameter name=\"(.*?)$", re.DOTALL
        )
        self._buf: str = ""

        # Streaming state variables
        self._current_function_name: str = ""
        self._current_parameters: Dict[str, Any] = {}
        self._streamed_parameters: Dict[
            str, str
        ] = {}  # Track what parameter content we've streamed
        self._in_tool_call: bool = False
        self._function_name_sent: bool = False

    def has_tool_call(self, text: str) -> bool:
        return self.tool_call_start_token in text

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        normal, calls = self._extract(text, tools)
        return StreamingParseResult(normal_text=normal, calls=calls)

    def _convert_param_value(self, value: str, param_type: str) -> Any:
        """Convert parameter value to the correct type (legacy single-type version)."""
        return self._convert_param_value_with_types(value, [param_type])

    def _extract_types_from_schema(self, schema: Any) -> list[str]:
        """
        Extract all possible types from a JSON schema definition.
        Handles anyOf, oneOf, allOf, type arrays, and enum fields.

        Args:
            schema: The JSON schema definition for a parameter

        Returns:
            List of type strings (e.g., ["string", "integer", "null"])
        """
        if schema is None:
            return ["string"]

        if not isinstance(schema, dict):
            return ["string"]

        types: set[str] = set()

        # Handle direct "type" field
        if "type" in schema:
            type_value = schema["type"]
            if isinstance(type_value, str):
                types.add(type_value)
            elif isinstance(type_value, list):
                for t in type_value:
                    if isinstance(t, str):
                        types.add(t)

        # Handle enum - infer types from enum values
        if "enum" in schema and isinstance(schema["enum"], list) and schema["enum"]:
            for value in schema["enum"]:
                if value is None:
                    types.add("null")
                elif isinstance(value, bool):
                    types.add("boolean")
                elif isinstance(value, int):
                    types.add("integer")
                elif isinstance(value, float):
                    types.add("number")
                elif isinstance(value, str):
                    types.add("string")
                elif isinstance(value, list):
                    types.add("array")
                elif isinstance(value, dict):
                    types.add("object")

        # Handle anyOf, oneOf, allOf - recursively extract types
        for choice_field in ("anyOf", "oneOf", "allOf"):
            if choice_field in schema and isinstance(schema[choice_field], list):
                for choice in schema[choice_field]:
                    extracted = self._extract_types_from_schema(choice)
                    types.update(extracted)

        # If no types found, default to string
        if not types:
            return ["string"]

        return list(types)

    def _convert_param_value_with_types(
        self, value: str, param_types: list[str]
    ) -> Any:
        """
        Convert parameter value to the correct type based on a list of possible types.
        Tries each type in order until one succeeds.

        Args:
            value: The string value to convert
            param_types: List of possible type strings

        Returns:
            The converted value
        """
        if value.lower() == "null":
            return None

        # Normalize types
        normalized_types = [t.lower() for t in param_types]

        # Try null first if it's in the list
        if "null" in normalized_types or value.lower() in ("null", "none", "nil"):
            return None

        # Try each type in order of preference (most specific first, string as fallback)
        # Priority: integer > number > boolean > object > array > string
        type_priority = [
            "integer",
            "int",
            "number",
            "float",
            "boolean",
            "bool",
            "object",
            "array",
            "string",
            "str",
            "text",
        ]

        for param_type in type_priority:
            if param_type not in normalized_types:
                continue

            if param_type in ["string", "str", "text"]:
                return value
            elif param_type in ["integer", "int"]:
                try:
                    return int(value)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["number", "float"]:
                try:
                    val = float(value)
                    return val if val != int(val) else int(val)
                except (ValueError, TypeError):
                    continue
            elif param_type in ["boolean", "bool"]:
                lower_val = value.lower().strip()
                if lower_val in ["true", "1", "yes", "on"]:
                    return True
                elif lower_val in ["false", "0", "no", "off"]:
                    return False
                continue
            elif param_type in ["object", "array"]:
                try:
                    return json.loads(value)
                except json.JSONDecodeError:
                    continue

        # Fallback: try JSON parse, then return as string
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value

    def _get_param_types_from_config(
        self, param_name: str, param_config: dict
    ) -> list[str]:
        """
        Get parameter types from parameter configuration.
        Handles anyOf, oneOf, allOf, and direct type definitions.

        Args:
            param_name: The name of the parameter
            param_config: The properties dict from the tool schema

        Returns:
            List of type strings
        """
        if param_name not in param_config:
            return ["string"]

        param_schema = param_config[param_name]
        if not isinstance(param_schema, dict):
            return ["string"]

        return self._extract_types_from_schema(param_schema)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        # Most assistant chunks contain no XML at all.  When there is no
        # retained prefix, a chunk without '<' cannot begin any MiniMax marker.
        if not self._in_tool_call and not self._buf and "<" not in new_text:
            return StreamingParseResult(normal_text=new_text)

        self._buf += new_text
        normal = ""
        calls: List[ToolCallItem] = []

        # Build tool indices for validation
        if not hasattr(self, "_tool_indices"):
            self._tool_indices = self._get_tool_indices(tools)

        while True:
            # If we're not in a tool call and don't see a start token, return normal text
            if not self._in_tool_call and self.tool_call_start_token not in self._buf:
                normal += self._buf
                self._buf = ""
                break

            # Look for tool call start
            if not self._in_tool_call:
                s = self._buf.find(self.tool_call_start_token)
                if s == -1:
                    normal += self._buf
                    self._buf = ""
                    break

                normal += self._buf[:s]
                self._buf = self._buf[s:]

                self._in_tool_call = True
                self._function_name_sent = False
                self._current_function_name = ""
                self._current_parameters = {}
                self._streamed_parameters = {}
                self._reset_parameter_state()

                # Remove the start token
                self._buf = self._buf[len(self.tool_call_start_token) :]
                continue

            # We're in a tool call, try to parse function name if not sent yet
            if not self._function_name_sent:
                # Look for function name pattern: <invoke name=name>
                function_match = re.search(r"<invoke name=\"([^>]+)\">", self._buf)
                if function_match:
                    function_name = function_match.group(1).strip()

                    # Validate function name
                    if function_name in self._tool_indices:
                        self._current_function_name = function_name
                        self._function_name_sent = True

                        # Initialize tool call tracking
                        if self.current_tool_id == -1:
                            self.current_tool_id = 0

                        # Ensure tracking arrays are large enough
                        while len(self.prev_tool_call_arr) <= self.current_tool_id:
                            self.prev_tool_call_arr.append({})
                        while len(self.streamed_args_for_tool) <= self.current_tool_id:
                            self.streamed_args_for_tool.append("")

                        # Store tool call info
                        self.prev_tool_call_arr[self.current_tool_id] = {
                            "name": function_name,
                            "arguments": {},
                        }

                        # Send tool name with empty parameters
                        calls.append(
                            ToolCallItem(
                                tool_index=self.current_tool_id,
                                name=function_name,
                                parameters="",
                            )
                        )

                        # Remove the processed function declaration
                        self._buf = self._buf[function_match.end() :]
                        continue
                    else:
                        # Invalid function name, reset state
                        logger.warning(f"Invalid function name: {function_name}")
                        self._reset_streaming_state()
                        normal += self._buf
                        self._buf = ""
                        break
                else:
                    # Function name not complete yet, wait for more text
                    break

            # Parse parameters incrementally
            if self._function_name_sent:
                # Process parameters and get any calls to emit
                parameter_calls = self._parse_and_stream_parameters(self._buf, tools)
                calls.extend(parameter_calls)

                # Check if tool call is complete
                if self.tool_call_function_end_token in self._buf:
                    end_pos = self._buf.find(self.tool_call_function_end_token)

                    # Add closing brace to complete the JSON object
                    current_streamed = self.streamed_args_for_tool[self.current_tool_id]
                    if current_streamed:
                        # Count opening and closing braces to check if JSON is complete
                        open_braces = current_streamed.count("{")
                        close_braces = current_streamed.count("}")
                        if open_braces > close_braces:
                            calls.append(
                                ToolCallItem(
                                    tool_index=self.current_tool_id,
                                    name=None,
                                    parameters="}",
                                )
                            )
                            self.streamed_args_for_tool[self.current_tool_id] = (
                                current_streamed + "}"
                            )

                    # Complete the tool call
                    self._buf = self._buf[
                        end_pos + len(self.tool_call_function_end_token) :
                    ]
                    self._reset_streaming_state(True)
                    self.current_tool_id += 1
                    continue
                else:
                    # Tool call not complete yet, wait for more text
                    break

        return StreamingParseResult(normal_text=normal, calls=calls)

    def _parse_and_stream_parameters(
        self, text_to_parse: str, tools: List[Tool]
    ) -> List[ToolCallItem]:
        """
        Incrementally parse parameter blocks and return tool call items to emit.

        Accepted input is removed from ``self._buf`` exactly once.  While a
        parameter value is incomplete, only suffixes that could finish a
        delimiter remain in the grammar buffer; the value itself is collected
        in parts and converted once when ``</parameter>`` arrives.

        Args:
            text_to_parse: The text to search for parameter blocks

        Returns:
            List of ToolCallItem objects to emit (may be empty)
        """
        calls: List[ToolCallItem] = []
        new_params: Dict[str, Any] = {}

        # CPython's regex engine is cheaper for short buffers.  Preserve that
        # path exactly until enough accepted history exists for repeated scans
        # and immutable-string appends to dominate.
        if (
            self._pending_parameter_name is None
            and len(text_to_parse) < self._INCREMENTAL_PARAMETER_THRESHOLD
        ):
            for match in self._STREAMING_PARAMETER_REGEX.finditer(text_to_parse):
                param_name = match.group(1).strip()
                param_value = match.group(2)
                new_params[param_name] = self._parse_parameter(
                    self._current_function_name, param_name, param_value, tools
                )
            if not new_params:
                return calls
            return self._stream_parameter_updates(new_params)

        # ``text_to_parse`` snapshots ``self._buf`` at the call boundary.  Use
        # the instance buffer below because it is compacted as transitions
        # accept input.

        while True:
            if self._pending_parameter_name is None:
                param_start = self._buf.find(self.tool_call_parameter_prefix)
                function_end = self._buf.find(self.tool_call_function_end_token)

                # Preserve the function terminator for the owning transition in
                # parse_streaming_increment.
                if function_end != -1 and (
                    param_start == -1 or function_end < param_start
                ):
                    break

                if param_start == -1:
                    held = max(
                        self._ends_with_partial_token(
                            self._buf, self.tool_call_parameter_prefix
                        ),
                        self._ends_with_partial_token(
                            self._buf, self.tool_call_function_end_token
                        ),
                    )
                    self._buf = self._buf[-held:] if held else ""
                    break

                name_start = param_start + len(self.tool_call_parameter_prefix)
                name_end = self._buf.find('">', name_start)
                if name_end == -1:
                    # Parameter names are schema-bounded.  Retain the incomplete
                    # header, but discard any already ignored prefix.
                    self._buf = self._buf[param_start:]
                    break

                self._pending_parameter_name = self._buf[name_start:name_end].strip()
                self._pending_parameter_value_parts = []
                self._buf = self._buf[name_end + 2 :]

            param_end = self._buf.find(self.tool_call_parameter_end_token)
            function_end = self._buf.find(self.tool_call_function_end_token)

            # Match the old malformed-input behavior: an invoke terminator seen
            # before a parameter terminator belongs to the outer transition.
            if function_end != -1 and (param_end == -1 or function_end < param_end):
                break

            if param_end == -1:
                held = max(
                    self._ends_with_partial_token(
                        self._buf, self.tool_call_parameter_end_token
                    ),
                    self._ends_with_partial_token(
                        self._buf, self.tool_call_function_end_token
                    ),
                )
                if held:
                    accepted = self._buf[:-held]
                    self._buf = self._buf[-held:]
                else:
                    accepted = self._buf
                    self._buf = ""
                if accepted:
                    self._pending_parameter_value_parts.append(accepted)
                break

            self._pending_parameter_value_parts.append(self._buf[:param_end])
            param_name = self._pending_parameter_name
            assert param_name is not None
            param_value = "".join(self._pending_parameter_value_parts)
            self._buf = self._buf[param_end + len(self.tool_call_parameter_end_token) :]
            self._reset_parameter_state()

            new_params[param_name] = self._parse_parameter(
                self._current_function_name, param_name, param_value, tools
            )

        return self._stream_parameter_updates(new_params)

    def _stream_parameter_updates(
        self, new_params: Dict[str, Any]
    ) -> List[ToolCallItem]:
        """Serialize newly completed parameters using the legacy wire format."""
        calls: List[ToolCallItem] = []
        if not new_params:
            return calls

        previous_args_json = self.streamed_args_for_tool[self.current_tool_id]

        # Build incremental JSON properly
        if not self._current_parameters:
            # First parameter(s) - start JSON object but don't close it yet
            items = []
            for key, value in new_params.items():
                items.append(
                    f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                )
            json_fragment = "{" + ", ".join(items)

            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=None,
                    parameters=json_fragment,
                )
            )
            self.streamed_args_for_tool[self.current_tool_id] = json_fragment

        else:
            # Additional parameters - add them incrementally
            new_keys = set(new_params.keys()) - set(self._current_parameters.keys())
            if new_keys:
                # Build the continuation part (no closing brace yet)
                continuation_parts = []
                for key in new_keys:
                    value = new_params[key]
                    continuation_parts.append(
                        f"{json.dumps(key, ensure_ascii=False)}: {json.dumps(value, ensure_ascii=False)}"
                    )

                json_fragment = ", " + ", ".join(continuation_parts)

                calls.append(
                    ToolCallItem(
                        tool_index=self.current_tool_id,
                        name=None,
                        parameters=json_fragment,
                    )
                )
                self.streamed_args_for_tool[self.current_tool_id] = (
                    previous_args_json + json_fragment
                )

        # Update current state.  Only newly completed parameters are in
        # new_params after the incremental transition; the guarded legacy path
        # may include prior keys, for which update is idempotent.
        self._current_parameters.update(new_params)
        self.prev_tool_call_arr[self.current_tool_id]["arguments"] = dict(
            self._current_parameters
        )

        return calls

    def _reset_parameter_state(self) -> None:
        self._pending_parameter_name = None
        self._pending_parameter_value_parts = []

    def _reset_streaming_state(self, still_in_tool_call: bool = False):
        """Reset streaming state for the next tool call"""
        self._in_tool_call = still_in_tool_call
        self._function_name_sent = False
        self._current_function_name = ""
        self._current_parameters = {}
        self._streamed_parameters = {}
        self.current_tool_name_sent = False
        self._reset_parameter_state()

    def _extract(self, text: str, tools: List[Tool]) -> Tuple[str, List[ToolCallItem]]:
        normal_parts: List[str] = []
        calls: List[ToolCallItem] = []
        cursor = 0
        while True:
            s = text.find(self.tool_call_start_token, cursor)
            if s == -1:
                normal_parts.append(text[cursor:])
                break
            normal_parts.append(text[cursor:s])
            e = text.find(self.tool_call_end_token, s)
            if e == -1:
                normal_parts.append(text[s:])
                break
            block = text[s : e + len(self.tool_call_end_token)]
            cursor = e + len(self.tool_call_end_token)
            calls.extend(self._parse_block(block, tools))
        return "".join(normal_parts), calls

    def _parse_block(self, block: str, tools: List[Tool]) -> List[ToolCallItem]:
        res: List[ToolCallItem] = []
        for m in self.tool_call_function_regex.findall(block):
            txt = m[0] if m[0] else m[1]
            if '">' not in txt:
                continue
            idx = txt.index('">')
            fname = txt[:idx].strip()
            body = txt[idx + 2 :]
            params: Dict[str, Any] = {}
            for pm in self.tool_call_parameter_regex.findall(body):
                ptxt = pm[0] if pm[0] else pm[1]
                if '">' not in ptxt:
                    continue
                pidx = ptxt.index('">')
                pname = ptxt[:pidx].strip()
                pval = ptxt[pidx + 2 :].lstrip("\n").rstrip("\n")
                params[pname] = self._parse_parameter(fname, pname, pval, tools)
            raw = {"name": fname, "arguments": params}
            try:
                # TODO: fix idx in function call, the index for a function
                # call will always be -1 in parse_base_json
                res.extend(self.parse_base_json(raw, tools))
            except Exception:
                logger.warning("invalid tool call for %s dropped", fname)
        return res

    def _parse_parameter(
        self, fname: str, pname: str, pval: str, tools: List[Tool]
    ) -> Any:
        param_config = {}
        for tool in tools:
            if tool.function.name == fname and tool.function.parameters is not None:
                parameters = tool.function.parameters
                if isinstance(parameters, dict):
                    param_config = get_schema_properties(parameters)
                    break

        param_type = self._get_param_types_from_config(pname, param_config)
        return self._convert_param_value_with_types(pval, param_type)

    def supports_structural_tag(self) -> bool:
        return False

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError
