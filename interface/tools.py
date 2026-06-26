"""
NFN Tool Calling — Structured Agentic Interface

Enables NFN to call external tools (functions) and integrate their results
back into the generation stream — the key primitive for AI agents.

How it works:
  1. Tool definitions are serialised into the system prompt as JSON schema
  2. The model generates text; if a <tool_call> block is detected, generation
     pauses and the tool is dispatched
  3. Tool result is injected as a <tool_result> token into the context
  4. Generation resumes with the result in context

This is model-agnostic: no fine-tuning needed for basic tool use.
The model learns tool calling the same way it learns any structured text.

Format:
  <tool_call>
  {"name": "search", "arguments": {"query": "...", "n": 3}}
  </tool_call>

  <tool_result name="search">
  [{"title": "...", "url": "...", "snippet": "..."}]
  </tool_result>

For fine-tuned models: use ToolAwareTrainer which injects synthetic tool-call
examples into the training mix (instruction following on tool calls).

Classes:
  ToolSpec         : schema for a callable tool
  ToolRegistry     : collection of tools available to the model
  ToolCallParser   : detects and parses <tool_call> blocks from generated text
  ToolCallingModel : wraps AGINFNModel.generate() with tool dispatch loop
"""

import json
import re
import traceback
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Tool specification
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ToolSpec:
    """
    Defines a callable tool.

    name:        tool identifier (used in <tool_call> JSON)
    description: what the tool does (goes in the system prompt)
    parameters:  JSON-schema style parameter spec
    fn:          the actual callable — (arguments: dict) → str
    """
    name:        str
    description: str
    parameters:  Dict[str, Any]
    fn:          Callable[[Dict], str]

    def to_schema(self) -> str:
        """Returns a compact JSON schema string for injection into prompts."""
        return json.dumps({
            "name":        self.name,
            "description": self.description,
            "parameters":  self.parameters,
        }, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# Tool registry
# ─────────────────────────────────────────────────────────────────────────────

class ToolRegistry:
    """
    Maintains available tools and generates the system prompt block.

    Usage:
        registry = ToolRegistry()
        registry.register(ToolSpec("calc", "Evaluate math", {...}, eval_fn))
        prompt = registry.system_block()
        result = registry.dispatch("calc", {"expr": "2+2"})
    """

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, tool: ToolSpec) -> "ToolRegistry":
        self._tools[tool.name] = tool
        return self

    def register_fn(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
    ):
        """Decorator form: @registry.register_fn("name", "desc", {...})"""
        def decorator(fn: Callable) -> Callable:
            self.register(ToolSpec(name, description, parameters, fn))
            return fn
        return decorator

    def system_block(self) -> str:
        """Returns the tools section of the system prompt."""
        if not self._tools:
            return ""
        schemas = "\n".join(t.to_schema() for t in self._tools.values())
        return (
            "You have access to the following tools. "
            "To call a tool, output a <tool_call> block with JSON:\n\n"
            "<tools>\n" + schemas + "\n</tools>\n\n"
            "Tool call format:\n"
            "<tool_call>\n{\"name\": \"tool_name\", \"arguments\": {...}}\n</tool_call>\n"
        )

    def dispatch(self, name: str, arguments: Dict) -> str:
        """Execute a tool call, return result as string."""
        if name not in self._tools:
            return f"[Error: unknown tool '{name}'. Available: {list(self._tools.keys())}]"
        try:
            result = self._tools[name].fn(arguments)
            return str(result)
        except Exception as e:
            return f"[Error in tool '{name}': {e}]"

    def __len__(self) -> int:
        return len(self._tools)


# ─────────────────────────────────────────────────────────────────────────────
# Tool call parser
# ─────────────────────────────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(.*?)\s*</tool_call>",
    re.DOTALL,
)

_TOOL_RESULT_TEMPLATE = "\n<tool_result name=\"{name}\">\n{result}\n</tool_result>\n"


class ToolCallParser:
    """
    Detects <tool_call> blocks in text and extracts (name, arguments).

    Stateless — can be applied to any generated text segment.
    """

    @staticmethod
    def find_calls(text: str) -> List[Tuple[str, Dict, int, int]]:
        """
        Returns list of (name, arguments, start_pos, end_pos) for each tool call.
        Only returns calls where the closing </tool_call> tag is present.
        """
        results = []
        for m in _TOOL_CALL_RE.finditer(text):
            raw = m.group(1).strip()
            try:
                parsed = json.loads(raw)
                name   = parsed.get("name", "")
                args   = parsed.get("arguments", {})
                if name:
                    results.append((name, args, m.start(), m.end()))
            except json.JSONDecodeError:
                pass
        return results

    @staticmethod
    def has_open_call(text: str) -> bool:
        """True if text has an unclosed <tool_call> (generation should continue)."""
        opens  = text.count("<tool_call>")
        closes = text.count("</tool_call>")
        return opens > closes

    @staticmethod
    def inject_result(text: str, name: str, result: str, call_end: int) -> str:
        """Insert tool result after the closing </tool_call> tag."""
        result_block = _TOOL_RESULT_TEMPLATE.format(name=name, result=result)
        return text[:call_end] + result_block + text[call_end:]


# ─────────────────────────────────────────────────────────────────────────────
# Tool calling model wrapper
# ─────────────────────────────────────────────────────────────────────────────

class ToolCallingModel:
    """
    Wraps AGINFNModel with an agentic tool-calling loop.

    Each generate() call may trigger zero or more tool calls:
      1. Generate tokens until stop condition or <tool_call> is complete
      2. Parse the tool call, dispatch to registry
      3. Inject <tool_result> into context
      4. Continue generating
      5. Repeat until max_tool_calls or EOS

    Usage:
        tool_model = ToolCallingModel(model, tokenizer, registry)
        output = tool_model.generate(
            prompt="What is 2^10?",
            max_new_tokens=200,
        )
    """

    def __init__(
        self,
        model,                        # AGINFNModel
        tokenizer: "NFNTokenizer",
        registry:  ToolRegistry,
        max_tool_calls: int = 5,      # safety limit per generation
    ):
        self.model          = model
        self.tokenizer      = tokenizer
        self.registry       = registry
        self.max_tool_calls = max_tool_calls

    def _encode(self, text: str, device: torch.device) -> torch.Tensor:
        ids = self.tokenizer.encode(text, add_bos=True)
        return torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    def _decode(self, ids: torch.Tensor) -> str:
        flat = ids[0].tolist()
        # Drop BOS
        bos = getattr(self.tokenizer, "bos_id", 1)
        if flat and flat[0] == bos:
            flat = flat[1:]
        return self.tokenizer.decode(flat)

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_new_tokens: int = 512,
        temperature: float = 0.8,
        top_p: float = 0.95,
        stop_sequences: Optional[List[str]] = None,
    ) -> str:
        """
        Generate a response with automatic tool dispatch.

        Returns the full generated text including tool results.
        """
        device = next(self.model.parameters()).device

        # Build full prompt
        parts = []
        if system_prompt:
            parts.append(system_prompt)
        if self.registry:
            parts.append(self.registry.system_block())
        parts.append(f"User: {prompt}\nAssistant:")
        full_prompt = "\n".join(p for p in parts if p)

        input_ids = self._encode(full_prompt, device)
        generated_text = ""
        tool_calls_made = 0

        while True:
            remaining = max_new_tokens - len(generated_text.split())
            if remaining <= 0:
                break

            # Generate chunk until potential tool call or EOS
            out_ids = self.model.generate(
                input_ids,
                max_new_tokens=min(remaining, 256),
                temperature=temperature,
                top_p=top_p,
            )

            new_ids    = out_ids[:, input_ids.shape[1]:]
            new_text   = self._decode(new_ids)
            generated_text += new_text

            # Check for completed tool calls
            calls = ToolCallParser.find_calls(generated_text)
            if not calls or tool_calls_made >= self.max_tool_calls:
                # No tool calls — check for open (incomplete) call
                if ToolCallParser.has_open_call(generated_text):
                    # Keep generating to complete the call
                    input_ids = out_ids
                    continue
                break  # Done

            # Process the first unhandled tool call
            name, args, start, end = calls[-1]   # last = most recent
            result = self.registry.dispatch(name, args)

            # Inject result into text
            generated_text = ToolCallParser.inject_result(
                generated_text, name, result, end
            )
            tool_calls_made += 1

            # Re-encode full context for next generation pass
            full_with_results = full_prompt + generated_text
            input_ids = self._encode(full_with_results, device)
            # Trim to max context
            max_ctx = self.model.cfg.max_seq_len
            if input_ids.shape[1] > max_ctx:
                input_ids = input_ids[:, -max_ctx:]

            # Check stop sequences
            if stop_sequences:
                for stop in stop_sequences:
                    if stop in generated_text:
                        idx = generated_text.find(stop)
                        generated_text = generated_text[:idx]
                        return generated_text.strip()

        return generated_text.strip()


# ─────────────────────────────────────────────────────────────────────────────
# Built-in tools
# ─────────────────────────────────────────────────────────────────────────────

def make_default_registry() -> ToolRegistry:
    """
    Returns a ToolRegistry with safe built-in tools.
    Only includes tools that are safe to expose without user confirmation.
    """
    registry = ToolRegistry()

    @registry.register_fn(
        "calculator",
        "Evaluate a mathematical expression safely.",
        {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "Math expression to evaluate, e.g. '2**10 + sqrt(3)'"}
            },
            "required": ["expression"],
        },
    )
    def _calc(args: Dict) -> str:
        import math as _math
        expr = args.get("expression", "").strip()
        # Whitelist: only safe math operations
        allowed = set("0123456789.+-*/()** ,eEijpPsqrtcosilgfabnmx")
        safe_names = {k: getattr(_math, k) for k in dir(_math) if not k.startswith("_")}
        safe_names["abs"] = abs
        safe_names["round"] = round
        try:
            result = eval(expr, {"__builtins__": {}}, safe_names)  # noqa: S307
            return str(result)
        except Exception as e:
            return f"Error: {e}"

    @registry.register_fn(
        "json_extract",
        "Extract a field from a JSON string.",
        {
            "type": "object",
            "properties": {
                "json_string": {"type": "string"},
                "path":        {"type": "string", "description": "Dot-separated path, e.g. 'data.items.0.name'"},
            },
            "required": ["json_string", "path"],
        },
    )
    def _json_extract(args: Dict) -> str:
        try:
            obj  = json.loads(args["json_string"])
            path = args["path"].split(".")
            for key in path:
                if isinstance(obj, list):
                    obj = obj[int(key)]
                else:
                    obj = obj[key]
            return json.dumps(obj)
        except Exception as e:
            return f"Error: {e}"

    @registry.register_fn(
        "string_ops",
        "Perform string operations: upper, lower, split, join, replace, count, slice.",
        {
            "type": "object",
            "properties": {
                "operation": {"type": "string", "enum": ["upper", "lower", "split", "join", "replace", "count", "slice", "len"]},
                "text":      {"type": "string"},
                "args":      {"type": "array", "description": "Additional arguments for the operation"},
            },
            "required": ["operation", "text"],
        },
    )
    def _string_ops(args: Dict) -> str:
        op   = args.get("operation", "")
        text = args.get("text", "")
        extra = args.get("args", [])
        ops = {
            "upper":   lambda: text.upper(),
            "lower":   lambda: text.lower(),
            "len":     lambda: str(len(text)),
            "split":   lambda: json.dumps(text.split(extra[0] if extra else None)),
            "join":    lambda: (extra[0] if extra else "").join(text),
            "replace": lambda: text.replace(extra[0], extra[1]) if len(extra) >= 2 else "need 2 args",
            "count":   lambda: str(text.count(extra[0]) if extra else 0),
            "slice":   lambda: text[int(extra[0]):int(extra[1]) if len(extra) > 1 else None],
        }
        fn = ops.get(op)
        if fn is None:
            return f"Unknown operation '{op}'"
        try:
            return fn()
        except Exception as e:
            return f"Error: {e}"

    return registry
