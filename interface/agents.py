"""
NFN AGI Agents v4.0

Agents built on AGIInferenceEngine:

  ChatAgent      — multi-turn conversation with persistent episodic memory
                   automatically retrieves relevant past context (RAG)
  ThinkAgent     — exposes internal reasoning rounds, shows think() in action
  ToolAgent      — ReAct-style with real tools via ToolCallingModel
  LearnAgent     — interactive continual learning from conversation
"""

import json
import time
from typing import Any, Dict, List, Optional

from inference.engine import AGIInferenceEngine


# ─────────────────────────────────────────────────────────────────────────────
# Chat Agent — persistent memory, RAG retrieval
# ─────────────────────────────────────────────────────────────────────────────

class ChatAgent:
    """
    Multi-turn conversational agent with episodic memory.

    Each turn:
      1. Retrieve relevant past context from KnowledgeStore (RAG)
      2. Build prompt from system + retrieved context + conversation history
      3. Generate reply with episodic write-through
      4. Optionally learn the turn (user message → episodic memory)
    """

    SYSTEM = (
        "Tu es NFN (Neural Fractal Network), une intelligence artificielle avancée "
        "basée sur une architecture fractale avec mémoire épisodique et raisonnement causal. "
        "Tu te souviens des conversations passées, tu raisonnes à plusieurs niveaux d'abstraction, "
        "et tu peux appeler des outils externes. "
        "Réponds avec précision, curiosité, et profondeur."
    )

    def __init__(
        self,
        engine: AGIInferenceEngine,
        use_rag: bool      = True,
        rag_top_k: int     = 3,
        learn_turns: bool  = True,   # write each user turn to episodic memory
        think_rounds: int  = 0,      # 0 = no thinking, >0 = internal reasoning
        use_tools: bool    = False,
    ):
        self.engine      = engine
        self.use_rag     = use_rag
        self.rag_top_k   = rag_top_k
        self.learn_turns = learn_turns
        self.think_rounds = think_rounds
        self.use_tools   = use_tools

    def reply(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int   = 512,
        temperature: float = 0.8,
        top_k: int        = 50,
        top_p: float      = 0.95,
    ) -> Dict[str, Any]:
        """
        Generate a reply.
        Returns {
            "reply": str,
            "retrieved": [...],
            "think_thoughts": [...],
            "elapsed_s": float,
        }
        """
        t0 = time.time()
        sys_text = system or self.SYSTEM

        # Optionally learn the latest user message
        user_text = next(
            (m["content"] for m in reversed(messages) if m["role"] == "user"),
            None,
        )
        if self.learn_turns and user_text:
            self.engine.learn(user_text, source="conversation")

        # Retrieve relevant memories
        retrieved = []
        if self.use_rag and user_text:
            retrieved = self.engine.retrieve(user_text, top_k=self.rag_top_k)

        # Inject retrieved context into system prompt
        if retrieved:
            ctx = "\n".join(
                f"[Mémoire {i+1} (score={r['score']:.2f})]: {r['text']}"
                for i, r in enumerate(retrieved)
            )
            sys_text = sys_text + f"\n\nContexte de ta mémoire:\n{ctx}"

        # Optional internal thinking
        thoughts = []
        if self.think_rounds > 0 and user_text:
            result   = self.engine.think(user_text, n_rounds=self.think_rounds)
            thoughts = result["thoughts"]
            # Prepend thinking summary to system
            thought_summary = " → ".join(
                t[:80] + "..." if len(t) > 80 else t for t in thoughts
            )
            sys_text = sys_text + f"\n\n[Raisonnement interne]: {thought_summary}"

        reply_text = self.engine.chat(
            messages,
            system=sys_text,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            use_tools=self.use_tools,
            write_memory=True,
        )

        return {
            "reply":          reply_text,
            "retrieved":      retrieved,
            "think_thoughts": thoughts,
            "elapsed_s":      round(time.time() - t0, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Think Agent — exposes internal reasoning transparently
# ─────────────────────────────────────────────────────────────────────────────

class ThinkAgent:
    """
    Exposes NFN's internal thinking process.

    Shows:
      - Goal phase set from the prompt
      - N rounds of internal reasoning (RecursiveReasoner)
      - Free energy / belief compression per round
      - Final answer after reasoning
    """

    def __init__(self, engine: AGIInferenceEngine, default_rounds: int = 3):
        self.engine = engine
        self.default_rounds = default_rounds

    def think_and_answer(
        self,
        question: str,
        n_rounds: Optional[int] = None,
        max_answer_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Returns {
            "question": str,
            "thoughts": [str],        # one per round
            "answer": str,
            "elapsed_s": float,
        }
        """
        t0     = time.time()
        rounds = n_rounds or self.default_rounds

        think_result = self.engine.think(question, n_rounds=rounds)
        thoughts     = think_result["thoughts"]
        final_prompt = think_result["final_prompt"]

        answer = self.engine.generate(
            final_prompt + "\nRéponse:",
            max_new_tokens=max_answer_tokens,
            temperature=temperature,
            top_p=top_p,
            write_memory=True,
        )

        return {
            "question":  question,
            "thoughts":  thoughts,
            "answer":    answer.strip(),
            "elapsed_s": round(time.time() - t0, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Tool Agent — ReAct-style with real tools
# ─────────────────────────────────────────────────────────────────────────────

class ToolAgent:
    """
    Agentic task solver using ToolCallingModel.

    The model generates text freely; when it emits a <tool_call> block,
    the tool is dispatched and the result is injected back into context.

    You can add custom tools via engine.register_tool(...).
    """

    SYSTEM = (
        "Tu es NFN-Agent, un agent IA avancé capable de résoudre des tâches complexes "
        "en combinant raisonnement interne, mémoire épisodique, et appels d'outils externes.\n"
        "Résous les problèmes étape par étape. Quand tu as besoin d'un calcul ou d'une "
        "opération précise, utilise les outils disponibles."
    )

    def __init__(self, engine: AGIInferenceEngine):
        self.engine = engine

    def run(
        self,
        task: str,
        system: Optional[str] = None,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
    ) -> Dict[str, Any]:
        """
        Run the tool agent on a task.
        Returns {"task": str, "result": str, "elapsed_s": float}
        """
        t0 = time.time()
        sys_text = system or self.SYSTEM

        result = self.engine.tool_model.generate(
            prompt=task,
            system_prompt=sys_text,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

        # Also learn from this interaction
        self.engine.learn(f"Task: {task}\nResult: {result}", source="agent_run")

        return {
            "task":      task,
            "result":    result,
            "elapsed_s": round(time.time() - t0, 3),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Learn Agent — interactive continual learning
# ─────────────────────────────────────────────────────────────────────────────

class LearnAgent:
    """
    Interactive continual learning agent.

    Commands:
      learn(text, source)      → store text in episodic memory
      retrieve(query, top_k)   → find relevant memories
      rag(prompt)              → generate with retrieval augmentation
      stats()                  → memory statistics
      save(path)               → persist knowledge store
    """

    def __init__(self, engine: AGIInferenceEngine):
        self.engine = engine

    def learn(self, text: str, source: str = "user") -> Dict:
        return self.engine.learn(text, source=source)

    def retrieve(self, query: str, top_k: int = 4) -> List[Dict]:
        return self.engine.retrieve(query, top_k=top_k)

    def rag(
        self,
        prompt: str,
        top_k: int = 3,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
    ) -> str:
        return self.engine.generate_with_rag(
            prompt, top_k=top_k,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )

    def stats(self) -> Dict:
        return {
            **self.engine.learner.stats(),
            **{"tools": list(self.engine.registry._tools.keys())},
        }

    def save(self, path: str) -> str:
        self.engine.learner.save_store(path)
        return path


# ─────────────────────────────────────────────────────────────────────────────
# Code Agent (kept for API compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class CodeAgent:
    """Code completion / generation / explanation."""

    def __init__(self, engine: AGIInferenceEngine):
        self.engine = engine

    def complete(self, code: str, language: str = "python", max_tokens: int = 512) -> str:
        prompt = (
            f"<system>Tu es un expert {language}. Complète le code.</system>\n"
            f"<user>```{language}\n{code}\n```</user>\n"
            f"<assistant>```{language}\n{code}"
        )
        return self.engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.3)

    def explain(self, code: str, language: str = "python", max_tokens: int = 300) -> str:
        prompt = (
            f"<system>Tu es un expert {language}. Explique ce code.</system>\n"
            f"<user>```{language}\n{code}\n```</user>\n"
            f"<assistant>Ce code"
        )
        return self.engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.5)

    def generate(self, description: str, language: str = "python", max_tokens: int = 512) -> str:
        prompt = (
            f"<system>Expert {language}.</system>\n"
            f"<user>{description}</user>\n"
            f"<assistant>```{language}\n"
        )
        return self.engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.4)


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Agent (ReAct-style, kept for API compatibility)
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningAgent(ToolAgent):
    """Alias for ToolAgent — backwards compatible."""

    # Allowed names for sandboxed math evaluation
    _MATH_SAFE = {
        "abs": abs, "round": round, "min": min, "max": max,
        "sum": sum, "pow": pow, "len": len, "int": int, "float": float,
    }
    _MATH_BLOCKED = frozenset([
        "__import__", "__builtins__", "eval", "exec", "compile",
        "open", "getattr", "setattr", "delattr", "globals", "locals",
        "vars", "dir", "type", "object", "print", "input",
    ])

    def _tool_calculate(self, expression: str) -> str:
        """
        Safe sandboxed math evaluator.

        Supports arithmetic, builtins (abs, round, min, max, sum, pow),
        and the full math module.  Blocks any access to dunders, system
        calls, function/class definitions, and attribute access.
        """
        import math, ast

        expr = expression.strip()

        # Block obvious injection patterns before even parsing
        for blocked in self._MATH_BLOCKED:
            if blocked in expr:
                return f"Erreur: '{blocked}' non autorisé"
        if "__" in expr:
            return "Erreur: accès aux attributs spéciaux non autorisé"
        if any(kw in expr for kw in ("def ", "class ", "lambda ", "import ")):
            return "Erreur: déclarations non autorisées"

        # Build safe namespace: builtins + math module contents
        # math.pow always returns float (matches test expectations for pow(2,3)='8.0')
        safe_ns: Dict = {k: v for k, v in vars(math).items()
                         if not k.startswith("_")}
        safe_ns.update(self._MATH_SAFE)
        safe_ns["pow"] = math.pow   # override builtin pow → always float

        try:
            tree = ast.parse(expr, mode="eval")
        except SyntaxError as e:
            return f"Erreur: syntaxe invalide — {e}"

        # AST whitelist: only allow safe node types
        _allowed = (
            ast.Expression, ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
            ast.Call, ast.Constant, ast.List, ast.Tuple,
            ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
            ast.Pow, ast.USub, ast.UAdd, ast.And, ast.Or, ast.Not,
            ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
            ast.Load, ast.Name, ast.keyword,
        )
        for node in ast.walk(tree):
            if not isinstance(node, _allowed):
                return f"Erreur: opération non autorisée ({type(node).__name__})"
            if isinstance(node, ast.Name) and node.id not in safe_ns:
                return f"Erreur: '{node.id}' non autorisé"
            if isinstance(node, ast.Attribute):
                return "Erreur: accès aux attributs non autorisé"

        try:
            result = eval(compile(tree, "<calc>", "eval"), {"__builtins__": {}}, safe_ns)
            return str(result)
        except Exception as e:
            return f"Erreur: {e}"

    async def run(self, goal: str, history=None, max_steps: int = 5) -> Dict:
        result = super().run(goal, max_new_tokens=max_steps * 200)
        return {
            "goal":         goal,
            "steps":        [],
            "final_answer": result["result"],
            "n_steps":      0,
        }
