"""
NFN Agents — Chat, Code, and Reasoning agents powered by NFNInferenceEngine.

ChatAgent     : multi-turn conversational assistant
CodeAgent     : code completion / generation / explanation
ReasoningAgent: step-by-step ReAct-style reasoning with tool use
"""

import ast
import json
import re
import traceback
from typing import Any, Dict, List, Optional

from inference.engine import NFNInferenceEngine


# ─────────────────────────────────────────────────────────────────────────────
# Chat Agent
# ─────────────────────────────────────────────────────────────────────────────

class ChatAgent:
    """Simple stateless chat agent — builds prompt from message history."""

    SYSTEM = (
        "Tu es NFN (Neural Fractal Network), un assistant IA avancé basé sur une "
        "architecture neuronale fractale. Tu es curieux, précis, et capable de raisonner "
        "à plusieurs niveaux d'abstraction. Réponds en français ou dans la langue de l'utilisateur."
    )

    def __init__(self, engine: NFNInferenceEngine):
        self.engine = engine

    def reply(
        self,
        messages: List[Dict[str, str]],
        system: Optional[str] = None,
        max_tokens: int = 400,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.95,
    ) -> str:
        return self.engine.chat(
            messages,
            system=system or self.SYSTEM,
            max_new_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )


# ─────────────────────────────────────────────────────────────────────────────
# Code Agent
# ─────────────────────────────────────────────────────────────────────────────

class CodeAgent:
    """
    Code completion and generation agent.

    Supports:
      - complete : continue an incomplete code snippet
      - explain  : describe what code does
      - refactor : suggest improvements
      - generate : create code from a description
    """

    def __init__(self, engine: NFNInferenceEngine):
        self.engine = engine

    def _build_prompt(self, instruction: str, code: str, language: str, task: str) -> str:
        lang_tag = language.lower()
        if task == "complete":
            return (
                f"<sys>Tu es un expert en {language}. Complète le code suivant.</sys>\n"
                f"<usr>Complète ce code {language}:\n```{lang_tag}\n{code}\n```</usr>\n"
                f"<ast>```{lang_tag}\n{code}"
            )
        elif task == "explain":
            return (
                f"<sys>Tu es un expert en {language}. Explique le code.</sys>\n"
                f"<usr>Explique ce code:\n```{lang_tag}\n{code}\n```</usr>\n"
                f"<ast>Ce code"
            )
        elif task == "refactor":
            return (
                f"<sys>Tu es un expert en {language}. Propose une version améliorée.</sys>\n"
                f"<usr>Refactorise:\n```{lang_tag}\n{code}\n```\n{instruction}</usr>\n"
                f"<ast>Version améliorée:\n```{lang_tag}\n"
            )
        else:  # generate
            return (
                f"<sys>Tu es un expert développeur {language}.</sys>\n"
                f"<usr>{instruction}</usr>\n"
                f"<ast>```{lang_tag}\n"
            )

    def complete(
        self,
        code: str,
        instruction: str = "",
        language: str = "python",
        max_tokens: int = 500,
        temperature: float = 0.4,
    ) -> Dict[str, Any]:
        task = "complete" if not instruction else "generate"
        prompt = self._build_prompt(instruction, code, language, task)
        result = self.engine.generate(
            prompt, max_new_tokens=max_tokens,
            temperature=temperature, top_k=40, top_p=0.9,
        )
        return {
            "language": language,
            "task": task,
            "result": result,
            "prompt": prompt,
        }

    def explain(self, code: str, language: str = "python", max_tokens: int = 300) -> Dict:
        prompt = self._build_prompt("", code, language, "explain")
        result = self.engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.5)
        return {"explanation": result, "language": language}

    def refactor(self, code: str, instruction: str = "", language: str = "python",
                 max_tokens: int = 500) -> Dict:
        prompt = self._build_prompt(instruction, code, language, "refactor")
        result = self.engine.generate(prompt, max_new_tokens=max_tokens, temperature=0.3)
        return {"refactored": result, "language": language}


# ─────────────────────────────────────────────────────────────────────────────
# Reasoning Agent (ReAct-style)
# ─────────────────────────────────────────────────────────────────────────────

class ReasoningAgent:
    """
    A ReAct-style agent that decomposes goals into steps and uses tools.

    Tools available:
      - calculate(expr)  : safe Python math evaluation
      - search(query)    : searches the built-in knowledge base
      - think(thought)   : explicit reasoning step
      - answer(text)     : final answer
    """

    SYSTEM = (
        "Tu es NFN-Agent, un agent de raisonnement basé sur le Neural Fractal Network. "
        "Tu résous des problèmes étape par étape en utilisant le format:\n"
        "Réflexion: [ta réflexion]\n"
        "Action: [outil]([paramètres])\n"
        "Observation: [résultat]\n"
        "... (répète si nécessaire)\n"
        "Réponse finale: [ta réponse]\n\n"
        "Outils disponibles: calculate(expr), search(query), think(thought), answer(text)"
    )

    KNOWLEDGE_BASE = {
        "nfn": (
            "Le Neural Fractal Network (NFN) est une architecture neuronale à topologie fractale "
            "avec des connexions sinusoïdales paramétriques (A·sin(ωt+φ)). Il combine "
            "auto-similarité, oscillations de phase, et apprentissage profond."
        ),
        "fractal": (
            "Une fractale est une structure géométrique dont chaque partie est une copie "
            "à échelle réduite de l'ensemble. Exemples: Sierpinski, Cantor, Mandelbrot."
        ),
        "transformer": (
            "Le Transformer est une architecture de réseau de neurones basée sur l'attention "
            "multi-têtes, introduite par Vaswani et al. en 2017."
        ),
        "agi": (
            "L'Intelligence Artificielle Générale (AGI) désigne une IA capable d'effectuer "
            "n'importe quelle tâche intellectuelle humaine avec des capacités générales."
        ),
    }

    def __init__(self, engine: NFNInferenceEngine):
        self.engine = engine

    # ── Tools ─────────────────────────────────────────────────────────────────

    def _tool_calculate(self, expr: str) -> str:
        try:
            # Restricted eval — math only
            allowed = {k: v for k, v in __builtins__.items()
                       if k in ("abs", "round", "min", "max", "sum", "pow")} \
                if isinstance(__builtins__, dict) else {}
            import math
            allowed.update({k: v for k, v in vars(math).items() if not k.startswith("_")})
            result = eval(expr, {"__builtins__": {}}, allowed)
            return str(result)
        except Exception as e:
            return f"Erreur de calcul: {e}"

    def _tool_search(self, query: str) -> str:
        q = query.lower()
        for key, val in self.KNOWLEDGE_BASE.items():
            if key in q:
                return val
        return f"Aucune information trouvée pour '{query}'."

    def _tool_think(self, thought: str) -> str:
        return f"[Réflexion enregistrée: {thought}]"

    def _parse_action(self, text: str):
        """Extract tool name and args from 'tool(args)' pattern."""
        m = re.search(r'(\w+)\(([^)]*)\)', text)
        if not m:
            return None, None
        tool = m.group(1).strip()
        args = m.group(2).strip().strip('"\'')
        return tool, args

    def _execute_tool(self, tool: str, args: str) -> str:
        if tool == "calculate":
            return self._tool_calculate(args)
        elif tool == "search":
            return self._tool_search(args)
        elif tool == "think":
            return self._tool_think(args)
        elif tool == "answer":
            return args
        else:
            return f"Outil inconnu: {tool}"

    # ── Main run loop ─────────────────────────────────────────────────────────

    async def run(
        self,
        goal: str,
        history: List[Dict] = [],
        max_steps: int = 5,
    ) -> Dict[str, Any]:
        steps = []
        prompt = (
            f"<sys>{self.SYSTEM}</sys>\n"
            f"<usr>Objectif: {goal}</usr>\n"
            f"<ast>Réflexion: Je vais analyser ce problème méthodiquement.\n"
        )

        final_answer = ""

        for step_idx in range(max_steps):
            # Generate next step
            continuation = self.engine.generate(
                prompt,
                max_new_tokens=150,
                temperature=0.5,
                top_k=40,
                top_p=0.9,
            )

            # Parse action
            action_match = re.search(r'Action:\s*(.+)', continuation)
            if not action_match:
                # No action found — assume final answer
                answer_match = re.search(r'Réponse finale:\s*(.+)', continuation, re.DOTALL)
                final_answer = answer_match.group(1).strip() if answer_match else continuation.strip()
                steps.append({
                    "step": step_idx + 1,
                    "thought": continuation,
                    "action": None,
                    "observation": None,
                    "final": True,
                })
                break

            action_str = action_match.group(1).strip()
            tool, args = self._parse_action(action_str)
            observation = self._execute_tool(tool, args) if tool else "Action non reconnue."

            step = {
                "step": step_idx + 1,
                "thought": continuation,
                "action": action_str,
                "tool": tool,
                "args": args,
                "observation": observation,
                "final": False,
            }
            steps.append(step)

            # Check for final answer
            if tool == "answer":
                final_answer = args
                step["final"] = True
                break

            # Extend prompt with observation
            prompt += f"{continuation}\nObservation: {observation}\nRéflexion: "

        return {
            "goal": goal,
            "steps": steps,
            "final_answer": final_answer or (steps[-1]["thought"] if steps else ""),
            "n_steps": len(steps),
        }
