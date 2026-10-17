"""Small, bounded conversation context for follow-up query interpretation."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ConversationTurn:
    role: str
    content: str


@dataclass(frozen=True)
class ConversationContext:
    turns: tuple[ConversationTurn, ...] = ()
    max_turns: int = 12

    @classmethod
    def from_history(cls, history: object, *, max_turns: int = 12) -> ConversationContext:
        if not isinstance(history, list):
            return cls(max_turns=max_turns)
        turns: list[ConversationTurn] = []
        for item in history[-max_turns:]:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role in {"user", "assistant"} and isinstance(content, str) and content.strip():
                turns.append(ConversationTurn(role, content.strip()))
        return cls(tuple(turns), max_turns=max_turns)

    def as_messages(self) -> tuple[dict[str, str], ...]:
        return tuple({"role": turn.role, "content": turn.content} for turn in self.turns)
