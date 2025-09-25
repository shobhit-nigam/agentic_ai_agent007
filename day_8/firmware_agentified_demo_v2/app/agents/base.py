from dataclasses import dataclass
from typing import Any, Dict, Optional

@dataclass
class Message:
    type: str
    payload: Dict[str, Any]

class Agent:
    def __init__(self, name: str, trace=None):
        self.name = name
        self.trace = trace

    def perceive(self, msg: Message) -> Dict[str, Any]:
        return msg.payload

    def think(self, state: Dict[str, Any]) -> Optional[Message]:
        raise NotImplementedError

    def act(self, decision: Optional[Message]) -> Optional[Message]:
        return decision

    def handle(self, msg: Message) -> Optional[Message]:
        state = self.perceive(msg)
        decision = self.think(state)
        out = self.act(decision)
        if self.trace:
            self.trace.record(self.name, msg, decision, out)
        return out
