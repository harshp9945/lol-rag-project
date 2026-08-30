"""LLM backends behind one interface. The mock backend makes the whole
pipeline testable offline with zero API cost; real backends (Ollama,
Anthropic) plug in for actual generation.
"""
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...
    @property
    @abstractmethod
    def name(self) -> str: ...


class MockBackend(LLMBackend):
    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str) -> str:
        n_sources = prompt.count("[source:")
        return f"[MOCK ANSWER based on {n_sources} retrieved chunk(s)]"


class OllamaBackend(LLMBackend):
    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self._model = model
        self._host = host

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def generate(self, prompt: str) -> str:
        import json, urllib.request
        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps({"model": self._model, "prompt": prompt, "stream": False}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"]


class AnthropicBackend(LLMBackend):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic
        self._client = anthropic.Anthropic()
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model, max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
