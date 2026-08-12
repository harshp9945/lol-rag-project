"""LLM backends behind one interface, same trick as the retrievers.

The mock backend matters more than it looks: it makes the ENTIRE
pipeline -- chunking, retrieval, guardrail, prompt assembly -- testable
offline, in CI, with zero API cost and zero flakiness. Real backends
(Ollama local, Anthropic API) plug in for actual answer generation.
"""
from abc import ABC, abstractmethod


class LLMBackend(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str: ...

    @property
    @abstractmethod
    def name(self) -> str: ...


class MockBackend(LLMBackend):
    """Deterministic echo backend for tests and CI.

    Returns a stub answer that embeds the prompt's retrieved context
    markers, so tests can assert that retrieval actually fed the prompt.
    """

    @property
    def name(self) -> str:
        return "mock"

    def generate(self, prompt: str) -> str:
        n_sources = prompt.count("[source:")
        return f"[MOCK ANSWER based on {n_sources} retrieved chunk(s)]"


class OllamaBackend(LLMBackend):
    """Local model via Ollama. Requires `ollama serve` and a pulled model.

    Note from hard experience: llama3.1 supports tool-calling, llama3
    does NOT -- irrelevant for plain RAG generation, but it matters when
    this backend is reused by the agent project.
    """

    def __init__(self, model: str = "llama3.1", host: str = "http://localhost:11434"):
        self._model = model
        self._host = host

    @property
    def name(self) -> str:
        return f"ollama:{self._model}"

    def generate(self, prompt: str) -> str:
        import json
        import urllib.request

        req = urllib.request.Request(
            f"{self._host}/api/generate",
            data=json.dumps(
                {"model": self._model, "prompt": prompt, "stream": False}
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            return json.loads(resp.read())["response"]


class AnthropicBackend(LLMBackend):
    """Anthropic API backend. Requires ANTHROPIC_API_KEY in the environment
    and `pip install anthropic`."""

    def __init__(self, model: str = "claude-sonnet-4-6"):
        import anthropic  # deferred so the package imports without the SDK

        self._client = anthropic.Anthropic()
        self._model = model

    @property
    def name(self) -> str:
        return f"anthropic:{self._model}"

    def generate(self, prompt: str) -> str:
        msg = self._client.messages.create(
            model=self._model,
            max_tokens=600,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text
