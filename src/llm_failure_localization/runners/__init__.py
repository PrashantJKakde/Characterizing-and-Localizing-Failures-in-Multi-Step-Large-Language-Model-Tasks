"""Model serving/inference backends (open-weights only): HF Transformers, vLLM, Ollama.

Implemented in M2. Every runner exposes the same `generate(prompt, **kwargs) -> str`
interface so swapping models is a config change, not a code change.
"""
