"""Replaceable local inference engines.

Every engine is a `LocalModel` (see app/models/base.py): `load()` places weights on the local GPU and
`generate_sync()` runs one chat completion. The default transformers engine lives in
app/models/text_model.py / vision_model.py; this package adds llama.cpp (GGUF), vLLM and Ollama.
"""

ENGINES = ("transformers", "llama_cpp", "vllm", "ollama")
