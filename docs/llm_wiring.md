## One-time LLM wiring notes

- New `ai.providers` registry offers `llm(name="deepseek")` or `llm()` for the configured default.
- DeepSeek support mirrors the existing OpenAI handler; set `DEEPSEEK_API_KEY` (and optional `DEEPSEEK_MODEL`) in `.env`.
