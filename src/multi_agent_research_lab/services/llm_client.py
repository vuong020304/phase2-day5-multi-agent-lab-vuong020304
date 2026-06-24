"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

from dataclasses import dataclass

from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from multi_agent_research_lab.core.config import get_settings


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
        self.model = settings.openai_model

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion with robust retry and token calculation."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )

        content = response.choices[0].message.content or ""

        input_tokens = None
        output_tokens = None
        cost_usd = None

        if response.usage:
            input_tokens = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens
            # Pricing for gpt-4o-mini: $0.150 / 1M input tokens, $0.600 / 1M output tokens
            # Pricing for gpt-4o: $2.50 / 1M input tokens, $10.00 / 1M output tokens
            if "gpt-4o-mini" in self.model:
                cost_usd = (input_tokens * 0.15 + output_tokens * 0.60) / 1_000_000
            elif "gpt-4o" in self.model:
                cost_usd = (input_tokens * 2.50 + output_tokens * 10.00) / 1_000_000
            else:
                cost_usd = (input_tokens * 0.50 + output_tokens * 1.50) / 1_000_000

        return LLMResponse(
            content=content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost_usd,
        )
