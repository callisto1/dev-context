import httpx

from src.config import settings


class LLMClient:
    """OpenAI-compatible client for vLLM-served Qwen3-Coder."""

    def __init__(self):
        self.base_url = settings.llm_base_url
        self.model = settings.llm_model
        self.api_key = settings.llm_api_key
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: list[dict],
        max_tokens: int | None = None,
        temperature: float | None = None,
        stream: bool = False,
        model: str | None = None,
    ) -> str:
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "temperature": temperature if temperature is not None else settings.llm_temperature,
            "stream": stream,
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def chat_stream(self, messages: list[dict], max_tokens: int | None = None, model: str | None = None):
        payload = {
            "model": model or self.model,
            "messages": messages,
            "max_tokens": max_tokens or settings.llm_max_tokens,
            "temperature": settings.llm_temperature,
            "stream": True,
        }

        async with self.client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    chunk = line[6:]
                    if chunk == "[DONE]":
                        break
                    import json

                    data = json.loads(chunk)
                    delta = data["choices"][0].get("delta", {})
                    if content := delta.get("content"):
                        yield content

    async def describe_image(
        self,
        image_b64: str,
        context: str = "",
        model: str | None = None,
    ) -> str:
        prompt = "Describe this image in detail. Focus on any technical content: diagrams, architecture, data flows, labels, and text visible in the image."
        if context:
            prompt += f"\n\nThis image is from a document titled: {context}"

        payload = {
            "model": model or settings.llm_vision_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                        },
                    ],
                }
            ],
            "max_tokens": 1024,
            "temperature": 0.2,
            "stream": False,
        }

        response = await self.client.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers={"Authorization": f"Bearer {self.api_key}"},
            timeout=180.0,
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]

    async def close(self):
        await self.client.aclose()


llm_client = LLMClient()
