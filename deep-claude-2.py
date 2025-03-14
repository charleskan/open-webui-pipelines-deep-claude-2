"""
title: DeepClaude2
author: charles.kan
description: A specialized pipe that combines DeepSeek Reasoner's Chain-of-Thought capabilities with Claude's response generation. It first sends the prompt to DeepSeek API to generate reasoning wrapped in tags, then passes both the original prompt and DeepSeek's reasoning to Claude to produce the final response. This creates an enhanced experience where Claude appears to "think" before answering.
version: 2.1.0
licence: MIT
github: https://github.com/charleskan/open-webui-pipelines-deep-claude-2
"""

import json
import httpx
import re
import asyncio
from typing import AsyncGenerator, Callable, Awaitable
from pydantic import BaseModel, Field

DATA_PREFIX = "data: "


def format_error(status_code, error) -> str:
    try:
        err_msg = json.loads(error).get("message", error.decode(errors="ignore"))[:200]
    except Exception:
        err_msg = error.decode(errors="ignore")[:200]
    return json.dumps({"error": f"HTTP {status_code}: {err_msg}"}, ensure_ascii=False)


async def claude_api_call(
    payload: dict, api_base_url: str, api_key: str
) -> AsyncGenerator[dict, None]:
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }

    async with httpx.AsyncClient(http2=True) as client:
        try:
            async with client.stream(
                "POST",
                f"{api_base_url}/messages",
                json=payload,
                headers=headers,
                timeout=300,
            ) as response:
                if response.status_code != 200:
                    error = await response.aread()
                    yield {"error": format_error(response.status_code, error)}
                    return

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue

                    try:
                        data = json.loads(line[6:])
                        event_type = data.get("type")

                        if event_type == "content_block_start":
                            yield {
                                "choices": [
                                    {
                                        "delta": {
                                            "content": data["content_block"]["text"]
                                        }
                                    }
                                ]
                            }
                        elif event_type == "content_block_delta":
                            yield {
                                "choices": [
                                    {"delta": {"content": data["delta"]["text"]}}
                                ]
                            }
                        elif event_type == "message_stop":
                            yield {"choices": [{"finish_reason": "stop"}]}
                        elif event_type == "message":
                            for content in data.get("content", []):
                                if content["type"] == "text":
                                    yield {
                                        "choices": [
                                            {"delta": {"content": content["text"]}}
                                        ]
                                    }

                    except (json.JSONDecodeError, KeyError) as e:
                        error_detail = f"ERROR - Content：{line}，Reason：{e}"
                        yield {"error": format_error("DataParseError", error_detail)}
                        return

        except Exception as e:
            yield {"error": format_error("ConnectionError", str(e))}


async def deepseek_api_call(
    payload: dict, api_base_url: str, api_key: str
) -> AsyncGenerator[dict, None]:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    async with httpx.AsyncClient(http2=True) as client:
        async with client.stream(
            "POST",
            f"{api_base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=300,
        ) as response:
            if response.status_code != 200:
                error = await response.aread()
                yield {"error": format_error(response.status_code, error)}
                return
            async for line in response.aiter_lines():
                if not line.startswith(DATA_PREFIX):
                    continue
                json_str = line[len(DATA_PREFIX) :]
                try:
                    data = json.loads(json_str)
                except json.JSONDecodeError as e:
                    error_detail = f"ERROR - Content：{json_str}，Reason：{e}"
                    yield {"error": format_error("JSONDecodeError", error_detail)}
                    return
                yield data


class Pipe:
    class Valves(BaseModel):
        DEEPSEEK_API_BASE_URL: str = Field(
            default="https://openrouter.ai/api/v1",
            description="DeepSeek API provider base URL",
        )
        DEEPSEEK_API_KEY: str = Field(
            default="", description="DeepSeek API provider key"
        )
        DEEPSEEK_API_MODEL: str = Field(
            default="deepseek/deepseek-r1",
            description="default deepseek-reasoner ",
        )
        ANTHROPIC_API_KEY: str = Field(
            default="",
            description="Claude API provider key",
            json_schema_extra={"format": "password"},
        )
        ANTHROPIC_API_BASE_URL: str = Field(
            default="https://api.anthropic.com/v1",
            description="Claude API provider base URL",
        )
        ANTHROPIC_API_MODEL: str = Field(
            default="claude-3-5-sonnet-latest",
            description="default claude-3-5-sonnet-latest ",
        )
        ANTHROPIC_API_MAX_TOKENS: int = Field(
            default=8192,
            description="max tokens for claude response",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = DATA_PREFIX
        self.emitter = None

    def pipes(self):
        return [{"id": "DeepClaude", "name": "DeepClaude"}]

    async def _emit(self, content: str) -> AsyncGenerator[str, None]:
        while content:
            yield content[0]
            content = content[1:]

    async def pipe(
        self,
        body: dict,
        __event_emitter__: Callable[[dict], Awaitable[None]] = None,
    ) -> AsyncGenerator[str, None]:
        self.emitter = __event_emitter__

        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "Missing api key"}, ensure_ascii=False)
            return

        deepseek_model_id = self.valves.DEEPSEEK_API_MODEL

        deepseek_response = ""
        deepseek_payload = {**body, "model": deepseek_model_id}

        async for chunk in self._emit("<think>"):
            yield chunk

        async for data in deepseek_api_call(
            payload=deepseek_payload,
            api_base_url=self.valves.DEEPSEEK_API_BASE_URL,
            api_key=self.valves.DEEPSEEK_API_KEY,
        ):
            if "error" in data:
                async for chunk in self._emit(data["error"]):
                    yield chunk
                return

            choice = data.get("choices", [{}])[0]
            delta = choice.get("delta", {})

            if content := delta.get("reasoning_content") or delta.get("reasoning"):
                async for chunk in self._emit(content):
                    yield chunk

            if choice.get("finish_reason"):
                break

        claude_messages = body.get("messages", []) + [
            {"role": "assistant", "content": deepseek_response}
        ]

        claude_payload = {
            "model": "claude-3-5-sonnet-20240620",
            "messages": claude_messages,
            # "stream": True,
            "max_tokens": self.valves.ANTHROPIC_API_MAX_TOKENS,
            **{k: v for k, v in body.items() if k not in ["model", "messages"]},
        }

        async for chunk in self._emit("</think>"):
            yield chunk

        async for claude_data in claude_api_call(
            payload=claude_payload,
            api_base_url=self.valves.ANTHROPIC_API_BASE_URL,
            api_key=self.valves.ANTHROPIC_API_KEY,
        ):
            if "error" in claude_data:
                async for chunk in self._emit(claude_data["error"]):
                    yield chunk
                return

            for choice in claude_data.get("choices", []):
                delta = choice.get("delta", {})
                if content := delta.get("content", ""):
                    async for chunk in self._emit(content):
                        yield chunk
