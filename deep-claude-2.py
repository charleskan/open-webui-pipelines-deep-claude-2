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
    """
    Claude API 流式调用核心函数
    """
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

                        # 事件类型映射到统一格式
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
                        error_detail = f"解析失败 - 内容：{line}，原因：{e}"
                        yield {"error": format_error("DataParseError", error_detail)}
                        return

        except Exception as e:
            yield {"error": format_error("ConnectionError", str(e))}


async def deepseek_api_call(
    payload: dict, api_base_url: str, api_key: str
) -> AsyncGenerator[dict, None]:
    """
    发送 DeepSeek API 请求，并以流式返回 JSON 数据。
    """
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
                    error_detail = f"解析失败 - 内容：{json_str}，原因：{e}"
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
        ANTHROPIC_API_BASE: str = Field(
            default="https://api.anthropic.com", description="Claude API provider base URL"
        )
        ANTHROPIC_API_MODEL: str = Field(
            default="claude-3-5-sonnet-latest",
            description="default claude-3-5-sonnet-latest ",
        )

    def __init__(self):
        self.valves = self.Valves()
        self.data_prefix = DATA_PREFIX
        self.thinking = -1  # -1:未开始 0:思考中 1:已回答
        self.emitter = None

    def pipes(self):
        return [{"id": "DeepClaude", "name": "DeepClaude"}]

    async def _emit(
        self, content: str, is_think_tag: bool = False
    ) -> AsyncGenerator[str, None]:
        """
        通用內容發送器 (類方法版本)
        """
        while content:
            yield content[0]
            content = content[1:]

    async def pipe(
        self, body: dict, __event_emitter__: Callable[[dict], Awaitable[None]] = None
    ) -> AsyncGenerator[str, None]:
        """
        使用函数式实现的 DeepSeek API 调用，结合 Pipe 类中的业务逻辑处理。
        """
        self.thinking = -1
        self.emitter = __event_emitter__

        if not self.valves.DEEPSEEK_API_KEY:
            yield json.dumps({"error": "未配置API密钥"}, ensure_ascii=False)
            return

        # 模型ID提取及 payload 预处理
        deepseek_model_id = self.valves.DEEPSEEK_API_MODEL

        deepseek_response = ""
        deepseek_payload = {**body, "model": deepseek_model_id}

        deepseek_api_base_url = self.valves.DEEPSEEK_API_BASE_URL
        deepseek_api_key = self.valves.DEEPSEEK_API_KEY

        claude_model_id = self.valves.ANTHROPIC_API_MODEL
        claude_api_base_url = self.valves.ANTHROPIC_API_BASE
        claude_api_key = self.valves.ANTHROPIC_API_KEY

        async for chunk in self._emit("[WAITING DEEPSEEK]\n"):
            yield chunk

        async for data in deepseek_api_call(
            payload=deepseek_payload,
            api_base_url=deepseek_api_base_url,
            api_key=deepseek_api_key,
        ):
            if "error" in data:
                async for chunk in self._emit(data["error"]):
                    yield chunk
                return

            # 先解析数据再操作
            choice = data.get("choices", [{}])[0]
            delta = choice.get("delta", {})  # 先获取delta

            # 现在可以安全访问delta
            reasoning = delta.get("reasoning_content") or delta.get("reasoning")
            if reasoning:
                if self.thinking == -1:
                    self.thinking = 0
                    async for chunk in self._emit("<think>", is_think_tag=True):
                        yield chunk
                async for chunk in self._emit(reasoning):
                    yield chunk
                deepseek_response += reasoning

            # 内容流式处理
            if content := delta.get("reasoning_content") or delta.get("reasoning"):
                async for chunk in self._emit(content):
                    yield chunk

            if choice.get("finish_reason"):
                break  # 结束DeepSeek处理

        # 构建增强后的消息历史
        claude_messages = body.get("messages", []) + [
            {"role": "assistant", "content": deepseek_response}
        ]

        claude_payload = {
            "model": self.valves.ANTHROPIC_API_MODEL,
            "messages": claude_messages,
            "stream": True,
            "max_tokens": 8192,
            **{k: v for k, v in body.items() if k not in ["model", "messages"]},
        }

        async for chunk in self._emit("</think>\n[WAITING CLAUDE]\n"):
            yield chunk

        # 第三阶段：处理Claude响应
        async for claude_data in claude_api_call(
            payload=claude_payload,
            api_base_url=self.valves.ANTHROPIC_API_BASE,
            api_key=self.valves.ANTHROPIC_API_KEY,
        ):
            if "error" in claude_data:
                async for chunk in self._emit(claude_data["error"]):
                    yield chunk
                return

            # 解析Claude响应结构
            for choice in claude_data.get("choices", []):
                delta = choice.get("delta", {})
                if content := delta.get("content", ""):
                    async for chunk in self._emit(content):
                        yield chunk
