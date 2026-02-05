#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2025 Beijing Volcano Engine Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""
Global LLM manager singleton wrapper
Provides global access to LLMManager instances
"""

import asyncio
import concurrent.futures
import json
import threading
from typing import Any, Dict, Optional

from opencontext.config.global_config import get_config
from opencontext.llm.llm_client import LLMClient, LLMType
from opencontext.storage.unified_storage import UnifiedStorage
from opencontext.utils.json_parser import parse_json_from_response
from opencontext.utils.logging_utils import get_logger

logger = get_logger(__name__)


class GlobalVLMClient:
    """
    Global LLM manager (singleton pattern)
    """

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        """Ensure singleton pattern"""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize VLM client"""
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._vlm_client: Optional[LLMClient] = None
                    self._vision_client: Optional[LLMClient] = None
                    self._auto_initialized = False
                    GlobalVLMClient._initialized = True

    @classmethod
    def get_instance(cls) -> "GlobalVLMClient":
        """
        Get global LLM manager instance
        """
        instance = cls()
        if not instance._auto_initialized and instance._vlm_client is None:
            instance._auto_initialize()
        return instance

    @classmethod
    def reset(cls):
        """Reset singleton instance (mainly for testing)"""
        with cls._lock:
            cls._instance = None
            cls._initialized = False

    def _auto_initialize(self):
        """Auto-initialize VLM client"""
        if self._auto_initialized:
            return
        from opencontext.tools.tools_executor import ToolsExecutor

        self._tools_executor = ToolsExecutor()
        try:
            vlm_config = get_config("vlm_model")
            if not vlm_config:
                logger.warning("No vlm config found in vlm_model")
                self._auto_initialized = True
                return

            self._vlm_client = LLMClient(llm_type=LLMType.CHAT, config=vlm_config)
            vision_config = get_config("vision_model")
            if vision_config:
                self._vision_client = LLMClient(llm_type=LLMType.CHAT, config=vision_config)
            logger.info("GlobalVLMClient auto-initialized successfully")
            self._auto_initialized = True
        except Exception as e:
            logger.error(f"GlobalVLMClient auto-initialization failed: {e}")
            self._auto_initialized = True

    def is_initialized(self) -> bool:
        return self._vlm_client is not None

    def _requires_vision(self, messages: list) -> bool:
        """
        Best-effort detection of whether the request contains image/video parts and therefore
        requires a vision-capable model.
        """
        try:
            for msg in messages or []:
                if not isinstance(msg, dict):
                    continue
                content = msg.get("content")
                if not isinstance(content, list):
                    continue
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("image_url", "input_image", "video_url", "input_video"):
                        return True
                    # Some providers may omit type but include payload keys.
                    if "image_url" in part or "video_url" in part:
                        return True
        except Exception:
            return False
        return False

    def _select_client(self, messages: list) -> LLMClient:
        if self._vision_client is not None and self._requires_vision(messages):
            return self._vision_client
        return self._vlm_client

    def reinitialize(self):
        """
        Thread-safe reinitialization of VLM client
        """
        with self._lock:
            try:
                vlm_config = get_config("vlm_model")
                if not vlm_config:
                    logger.error("No vlm config found during reinitialize")
                    raise ValueError("No vlm config found")
                new_client = LLMClient(llm_type=LLMType.CHAT, config=vlm_config)
                vision_config = get_config("vision_model")
                new_vision_client = (
                    LLMClient(llm_type=LLMType.CHAT, config=vision_config)
                    if vision_config
                    else None
                )
                self._vlm_client = new_client
                self._vision_client = new_vision_client
                logger.info("GlobalVLMClient reinitialized successfully")

            except Exception as e:
                logger.error(f"Failed to reinitialize VLM client: {e}")
                return False
            return True

    def generate_with_messages(
        self, messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
    ):
        client = self._select_client(messages)
        response = client.generate_with_messages(messages, **kwargs)
        call_count = 0
        while enable_executor:
            call_count += 1
            if call_count > max_calls:
                logger.warning(
                    f"Reached maximum tool call limit ({max_calls}), stopping further calls"
                )
                messages.append(
                    {
                        "role": "system",
                        "content": f"System notice: Maximum tool call limit ({max_calls}) reached. Cannot execute more tool calls. Please answer the user's question directly without attempting more tool calls.",
                    }
                )
                client = self._select_client(messages)
                response = client.generate_with_messages(messages, **kwargs)
                break
            message = response.choices[0].message
            if not message.tool_calls:
                break
            messages.append(message)
            tool_calls = message.tool_calls
            tool_call_info = []
            for tc in tool_calls:
                function_name = tc.function.name
                function_args = parse_json_from_response(tc.function.arguments)
                tool_call_info.append((tc.id, function_name, function_args))
            results = []
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future_to_tool = {
                    executor.submit(self._tools_executor.run, function_name, function_args): (
                        tool_id,
                        function_name,
                    )
                    for tool_id, function_name, function_args in tool_call_info
                }
                for future in concurrent.futures.as_completed(future_to_tool):
                    tool_id, function_name = future_to_tool[future]
                    try:
                        content = future.result()
                        # logger.info(f"Tool call {function_name} successful, result: {content}")
                        results.append((tool_id, function_name, content))
                    except Exception as e:
                        # logger.exception(f"Tool call {function_name} failed: {e}")
                        results.append((tool_id, function_name, "failed"))
            # logger.info(f"Tool call results: {results}")
            for tool_id, function_name, content in results:
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(content),
                        "tool_call_id": tool_id,
                    }
                )
            client = self._select_client(messages)
            response = client.generate_with_messages(messages, **kwargs)

        message = response.choices[0].message
        return message.content

    async def generate_with_messages_async(
        self, messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
    ):
        client = self._select_client(messages)
        response = await client.generate_with_messages_async(messages, **kwargs)
        call_count = 0
        while enable_executor:
            call_count += 1
            if call_count > max_calls:
                # logger.warning(f"Reached maximum tool call limit ({max_calls}), stopping further calls")
                messages.append(
                    {
                        "role": "system",
                        "content": f"System notice: Maximum tool call limit ({max_calls}) reached. Cannot execute more tool calls. Please answer the user's question directly without attempting more tool calls.",
                    }
                )
                client = self._select_client(messages)
                response = await client.generate_with_messages_async(messages, **kwargs)
                break
            message = response.choices[0].message
            if not message.tool_calls:
                break
            messages.append(message)
            tool_calls = message.tool_calls

            # Collect all async tasks for tool calls
            tasks = []
            tool_call_info = []
            for tc in tool_calls:
                function_name = tc.function.name
                function_args = parse_json_from_response(tc.function.arguments)
                if function_args is not None:
                    tasks.append(self._tools_executor.run_async(function_name, function_args))
                    tool_call_info.append((tc.id, function_name))
                else:
                    logger.error(
                        f"Failed to parse arguments for {function_name}: {tc.function.arguments}"
                    )

            # Execute all tool calls in parallel
            results = await asyncio.gather(*tasks)

            # Collect results and add to message list
            for result, (tool_id, function_name) in zip(results, tool_call_info):
                # logger.info(f"Tool call {function_name} successful, result: {result}")
                messages.append(
                    {
                        "role": "tool",
                        "name": function_name,
                        "content": json.dumps(result),
                        "tool_call_id": tool_id,
                    }
                )

            # Call LLM again
            client = self._select_client(messages)
            response = await client.generate_with_messages_async(messages, **kwargs)

        message = response.choices[0].message
        return message.content

    async def generate_for_agent_async(self, messages: list, tools: list = None, **kwargs):
        """
        Agent-specific generation method that returns raw response without auto-executing tool calls

        Args:
            messages: Message list
            tools: Available tool definitions
            **kwargs: Other parameters

        Returns:
            Raw LLM response object, including possible tool_calls
        """
        client = self._select_client(messages)
        response = await client.generate_with_messages_async(
            messages, tools=tools, **kwargs
        )
        return response

    async def generate_stream_for_agent(self, messages: list, tools: list = None, **kwargs):
        """
        Agent-specific streaming generation method
        """
        client = self._select_client(messages)
        async for chunk in client._openai_chat_completion_stream_async(
            messages, tools=tools, **kwargs
        ):
            yield chunk

    async def execute_tool_async(self, tool_call):
        """
        Execute a single tool call independently

        Args:
            tool_call: OpenAI format tool call object

        Returns:
            Tool execution result
        """
        function_name = tool_call.function.name
        function_args = parse_json_from_response(tool_call.function.arguments)

        if function_args is None:
            logger.error(
                f"Failed to parse arguments for {function_name}: {tool_call.function.arguments}"
            )
            return {"error": f"Failed to parse arguments for {function_name}"}

        try:
            result = await self._tools_executor.run_async(function_name, function_args)
            logger.info(f"Tool {function_name} executed successfully")
            return result
        except Exception as e:
            logger.exception(f"Tool {function_name} execution failed: {e}")
            return {"error": str(e)}


def is_initialized() -> bool:
    return GlobalVLMClient.get_instance()._auto_initialized


def generate_with_messages(
    messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
):
    return GlobalVLMClient.get_instance().generate_with_messages(
        messages, enable_executor, max_calls, **kwargs
    )


async def generate_with_messages_async(
    messages: list, enable_executor: bool = True, max_calls: int = 5, **kwargs
):
    return await GlobalVLMClient.get_instance().generate_with_messages_async(
        messages, enable_executor, max_calls, **kwargs
    )


async def generate_for_agent_async(messages: list, tools: list = None, **kwargs):
    return await GlobalVLMClient.get_instance().generate_for_agent_async(messages, tools, **kwargs)


async def generate_stream_for_agent(messages: list, tools: list = None, **kwargs):
    async for chunk in GlobalVLMClient.get_instance().generate_stream_for_agent(
        messages, tools, **kwargs
    ):
        yield chunk
