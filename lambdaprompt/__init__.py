import nest_asyncio
import asyncio

from .gpt3 import AsyncGPT3Chat, AsyncGPT3Prompt, GPT3Chat, GPT3Prompt
from .prompt import (AsyncPrompt, Prompt, prompt, Completion, AsyncCompletion, Chat, AsyncChat, register_call_callback,
                     register_creation_callback)
from . import backends

try:
    asyncio.get_running_loop()
except RuntimeError:
    pass

try:
    nest_asyncio.apply()
except RuntimeError:
    pass

__all__ = [
    "AsyncGPT3Prompt",
    "GPT3Prompt",
    "AsyncGPT3Chat",
    "GPT3Chat",
    "AsyncPrompt",
    "Prompt",
    "prompt",
    "register_call_callback",
    "register_creation_callback",
    "backends",
    "Completion",
    "AsyncCompletion",
    "Chat",
    "AsyncChat",
]
