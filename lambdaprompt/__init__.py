import nest_asyncio

from .gpt3 import AsyncGPT3Chat, AsyncGPT3Prompt, GPT3Chat, GPT3Prompt
from .gpt4 import (
    GPT4Prompt,
    GPT4Chat,
    AsyncGPT4Prompt,
    AsyncGPT4Chat,
    AsyncAzureGPT4Chat,
    AsyncAzureGPT4Prompt,
    AzureGPT4Chat,
    AzureGPT4Prompt,
)

from .prompt import (
    AsyncPrompt,
    Prompt,
    prompt,
    Completion,
    AsyncCompletion,
    Chat,
    AsyncChat,
    register_call_callback,
    register_creation_callback,
)
from . import backends

nest_asyncio.apply()

gpt4_prompts = [
    "AsyncGPT4Prompt",
    "GPT4Prompt",
    "AsyncGPT4Chat",
    "GPT4Chat",
]
gpt4_azure_prompts = [
    "AzureGPT4Prompt",
    "AsyncAzureGPT4Prompt",
    "AzureGPT4Chat",
    "AsyncAzureGPT4Chat",
]
gpt3_prompts = [
    "AsyncGPT3Prompt",
    "GPT3Prompt",
    "AsyncGPT3Chat",
    "GPT3Chat",
]

__all__ = (
    gpt4_prompts
    + gpt4_azure_prompts
    + gpt3_prompts
    + [
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
)
