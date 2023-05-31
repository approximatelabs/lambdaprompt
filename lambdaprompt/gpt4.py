from .prompt import Completion, Chat, AsyncCompletion, AsyncChat
from .backends import (
    AzureOpenAIGPT4Chat,
    AzureOpenAIGPT4Completion,
    OpenAIGPT4Completion,
    OpenAIGPT4Chat,
    AzureOpenAIGPT4Completion,
    AzureOpenAIGPT4Chat,
)


class GPT4Prompt(Completion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIGPT4Completion())


class AsyncGPT4Prompt(AsyncCompletion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIGPT4Completion())


class GPT4Chat(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIGPT4Chat())


class AsyncGPT4Chat(AsyncChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIGPT4Chat())


class AzureGPT4Prompt(Completion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAIGPT4Completion())


class AsyncAzureGPT4Prompt(AsyncCompletion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAIGPT4Completion())


class AzureGPT4Chat(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAIGPT4Chat())


class AsyncAzureGPT4Chat(AsyncChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAIGPT4Chat())
