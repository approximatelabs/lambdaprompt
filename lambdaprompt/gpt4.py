from .prompt import Completion, Chat, AsyncCompletion, AsyncChat
from .backends import AzureOpenAIChat, AzureOpenAICompletion


class AzureGPT4Prompt(Completion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAICompletion())


class AsyncAzureGPT4Prompt(AsyncCompletion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAICompletion())


class AzureGPT4Chat(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAIChat())


class AsyncAzureGPT4Chat(AsyncChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=AzureOpenAIChat())
