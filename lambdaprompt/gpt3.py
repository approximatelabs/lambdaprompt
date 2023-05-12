from .prompt import PromptTemplate, AsyncPromptTemplate, ChatTemplate, AsyncChatTemplate
from .backends import OpenAICompletion, OpenAIChat
from functools import partial


class GPT3Prompt(PromptTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAICompletion())

class AsyncGPT3Prompt(AsyncPromptTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAICompletion())

class GPT3Chat(ChatTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIChat())

class AsyncGPT3Chat(AsyncChatTemplate):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIChat())