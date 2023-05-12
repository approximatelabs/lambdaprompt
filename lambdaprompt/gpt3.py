from .prompt import Completion, AsyncCompletion, Chat, AsyncChat
from .backends import OpenAICompletion, OpenAIChat


class GPT3Prompt(Completion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAICompletion())

class AsyncGPT3Prompt(AsyncCompletion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAICompletion())

class GPT3Chat(Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIChat())

class AsyncGPT3Chat(AsyncChat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, backend=OpenAIChat())