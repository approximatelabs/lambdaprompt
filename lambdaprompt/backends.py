import os
import aiohttp
from pydantic import BaseModel, Extra
from typing import Optional
import tenacity


class Backend:
    class Parameters(BaseModel):
        class Config:
            extra = Extra.forbid

    def __init__(self, **param_override):
        self.param_override = self.Parameters(**param_override)
    
    def parse_param(self, **kwargs):
        return self.Parameters(**{**self.param_override.dict(), **kwargs}).dict()


class RateLimitError(Exception):
    pass


class RequestBackend(Backend):
    def __init__(self, endpoint_url, **param_override):
        self.endpoint_url = endpoint_url
        super().__init__(**param_override)
    
    def headers(self, *args, **kwargs):
        raise NotImplementedError("Must implement headers")

    def body(self, *args, **kwargs):
        raise NotImplementedError("Must implement body")

    def parse_response(self, result):
        raise NotImplementedError("Must implement result_parser")

    @tenacity.retry(
            wait=tenacity.wait_exponential(multiplier=1, min=4, max=10),
            stop=tenacity.stop_after_attempt(4),
            retry=tenacity.retry_if_exception_type(RateLimitError),
            reraise=True
            )
    async def __call__(self, *args, **kwargs):
        headers = self.headers(*args, **kwargs)
        data = self.body(*args, **kwargs)
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                self.endpoint_url, headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                result = self.parse_response(answer)
        return result


class OpenAICompletion(RequestBackend):
    class Parameters(RequestBackend.Parameters):
        max_tokens: int = 500
        temperature: float = 0.0
        model: str = 'text-davinci-003'
        presence_penalty: float = 0.2
        frequency_penalty: float = 0.2
        stop: Optional[str]
    
    def __init__(self, openai_api_key=None, **param_override):
        self.openai_api_key = openai_api_key or os.environ.get("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise Exception("No OpenAI API key found (envvar OPENAI_API_KEY))")
        super().__init__("https://api.openai.com/v1/completions", **param_override)

    def headers(self, *args, **kwargs):
        return {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
            "Content-Type": "application/json",
        }

    def body(self, prompt, **kwargs):
        data = {
            "prompt": prompt,
            **self.parse_param(**kwargs)
        }
        if stop := data.pop('stop'):
            data["stop"] = stop
        return data

    def parse_response(self, answer):
        if "error" in answer:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                raise RateLimitError()
            else:
                raise Exception(f"Not sure what happened: {answer}")
        return answer["choices"][0]["text"]


class OpenAIChat(OpenAICompletion):
    class Parameters(OpenAICompletion.Parameters):
        model: str = 'gpt-3.5-turbo'
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.endpoint_url = "https://api.openai.com/v1/chat/completions"

    def body(self, messages, **kwargs):
        data = {
            "messages": messages,
            **self.parse_param(**kwargs)
        }
        if stop := data.pop('stop'):
            data["stop"] = stop
        return data

    def parse_response(self, answer):
        if "error" in answer:
            if "Rate limit" in answer.get("error", {}).get("message", ""):
                raise RateLimitError()
            else:
                raise Exception(f"Not sure what happened: {answer}")
        return answer["choices"][0]["message"]['content']


class HuggingFaceBackend(Backend):
    pass
