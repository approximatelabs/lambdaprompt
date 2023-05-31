import asyncio

import pytest
import json
from aioresponses import aioresponses
from openai.openai_object import OpenAIObject

from lambdaprompt import AsyncGPT4Prompt, GPT4Prompt, prompt, register_call_callback

gpt4_response = {
    "id": "chatcmpl-7MGn0BnpSCdy0FmBGf4bR4Vv3DJ0g",
    "object": "chat.completion",
    "created": 1685542110,
    "model": "gpt-4",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "message": {
                "role": "assistant",
                "content": "As an AI, I am programmed to adhere to ethical guidelines and provide accurate, reliable information. Creating fake data goes against these principles. If you need help finding or generating information for a specific purpose, please let me know and I would be happy to assist you.",
            },
        }
    ],
    "usage": {"completion_tokens": 53, "prompt_tokens": 26, "total_tokens": 79},
}


def test_api_key_needed():
    with aioresponses() as m:
        m.post(
            "https://api.openai.com/v1/completions",
            payload={"choices": [{"text": "Wow!"}]},
        )
        with pytest.raises(Exception):
            prompt = GPT4Prompt("test {{ hello }}")
            prompt(hello="world")


# def test_gpt4_prompt_returns(mocker):
#    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "TEST"})
#    openai_object = OpenAIObject()
#
#    mocker.patch.object(openai_object, "to_dict_recursive", return_value=gpt4_response)
#
#    class CustomEncoder(json.JSONEncoder):
#        def default(self, obj):
#            if isinstance(obj, OpenAIObject):
#                # Convert OpenAIObjet to a JSON serializable dictionary
#                return {"key": obj.key, "value": obj.to_dict_recursive()}
#            return super().default(obj)
#
#    payload = CustomEncoder().encode(openai_object)
#    with aioresponses() as m:
#        m.post(
#            "https://api.openai.com/v1/completions",
#            payload=payload,
#        )
#        prompt = GPT4Prompt("test {{ hello }}")
#        assert prompt(hello="world") == "Wow GPT4!"
#
#    # aa = openai_object.to_dict_recursive()
#    # prompt = GPT4Prompt("test {{ hello }}")
#    # assert prompt(hello="world") == "Wow GPT4!"
