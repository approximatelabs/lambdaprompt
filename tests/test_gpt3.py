import asyncio

import pytest
from aioresponses import aioresponses

from lambdaprompt import AsyncGPT3Prompt, GPT3Prompt, prompt, register_call_callback


@pytest.fixture(autouse=True)
def ignore_load_dotenv(mocker):
    mocker.patch("lambdaprompt.config.load_dotenv", return_value=None)


def test_api_key_needed():
    with aioresponses() as m:
        m.post(
            "https://api.openai.com/v1/completions",
            payload={"choices": [{"text": "Wow!"}]},
        )
        prompt = GPT3Prompt("test {{ hello }}")
        with pytest.raises(Exception):
            prompt(hello="world")


def test_gpt3_prompt_returns(mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "TEST"})
    with aioresponses() as m:
        m.post(
            "https://api.openai.com/v1/completions",
            payload={"choices": [{"text": "Wow!"}]},
        )
        prompt = GPT3Prompt("test {{ hello }}")
        assert prompt(hello="world") == "Wow!"


@pytest.mark.asyncio
async def test_async_gpt3_prompt_returns(mocker):
    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "TEST"})
    with aioresponses() as m:
        m.post(
            "https://api.openai.com/v1/completions",
            payload={"choices": [{"text": "Wow!"}]},
        )
        prompt = AsyncGPT3Prompt("test2 {{ hello }}")
        assert await prompt("world") == "Wow!"


@pytest.mark.asyncio
async def test_async_gpt3_prompt_stacktrace(mocker):
    logs = []
    register_call_callback(lambda *args: logs.append(args))

    mocker.patch.dict("os.environ", {"OPENAI_API_KEY": "TEST"})
    with aioresponses() as m:
        m.post(
            "https://api.openai.com/v1/completions",
            payload={"choices": [{"text": "Wow!"}]},
        )
        test2 = AsyncGPT3Prompt("test2 {{ hello }}")

        @prompt
        async def outer(haha):
            return (await test2("yeah")) + haha

        assert await outer("meow") == "Wow!meow"

    # assume deterministic ordering of logs
    outer_id = logs[0][2]["exec_uuid"]
    inner_id = logs[1][2]["exec_uuid"]
    assert outer_id in logs[1][2]["callstack"]
    assert inner_id in logs[1][2]["callstack"]
