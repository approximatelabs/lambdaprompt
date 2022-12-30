import asyncio
import warnings

import pytest

import lambdaprompt
from lambdaprompt import AsyncPrompt, Prompt, prompt, register_call_callback


def test_prompt_decorator():
    @prompt
    def test_prompt():
        return "test"

    assert isinstance(test_prompt, Prompt)
    assert test_prompt() == "test"


@pytest.mark.asyncio
async def test_async_prompt_decorator():
    @prompt
    async def test_prompt():
        return "test"

    assert isinstance(test_prompt, AsyncPrompt)
    assert await test_prompt() == "test"


def test_prompt_error_handler():
    @prompt
    def test_prompt():
        raise Exception("wow, failure")

    with pytest.raises(Exception):
        test_prompt()


def test_run_asyncfunction_in_sync_prompt():
    async def test_prompt():
        return "test"

    prompt_with_async = Prompt(test_prompt)
    assert isinstance(prompt_with_async, Prompt)
    assert prompt_with_async() == "test"


def test_callstack_takes_sync_and_async():
    log = []
    register_call_callback(lambda *args: log.append(args))

    asynclog = []

    async def async_log_callback(*logs):
        asynclog.append(logs)

    register_call_callback(async_log_callback)

    @prompt
    def inner():
        return "inner"

    @prompt
    def outer():
        return inner()

    assert outer() == "inner"
    assert len(log) == 4
    assert len(asynclog) == 4
    assert outer() == "inner"
    assert len(log) == 8
    assert len(asynclog) == 8


@pytest.mark.asyncio
async def test_callstack_takes_sync_and_async_in_async():
    log = []
    register_call_callback(lambda *args: log.append(args))

    asynclog = []

    async def async_log_callback(*logs):
        asynclog.append(logs)

    register_call_callback(async_log_callback)

    @prompt
    async def inner():
        return "inner"

    @prompt
    async def outer():
        return await inner()

    assert await outer() == "inner"
    assert len(log) == 4
    assert len(asynclog) == 4
    assert await outer() == "inner"
    assert len(log) == 8
    assert len(asynclog) == 8
    assert any([len(l[2]["callstack"]) == 2 for l in log])


def test_async_function_in_sync_prompt_has_callstack():
    log = []
    register_call_callback(lambda *args: log.append(args))

    @prompt
    async def inner():
        return "inner"

    async def outer():
        return await inner() + "!"

    pro = Prompt(outer)

    assert pro() == "inner!"
    assert len(log) == 4
    if not any([len(l[2]["callstack"]) == 2 for l in log]):
        warnings.warn(Warning("Callstack not 2 deep in async function in sync prompt"))


@pytest.mark.asyncio
async def test_callstack_works_with_gather():
    log = []
    register_call_callback(lambda *args: log.append(args))

    @prompt
    async def inner(x):
        return str(x) + "!"

    @prompt
    async def outer():
        return await asyncio.gather(inner(1), inner(2), inner(3))

    assert await outer() == ["1!", "2!", "3!"]
    assert len(log) == 8
    if any([len(l[2]["callstack"]) == 2 for l in log]):
        warnings.warn(Warning("Callstack not 2 deep when using gather"))
