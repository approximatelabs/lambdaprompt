import asyncio

from lambdaprompt import prompt


@prompt
def greet_name(name):
    return f"Hello {name}"


@prompt
async def delayed_greet(name: str, sleeptime: float):
    await asyncio.sleep(sleeptime)
    return greet_name(name)
