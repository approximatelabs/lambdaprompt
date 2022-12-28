import asyncio
import os

import aiohttp
from jinja2 import Environment, meta

from .config import CONFIG
from .prompt import AsyncPrompt, PromptTemplate, get_uid_from_obj

env = Environment()


def get_gpt3_completion_reqs(
    prompt, temperature=0.0, stop=None, model_name="text-davinci-003"
):
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("No OpenAI API key found")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "max_tokens": 500,
        "temperature": temperature,
        "model": model_name,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
    }
    if stop:
        data["stop"] = stop
    return headers, data


def get_gpt3_response_choice(answer):
    if "choices" in answer:
        return answer["choices"][0]["text"]
    else:
        print("Possible error: query returned:", answer)
    return answer.get("choices", [{"text": ""}])[0]["text"]


async def async_get_gpt3_response(
    prompt, temperature=0.0, stop=None, model_name="text-davinci-003"
):
    headers, data = get_gpt3_completion_reqs(prompt, temperature, stop, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/completions", headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                if "choices" in answer:
                    return get_gpt3_response_choice(answer)
                else:
                    if "Rate limit" in answer.get("error", {}).get("message", ""):
                        print(".", end="")
                        await asyncio.sleep(trying * 10)
                    else:
                        print(f"Not sure what happened: {answer}")
    return get_gpt3_response_choice(answer)


def get_gpt3_edit_reqs(
    instruction, input="", temperature=0.0, model_name="text-davinci-edit-001"
):
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("No OpenAI API key found")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "input": input,
        "instruction": instruction,
        "temperature": temperature,
        "model": model_name,
    }
    return headers, data


async def async_get_gpt3_edit_response(
    instruction, input="", temperature=0.0, model_name="text-davinci-edit-001"
):
    headers, data = get_gpt3_edit_reqs(instruction, input, temperature, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        async with aiohttp.ClientSession() as session:
            async with session.post(
                "https://api.openai.com/v1/edits", headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                if "choices" in answer:
                    return get_gpt3_response_choice(answer)
                else:
                    if "Rate limit" in answer.get("error", {}).get("message", ""):
                        print(".", end="")
                        await asyncio.sleep(trying * 10)
                    else:
                        print(f"Not sure what happened: {answer}")
    return get_gpt3_response_choice(answer)


class GPT3Prompt(PromptTemplate):
    def __init__(
        self,
        template_string,
        name=None,
        temperature=0.0,
        stop=None,
        model_name="text-davinci-003",
    ):
        name = name or f"GPT3_{get_uid_from_obj(template_string)}"
        super().__init__(
            template_string,
            name=name,
            temperature=temperature,
            stop=stop,
            model_name=model_name,
        )

    @staticmethod
    async def call_on_template(prompt, **kwargs):
        return await async_get_gpt3_response(prompt, **kwargs)


class AsyncGPT3Prompt(GPT3Prompt, AsyncPrompt):
    pass


class GPT3Edit(PromptTemplate):
    def __init__(
        self,
        instruction_template_string,
        name=None,
        temperature=0.0,
        model_name="text-davinci-edit-001",
    ):
        name = name or f"GPT3Edit_{get_uid_from_obj(instruction_template_string)}"
        super().__init__(
            instruction_template_string,
            name=name,
            temperature=temperature,
            model_name=model_name,
        )

        async def function(*prompt_args, **prompt_kwargs):
            input = prompt_kwargs.get("input")
            prompt = self.get_prompt(*prompt_args, **prompt_kwargs)
            return await self.call_on_template(
                prompt, input=input, temperature=temperature, model_name=model_name
            )

        self.function = function

    @staticmethod
    async def call_on_template(prompt, **kwargs):
        return await async_get_gpt3_edit_response(prompt, **kwargs)


class AsyncGPT3Edit(GPT3Edit, AsyncPrompt):
    pass
