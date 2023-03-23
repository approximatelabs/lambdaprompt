import asyncio
import json
import os

import aiohttp
import yaml
from jinja2 import Environment, meta

from .config import CONFIG
from .prompt import AsyncPrompt, Prompt, PromptTemplate, get_uid_from_obj

env = Environment()


def get_gpt3_completion_reqs(
    prompt, temperature=0.0, stop=None, model_name="text-davinci-003", max_tokens=500
):
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("No OpenAI API key found")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
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
    prompt, temperature=0.0, stop=None, model_name="text-davinci-003", max_tokens=500
):
    headers, data = get_gpt3_completion_reqs(prompt, temperature, stop, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        async with aiohttp.ClientSession(trust_env=True) as session:
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


# message = {'role': ('system', 'user', 'assistant'), 'content': 'text'}
def get_gpt3_chat_reqs(
    messages, temperature=0.0, stop=None, model_name="gpt-3.5-turbo", max_tokens=500
):
    if not os.environ.get("OPENAI_API_KEY"):
        raise Exception("No OpenAI API key found")
    headers = {
        "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY')}",
        "Content-Type": "application/json",
    }
    data = {
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "model": model_name,
        "presence_penalty": 0.2,
        "frequency_penalty": 0.2,
    }
    if stop:
        data["stop"] = stop
    return headers, data


def get_gpt3_chat_response_choice(answer):
    # assuming that we get an assistant response? 
    # TODO: validate Is it always that?
    if "choices" in answer:
        return answer["choices"][0]["message"]['content']
    else:
        print("Possible error: query returned:", answer)
    return ""


async def async_get_gpt3_chat_response(
    messages, temperature=0.0, stop=None, model_name="gpt-3.5-turbo", max_tokens=500
):
    headers, data = get_gpt3_chat_reqs(messages, temperature, stop, model_name)
    trying = 0
    while trying < 4:
        trying += 1
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.post(
                "https://api.openai.com/v1/chat/completions", headers=headers, json=data
            ) as resp:
                answer = await resp.json()
                if "choices" in answer:
                    return get_gpt3_chat_response_choice(answer)
                else:
                    if "Rate limit" in answer.get("error", {}).get("message", ""):
                        print(".", end="")
                        await asyncio.sleep(trying * 10)
                    else:
                        print(f"Not sure what happened: {answer}")
    return get_gpt3_chat_response_choice(answer)


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
        async with aiohttp.ClientSession(trust_env=True) as session:
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
        max_tokens=500
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


class GPT3Chat(PromptTemplate):
    def __init__(
        self,
        base_conversation=None,
        name=None,
        temperature=0.0,
        stop=None,
        model_name="gpt-3.5-turbo",
        max_tokens=500
    ):
        if isinstance(base_conversation, str):
            base_conversation = yaml.safe_load(base_conversation)

        base_conversation = [] if base_conversation is None else base_conversation
        assert isinstance(base_conversation, list)
        assert all([isinstance(x, dict) for x in base_conversation])
        # right now, assume its set up as {'system': 'hello {{user}}}', 'user': 'Hey there!', 'assistant': 'Hi!'}
        assert all([k in ['user', 'assistant', 'system'] for x in base_conversation for k in x.keys()])
        name = name or f"GPT3Chat_{get_uid_from_obj(base_conversation)}"

        self.kwargs = {'temperature': temperature, 'stop': stop, 'model_name': model_name}
        self.roles = [next(iter(template.keys())) for template in base_conversation]
        self.message_template_strings = [next(iter(template.values())) for template in base_conversation]
        self.message_templates = [env.from_string(x) for x in self.message_template_strings]
        
        async def function(user_input=None, **prompt_kwargs):
            messages = self.resolve_templated_conversation(user_input, **prompt_kwargs)
            return await async_get_gpt3_chat_response(
                messages, temperature=temperature, model_name=model_name, stop=stop
            )

        super(PromptTemplate, self).__init__(function, name=name)

    def resolve_templated_conversation(self, user_input=None, **prompt_kwargs):
        messages = []
        for i, template in enumerate(self.message_templates):
            messages.append({'role': self.roles[i], 'content': template.render(**prompt_kwargs)})
        if user_input is not None:
            messages.append({'role': 'user', 'content': user_input})
        return messages
    
    def add(self, user=None, assistant=None, system=None):
        # check that only 1 user, assistant or system is defined
        assert sum([user is not None, assistant is not None, system is not None]) == 1, "only define one"
        new_base_conversation = [{k: v} for k, v in zip(self.roles, self.message_template_strings)]
        if user is not None:
            new_base_conversation.append({'user': user})
        if assistant is not None:
            new_base_conversation.append({'assistant': assistant})
        if system is not None:
            new_base_conversation.append({'system': system})
        return self.__class__(new_base_conversation, **self.kwargs)

    def get_named_args(self):
        variables = []
        for template_string in self.message_template_strings:
            variables.extend(meta.find_undeclared_variables(env.parse(template_string)))
        return variables
    
    @property
    def code(self):
        return json.dumps([{k: v} for k, v in zip(self.roles, self.message_template_strings)])


class AsyncGPT3Chat(GPT3Chat, AsyncPrompt):
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
