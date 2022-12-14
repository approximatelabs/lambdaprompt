import asyncio
import inspect
import json
import time
import traceback
import uuid
from functools import partial
from hashlib import sha256

from jinja2 import Environment, meta
from unsync import unsync

CALLBACKS = []


def register_callback(callback):
    CALLBACKS.append(callback)


async def call_callbacks(*args):
    to_await = []
    for callback in CALLBACKS:
        if inspect.iscoroutinefunction(callback):
            to_await.append(callback(*args))
        else:
            callback(*args)
    await asyncio.gather(*to_await)


def get_uid_from_obj(obj):
    return sha256((json.dumps(obj, sort_keys=True)).encode("utf-8")).hexdigest()


def get_prompt_stack():
    # TODO: Figure out how to "see" through asyncio.gather and unync(...).result()
    promptstack = []
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                if isinstance(selfobj, Prompt):
                    exec_uuid = frame[0].f_locals.get("exec_uuid")
                    promptstack.append(exec_uuid)
    return promptstack


def get_exec_repr():
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                if isinstance(selfobj, Prompt):
                    return {
                        "prompt": selfobj.serialized_repr,
                        "args": frame[0].f_locals.get("args"),
                        "kwargs": frame[0].f_locals.get("kwargs"),
                        "exec_uuid": frame[0].f_locals.get("exec_uuid"),
                        "callstack": frame[0].f_locals.get("ps"),
                    }


class Prompt:
    def __init__(self, function, name=None):
        self.name = name or function.__name__
        self.function = function

    def execute(self, *args, **kwargs):
        if not isinstance(self, AsyncPrompt) and inspect.iscoroutinefunction(
            self.function
        ):
            return unsync(self.function)(*args, **kwargs).result()
        return self.function(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        exec_uuid = str(uuid.uuid4())
        ps = get_prompt_stack()
        exec_repr = get_exec_repr()
        st = time.time()
        unsync(call_callbacks)("enter", st, exec_repr, None, None).result()
        try:
            response = self.execute(*args, **kwargs)
        except Exception:
            response = f"{traceback.format_exc()}"
            raise
        finally:
            et = time.time()
            unsync(call_callbacks)("enter", st, exec_repr, response, et - st).result()
        return response

    @property
    def id(self):
        return get_uid_from_obj(self.code)

    @property
    def serialized_repr(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.__class__.__name__,
            "code": self.code,
        }

    @property
    def code(self):
        return inspect.getsource(self.function)


class AsyncPrompt(Prompt):
    async def __call__(self, *args, **kwargs):
        exec_uuid = str(uuid.uuid4())
        ps = get_prompt_stack()
        exec_repr = get_exec_repr()
        st = time.time()
        await call_callbacks("enter", st, exec_repr, None, None)
        try:
            response = await self.execute(*args, **kwargs)
        except Exception:
            response = f"{traceback.format_exc()}"
            raise
        finally:
            et = time.time()
            await call_callbacks("enter", st, exec_repr, response, et - st)
        return response


def prompt(f):
    if inspect.iscoroutinefunction(f):
        return AsyncPrompt(f)
    return Prompt(f)


env = Environment()


class PromptTemplate(Prompt):
    def __init__(self, prompt_template_string, name=None, **kwargs):
        self.prompt_template_string = prompt_template_string
        self.prompt_template = env.from_string(prompt_template_string)
        self.kwargs = kwargs

        async def function(*prompt_args, **prompt_kwargs):
            prompt = self.get_prompt(*prompt_args, **prompt_kwargs)
            return await self.call_on_template(prompt, **kwargs)

        super().__init__(function, name=name)

    def get_prompt(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(
                **{n: a for n, a in zip(self.get_named_args(), args)}
            )
        return self.prompt_template.render(**kwargs)

    @staticmethod
    async def call_on_template(prompt, **kwargs):
        raise NotImplementedError

    def get_named_args(self):
        return meta.find_undeclared_variables(env.parse(self.prompt_template_string))

    @property
    def code(self):
        return self.prompt_template_string

    @property
    def serialized_repr(self):
        return {
            "id": self.id,
            "name": self.name,
            "type": self.__class__.__name__,
            "code": self.code,
            "special_kwargs": self.kwargs,
        }
