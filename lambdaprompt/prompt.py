import asyncio
import inspect
import json
import time
import traceback
import uuid
import yaml

from hashlib import sha256

from jinja2 import Environment, meta

from .backends import get_backend

CALL_CALLBACKS = []
CREATION_CALLBACKS = []


def resolve(obj):
    if inspect.isawaitable(obj):
        obj = asyncio.run(obj)
    return obj


def register_call_callback(callback):
    if callback not in CALL_CALLBACKS:
        CALL_CALLBACKS.append(callback)


def register_creation_callback(callback):
    if callback not in CREATION_CALLBACKS:
        CREATION_CALLBACKS.append(callback)


async def call_callbacks(*args):
    to_await = []
    for callback in CALL_CALLBACKS:
        if inspect.iscoroutinefunction(callback):
            to_await.append(callback(*args))
        else:
            callback(*args)
    await asyncio.gather(*to_await)


async def creation_callbacks(*args):
    to_await = []
    for callback in CREATION_CALLBACKS:
        if inspect.iscoroutinefunction(callback):
            to_await.append(callback(*args))
        else:
            callback(*args)
    await asyncio.gather(*to_await)


def get_uid_from_obj(obj):
    return sha256((json.dumps(obj, sort_keys=True)).encode("utf-8")).hexdigest()


def get_prompt_stack():
    promptstack = []
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                if isinstance(selfobj, Prompt):
                    exec_uuid = frame[0].f_locals.get("exec_uuid")
                    if exec_uuid and exec_uuid not in promptstack:
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
        # We have now made a prompt, so we should call the creation callbacks
        resolve(creation_callbacks(self))

    def execute(self, *args, **kwargs):
        if not isinstance(self, AsyncPrompt) and inspect.iscoroutinefunction(
            self.function
        ):
            return resolve(self.function(*args, **kwargs))
        return self.function(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        exec_uuid = str(uuid.uuid4())
        ps = get_prompt_stack()
        exec_repr = get_exec_repr()
        st = time.time()
        resolve(call_callbacks("enter", st, exec_repr, None, None))
        response = None
        try:
            response = resolve(self.execute(*args, **kwargs))
        except Exception:
            response = f"{traceback.format_exc()}"
            raise
        finally:
            et = time.time()
            resolve(call_callbacks("exit", st, exec_repr, response, et - st))
        return response

    def get_signature(self):
        return inspect.signature(self.function)

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
        resolve(call_callbacks("enter", st, exec_repr, None, None))
        response = None
        try:
            response = await self.execute(*args, **kwargs)
        except Exception:
            response = f"{traceback.format_exc()}"
            raise
        finally:
            et = time.time()
            resolve(call_callbacks("exit", st, exec_repr, response, et - st))
        return response


def prompt(f):
    if inspect.iscoroutinefunction(f):
        return AsyncPrompt(f)
    return Prompt(f)


env = Environment()


class Completion(Prompt):
    def __init__(self, prompt_template_string, name=None, backend=None, **kwargs):
        self.prompt_template_string = prompt_template_string
        self.prompt_template = env.from_string(prompt_template_string)
        self.kwargs = kwargs
        self.backend = backend

        async def function(*prompt_args, **prompt_kwargs):
            prompt = self.get_prompt(*prompt_args, **prompt_kwargs)
            backend = self.backend or get_backend('completion')
            return await backend(prompt, **self.kwargs)

        super().__init__(function, name=name)

    def get_prompt(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(
                **{n: a for n, a in zip(self.get_named_args(), args)}
            )
        return self.prompt_template.render(**kwargs)

    def get_named_args(self):
        return meta.find_undeclared_variables(env.parse(self.prompt_template_string))

    def get_signature(self):
        # create a function signature that defines kwargs based on `get_named_args`
        return inspect.Signature(
            [
                inspect.Parameter(
                    name,
                    inspect.Parameter.KEYWORD_ONLY,
                    default=inspect.Parameter.empty,
                    annotation=str,
                )
                for name in self.get_named_args()
            ]
        )

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

class AsyncCompletion(Completion, AsyncPrompt):
    pass

class Chat(Completion):
    def __init__(
        self,
        base_conversation=None,
        name=None,
        backend=None,
        **kwargs
    ):
        if isinstance(base_conversation, str):
            base_conversation = yaml.safe_load(base_conversation)

        base_conversation = [] if base_conversation is None else base_conversation
        assert isinstance(base_conversation, list)
        assert all([isinstance(x, dict) for x in base_conversation])
        # right now, assume its set up as {'system': 'hello {{user}}}', 'user': 'Hey there!', 'assistant': 'Hi!'}
        assert all([k in ['user', 'assistant', 'system'] for x in base_conversation for k in x.keys()])
        name = name or f"Chat_{get_uid_from_obj(base_conversation)}"

        self.kwargs = kwargs
        self.roles = [next(iter(template.keys())) for template in base_conversation]
        self.message_template_strings = [next(iter(template.values())) for template in base_conversation]
        self.message_templates = [env.from_string(x) for x in self.message_template_strings]
        self.backend = backend

        async def function(user_input=None, **prompt_kwargs):
            messages = self.resolve_templated_conversation(user_input, **prompt_kwargs)
            backend = self.backend or get_backend('chat')
            return await backend(messages, **self.kwargs)

        super(Completion, self).__init__(function, name=name)

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

    @staticmethod
    async def call_on_messages(messages, **kwargs):
        raise NotImplementedError

    @property
    def code(self):
        return json.dumps([{k: v} for k, v in zip(self.roles, self.message_template_strings)])

class AsyncChat(Chat, AsyncPrompt):
    pass