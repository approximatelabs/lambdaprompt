import asyncio
import hashlib
import inspect
import json
import time

# TODO: hook or logger stuff? just use regular logger and let someone add a database logger??
# TODO: Need to decide a better data model, prompt vs. prompt execution id, and task_id, etc.
# TODO: Need to decide how to handle async vs. sync prompts -- I think make everything async, then make the non async wrapper

# HOOKS = {
#     "before_prompt": [],
#     "after_prompt": [],
# }


def get_uid_from_args_kwargs(prompt, args, kwargs):
    strversion = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True)
    return hashlib.sha256(
        prompt.id.encode("utf-8") + strversion.encode("utf-8")
    ).hexdigest()


def get_prompt_stack_and_outer_id():
    promptstack = {"prompts": [], "task_id": None}
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                # we are in a class
                if isinstance(selfobj, Prompt):
                    (args,) = frame[0].f_locals.get("args")
                    kwargs = frame[0].f_locals.get("kwargs")
                    uid_from_args = get_uid_from_args_kwargs(selfobj, args, kwargs)
                    promptstack["prompts"].append(uid_from_args)
                elif getattr(selfobj, "task_id", None):
                    # special class that has "task_id" that is used for parent.
                    promptstack["task_id"] = selfobj.task_id
    return promptstack


class Prompt:
    def __init__(self, name, function=None):
        self.name = name
        self.function = function

    def execute(self, *args, **kwargs):
        if self.function is None:
            raise NotImplementedError("Must implement function")
        return self.function(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        promptstack = get_prompt_stack_and_outer_id()
        execution_id = get_uid_from_args_kwargs(self, args, kwargs)
        st = time.time()
        asyncio.run(
            record_prompt(
                database,
                execution_id,
                self.name,
                {"args": args, "kwargs": kwargs},
                None,
                None,
                promptstack,
            )
        )
        print("Entering prompt", self.name, promptstack)
        toRaise = None
        try:
            response = self.execute(*args, **kwargs)
        except Exception as e:
            response = f"ERROR\n{e}"
            toRaise = e
        print("Exiting prompt", self.name, promptstack)
        et = time.time()
        if PM_SETTINGS["VERBOSE"]:
            print_prompt_json(
                execution_id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
            )
        asyncio.run(
            record_prompt(
                database,
                execution_id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
                promptstack,
            )
        )
        if toRaise is not None:
            raise toRaise
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        return md5(inspect.getsource(self.function).encode("utf-8")).hexdigest()


class asyncPrompt(Prompt):
    async def __call__(self, *args, **kwargs):
        promptstack = get_prompt_stack_and_outer_id()
        execution_id = get_uid_from_args_kwargs(self, args, kwargs)
        await record_prompt(
            database,
            execution_id,
            self.name,
            {"args": args, "kwargs": kwargs},
            None,
            None,
            promptstack,
        )
        st = time.time()
        print("Entering prompt", self.name, promptstack)
        toRaise = None
        try:
            response = await self.execute(*args, **kwargs)
        except Exception as e:
            toRaise = e
            response = f"ERROR\n{e}"
        print("Exiting prompt", self.name, promptstack)
        et = time.time()
        if PM_SETTINGS["VERBOSE"]:
            print_prompt_json(
                execution_id,
                self.name,
                {"args": args, "kwargs": kwargs},
                response,
                et - st,
            )
        await record_prompt(
            database,
            execution_id,
            self.name,
            {"args": args, "kwargs": kwargs},
            response,
            et - st,
            promptstack,
        )
        if toRaise is not None:
            raise toRaise
        return response


class GPT3Prompt(Prompt):
    id_keys = ["prompt_template_string", "stop", "temperature", "model_name"]

    def __init__(
        self,
        name,
        prompt_template_string,
        temperature=0.0,
        stop=None,
        model_name="text-davinci-002",
    ):
        super().__init__(name)
        self.prompt_template_string = prompt_template_string
        self.prompt_template = env.from_string(prompt_template_string)
        self.stop = stop
        self.temperature = temperature
        self.model_name = model_name

    def get_named_args(self):
        return meta.find_undeclared_variables(env.parse(self.prompt_template_string))

    def get_prompt(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(
                **{n: a for n, a in zip(self.get_named_args(), args)}
            )
        return self.prompt_template.render(**kwargs)

    def execute(self, *args, **kwargs):
        prompt = self.get_prompt(*args, **kwargs)
        response = get_gpt3_response(
            prompt, self.temperature, self.stop, self.model_name
        )
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        reprstuff = " ".join([str(getattr(self, k)) for k in self.id_keys])
        return md5(reprstuff.encode("utf-8")).hexdigest()


class asyncGPT3Prompt(asyncPrompt, GPT3Prompt):
    async def execute(self, *args, **kwargs):
        prompt = self.get_prompt(*args, **kwargs)
        response = await async_get_gpt3_response(
            prompt, self.temperature, self.stop, self.model_name
        )
        return response


class GPT3Edit(Prompt):
    id_keys = [
        "instruction_template_string",
        "temperature",
        "model_name",
    ]

    def __init__(
        self,
        name,
        instruction_template_string,
        temperature=0.0,
        model_name="code-davinci-edit-001",
    ):
        super().__init__(name)
        self.instruction_template_string = instruction_template_string
        self.instruction_template = env.from_string(instruction_template_string)
        self.temperature = temperature
        self.model_name = model_name

    def get_named_args(self):
        return meta.find_undeclared_variables(
            env.parse(self.instruction_template_string)
        )

    def get_instruction(self, *args, **kwargs):
        if len(args) > 0:
            # also consider mixing kwargs and args
            # also consider partial parsing with kwargs first, then applying remaining named args
            return self.get_prompt(
                **{n: a for n, a in zip(self.get_named_args(), args)}
            )
        return self.instruction_template.render(**kwargs)

    def get_input(self, *args, **kwargs):
        return kwargs.get("input", "")

    def execute(self, *args, **kwargs):
        input = self.get_input(*args, **kwargs)
        instruction = self.get_instruction(*args, **kwargs)
        response = get_gpt3_edit_response(
            instruction,
            input=input,
            temperature=self.temperature,
            model_name=self.model_name,
        )
        return response

    @property
    def id(self):
        # grab the code from execute method and hash it
        reprstuff = " ".join([str(getattr(self, k)) for k in self.id_keys])
        return md5(reprstuff.encode("utf-8")).hexdigest()


class asyncGPT3Edit(asyncPrompt, GPT3Edit):
    async def execute(self, *args, **kwargs):
        input = self.get_input(*args, **kwargs)
        instruction = self.get_instruction(*args, **kwargs)
        response = await async_get_gpt3_edit_response(
            instruction,
            input=input,
            temperature=self.temperature,
            model_name=self.model_name,
        )
        return response


def prompt(f):
    # check if f is async or not
    if inspect.iscoroutinefunction(f):
        return asyncPrompt(f.__name__, f)
    return Prompt(f.__name__, f)
