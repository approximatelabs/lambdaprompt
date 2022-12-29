import asyncio
import importlib
import inspect
import json
import sys
import traceback
import uuid
from collections import defaultdict
from functools import lru_cache
from typing import List

from databases import Database
from fastapi import BackgroundTasks, Depends, FastAPI
from pydantic import BaseSettings

import lambdaprompt

from .data import get_logs_for_task, log_event, setup_database

app = FastAPI()


class Settings(BaseSettings):
    # for now just making it work with sqlite...
    sqlite_path: str = "sqlite:///./history.db"


@lru_cache()
def get_settings():
    return Settings()


database = None


class CallWithId:
    def __init__(self, func, task_id):
        self.func = func
        self.task_id = task_id

    def __call__(self, *args, **kwargs):
        try:
            result = self.func(*args, **kwargs)
            if inspect.isawaitable(result):
                result = asyncio.run(result)
        except Exception as e:
            print("Error in task", self.task_id)
            traceback.print_tb(e.__traceback__)
        return result


def registration_as_async_callback(prompt: lambdaprompt.Prompt):
    def background_callable(background_tasks: BackgroundTasks, *args, **kwargs):
        taskuid = str(uuid.uuid4())
        background_tasks.add_task(CallWithId(prompt, taskuid), *args, **kwargs)
        return {"jobid": taskuid}

    background_callable.__signature__ = inspect.Signature(
        [
            inspect.Parameter(
                "background_tasks",
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=BackgroundTasks,
            ),
            *prompt.get_signature().parameters.values(),
        ]
    )
    background_callable.__name__ = "Background " + prompt.name

    print(f"Registering Async {prompt.name}")
    app.add_api_route(f"/async/{prompt.name}", background_callable)


def registration_as_prompt_callback(prompt: lambdaprompt.Prompt):
    def basiccall(*args, **kwargs):
        result = prompt(*args, **kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result

    basiccall.__signature__ = prompt.get_signature()
    basiccall.__name__ = prompt.name
    print("Registering Direct Callable", prompt.name)
    app.add_api_route(f"/prompt/{prompt.name}", basiccall)


lambdaprompt.register_creation_callback(registration_as_async_callback)
lambdaprompt.register_creation_callback(registration_as_prompt_callback)


async def log_call(*args):
    # get the task_id if it exists
    taskid = None
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                print(selfobj, selfobj.__class__.__name__)
                if isinstance(selfobj, CallWithId):
                    taskid = selfobj.task_id
    print("Logging call", taskid, args)
    await log_event(database, str(uuid.uuid4()), json.dumps(args), taskid)


lambdaprompt.register_call_callback(log_call)


@app.on_event("startup")
async def startup(settings: Settings = Depends(get_settings)):
    __import__("app.library")
    global database
    database = Database(settings.sqlite_path)
    await setup_database(database)


# @app.get("/add_gpt3_prompt")
# def add_gpt3_prompt(prompt: str, name: str = None):
#     prompt = lambdaprompt.GPT3Prompt(prompt, name=name)
#     return {"prompt_name": prompt.name}


@app.get("/list_prompts")
def list_prompts():
    url_list = [
        {"path": route.path, "name": route.name}
        for route in app.routes
        if route.path.startswith("/prompt/")
    ]
    return {"prompts": url_list}


@app.get("/background_status")
async def background_status(task_id: str):
    logs = await get_logs_for_task(database, task_id)
    return {"status": logs}


@app.get("/background_result")
def background_result(task_id: str):
    # if the task is done, return the result
    # if the task is not done, return "not done"
    # if the task is not found, return "not found"
    return {"result": "not done"}


@app.get("/")
def root():
    return {"message": "Hello World"}
