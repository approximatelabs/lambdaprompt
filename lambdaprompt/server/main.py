import asyncio
import importlib
import inspect
import json
import logging
import os
import sys
import traceback
import uuid
from functools import lru_cache
from typing import List

import aiosqlite
from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic_settings import BaseSettings

import lambdaprompt

from .data import add_prompt_if_not_exists, get_logs_for_task, log_event, setup_database

app = FastAPI()


class Settings(BaseSettings):
    # for now just making it work with sqlite...
    sqlite_path: str = "history.db"
    prompt_library_paths: List[str] = []


@lru_cache()
def get_settings():
    return Settings()


database = None


class CallWithId:
    def __init__(self, func, task_id):
        self.func = func
        self.task_id = task_id

    def __call__(self, *args, **kwargs):
        result = None
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
    logging.info(f"Registering Async {prompt.name}")
    app.add_api_route(f"/async/{prompt.name}", background_callable)


def registration_as_prompt_callback(prompt: lambdaprompt.Prompt):
    def basiccall(*args, **kwargs):
        result = prompt(*args, **kwargs)
        if inspect.isawaitable(result):
            result = asyncio.run(result)
        return result

    basiccall.__signature__ = prompt.get_signature()
    basiccall.__name__ = prompt.name
    logging.info("Registering Direct Callable", prompt.name)
    app.add_api_route(f"/prompt/{prompt.name}", basiccall)


async def log_call(enter_or_exit, st, exec_repr, result, duration):
    prompt = exec_repr.pop("prompt")
    await add_prompt_if_not_exists(database, prompt["id"], json.dumps(prompt))
    new_prompt = {k: v for k, v in prompt.items() if k in ["id", "name", "type"]}
    exec_repr["prompt"] = new_prompt
    taskid = None
    for frame in inspect.stack():
        if frame.function == "__call__":
            selfobj = frame[0].f_locals.get("self")
            if selfobj is not None:
                if isinstance(selfobj, CallWithId):
                    taskid = selfobj.task_id
    await log_event(
        database,
        str(uuid.uuid4()),
        json.dumps((enter_or_exit, st, exec_repr, result, duration)),
        taskid,
    )


def import_modules(paths):
    for directory in paths:
        sys.path.append(directory)
        for file in [f for f in os.listdir(directory) if f.endswith(".py")]:
            module_name = file[:-3]
            importlib.import_module(module_name, package=directory)


lambdaprompt.register_creation_callback(registration_as_async_callback)
lambdaprompt.register_creation_callback(registration_as_prompt_callback)
lambdaprompt.register_call_callback(log_call)


@app.on_event("startup")
async def startup():
    settings = get_settings()

    global database
    database = await aiosqlite.connect(settings.sqlite_path)
    await setup_database(database)

    import_modules(settings.prompt_library_paths)


@app.on_event("shutdown")
async def shutdown():
    await database.close()


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


@app.get("/background_task_trace")
async def background_task_trace(jobid: str):
    logs = await get_logs_for_task(database, jobid)
    return {"trace": logs}


@app.get("/background_result")
async def background_result(jobid: str):
    logs = [json.loads(l) for l in await get_logs_for_task(database, jobid)]
    if len(logs) == 0:
        raise HTTPException(status_code=422, detail="Job not found")
    parent_task, *_ = [
        log for log in logs if log[0] == "enter" and len(log[2]["callstack"]) == 1
    ]
    exec_uuid = parent_task[2]["exec_uuid"]
    possible_final = [
        log for log in logs if log[0] == "exit" and log[2]["exec_uuid"] == exec_uuid
    ]
    if len(possible_final) == 0:
        return {"status": "running"}
    (final,) = possible_final
    return {"status": "complete", "result": final[3], "duration": final[4]}
