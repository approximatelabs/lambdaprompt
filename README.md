[![](https://dcbadge.vercel.app/api/server/kW9nBQErGe?compact=true&style=flat)](https://discord.gg/kW9nBQErGe)

# λprompt - Build, compose and call templated LLM prompts!

Write LLM prompts with jinja templates, compose them in python as functions, and call them directly or use them as a webservice!

We believe that large language model prompts are a lot like "functions" in a programming sense and would benefit greatly by the power of an interpreted language. lambdaprompt is a library to offer an interface to back that belief up. This library allows for building full large language model based "prompt machines", including ones that self-edit to correct and even self-write their own execution code. 

`pip install lambdaprompt`

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/bluecoconut/bc5925d0de83b478852f5457ef8060ad/example-prompt.ipynb)

[A webserver (built on `FastAPI`) example repository](https://github.com/approximatelabs/example-lambdaprompt-server)

## Environment variables for using hosted models

For using openAI, set up API keys as environment variables or set after importing (also easy to just make a `.env` file, since this uses `dotenv` package)

`OPENAI_API_KEY=...`

## Creating a prompt

Prompts use JINJA templating to create a string, the string is passed to the LLM for completion.

```python
from lambdaprompt import GPT3Prompt

example = GPT3Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
# then use it as a function
example(number=12, thing="apples")
```


## Creating ChatGPT3 Conversational prompts

Each prompt can be thought of as a parameterizable conversation, and executing the prompt with an input will apply that as "the next line of conversation" and then generate the response. 

In order to update the memory state of the prompt, call the `.add()` method on the prompt, which can be used to add steps to a conversation and make the prompt "remember" what has been said.

```python
>>> import lambdaprompt as lp

>>> convo = lp.AsyncGPT3Chat([{'system': 'You are a {{ type_of_bot }}'}])
>>> await convo("What should we get for lunch?", type_of_bot="pirate")
As a pirate, I would suggest we have some hearty seafood such as fish and chips or a seafood platter. We could also have some rum to wash it down! Arrr!
```
## General prompt creation

You can also turn any function into a prompt (useful for composing prompts, or creating programs out of prompts. This is commonly called "prompt chaining". See how you can achieve this with simple python composition.
```python
from lambdaprompt import prompt, GPT3Prompt

generate_n_tasks = GPT3Prompt("Today I will do {{ n }} things (comma separated) [", stop="]")
is_happy = GPT3Prompt("The task {{ task_detail }} is a task that will make me happy? (y/n):")

@prompt
def get_tasks_and_rate_is_happy(n=3):
    results = []
    for task in generate_n_tasks(n=n).split(","):
        results.append((task, is_happy(task)))
    return results

print(get_tasks_and_rate_is_happy())
```

## Async and Sync

Lambdaprompt works on both sync and async functions, and offers a sync and async templated prompt interface

```python
from lambdaprompt import GPT3Prompt, asyncGPT3Prompt

#sync
first = GPT3Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
first(number=12, thing="apples")

#async
first = asyncGPT3Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
await first(number=12, thing="apples")
```

```python
from lambdaprompt import prompt

@prompt
def sync_example(a):
    return a + "!"

sync_example("hello")

@prompt
async def async_example(a):
    return a + "!"

await async_example("hello")
```

### Some special properties

For templated prompts with only template variable, can directly call with the variable as positional argument (no need to define in kwarg)
```python
basic_qa = asyncGPT3Prompt("basic_qa", """What is the answer to the question [{{ question }}]?""")

await basic_qa("Is it safe to eat pizza with chopsticks?")
```


## Using lambdaprompt as a webservice
Simply `pip install lambdaprompt[server]` and then add `from lambdaprompt.server.main import app` to the top of your file!

make a file

`app.py`
````python
from lambdaprompt import AsyncGPT3Prompt, prompt
from lambdaprompt.server.main import app

AsyncGPT3Prompt(
    """Rewrite the following as a {{ target_author }}. 
```
{{ source_text }}
```
Output:
```
""",
    name="rewrite_as",
    stop="```",
)
````

Then run
```
uvicorn app:app --reload
```

browse to `http://localhost:8000/docs` to see the swagger docs generated for the prompts!

## Running inside docker

First, create an .env file with your OpenAI API key: (like `OPENAI_API_KEY=sk-dskj32094klsaj9024lkjsa`)

```
docker build . -t lambdaprompt:0.0.1
docker run -it --env-file .env lambdaprompt:0.0.1  bash -c "python two.py"
```

This will output something like this:

```
docker run -it --env-file .env lambdaprompt:0.0.1  bash -c "python two.py"
[('example: go for a walk', '\n\nYes. Going for a walk can be a great way to boost your mood and get some fresh air.'), (' read a book', '\n\nYes'), (' call a friend', '\n\nYes')]

docker run -it --env-file .env lambdaprompt:0.0.1  bash -c "python two.py"
[(' edit ', '\n\nNo. Editing can be a tedious and time-consuming task, so it may not necessarily make you happy.')]
```


## Design Patterns (TODO)
- Response Optimization
  - [Ideation, Scoring and Selection](link)
  - [Error Correcting Language Loops](link)
- Summarization and Aggregations
  - [Rolling](link)
  - [Fan-out-tree](link)
- [Meta-Prompting](link)
