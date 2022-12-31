# λprompt - The easiest way to call and compose LLM prompts!

Write LLM prompts with jinja templates, compose them in python as functions, and call them directly or use them as a webservice!

`pip install lambdaprompt`

(For use as a server, see the example: https://github.com/approximatelabs/example-lambdaprompt-server )

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


## Design Patterns (TODO)
- Response Optimization
  - [Ideation, Scoring and Selection](link)
  - [Error Correcting Language Loops](link)
- Summarization and Aggregations
  - [Rolling](link)
  - [Fan-out-tree](link)
- [Meta-Prompting](link)
