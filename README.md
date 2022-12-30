# λprompt - Turn prompts into functions

`pip install lambdaprompt`

lambdaprompt is a python package, ...
* minimalistic API
* functional helpers
* create complex and emergent behavior
* use as a webserver, easily host prompts as HTTP endpoints

(For use as a server, see the example: https://github.com/approximatelabs/example-lambdaprompt-server )

For using openAI, set up API keys as environment variables or set after importing (also easy to just make a `.env` file, since this uses `dotenv` package)

`OPENAI_API_KEY=...`

## Creating a prompt

Prompts use JINJA templating to create a string, the string is passed to the LLM for completion.

```python
from lambdaprompt import GPT3Prompt

first = GPT3Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
# then use it as a function
first(number=12, thing="apples")
```

You can also turn any function into a prompt (useful for composing prompts, or creating programs out of prompts.
```python
from lambdaprompt import prompt

@prompt
def standard_function(text_input):
    res = is_a_question(text_input)
    if res.lower().startswith('yes'):
        return answer_the_question(text_input)
    else:
        return "That was not a question, please try again"
```

## Using a prompt -- just call like a function

```python
first(number=12, thing="apples")
```

### some examples
```python
>>> from lambdaprompt.gpt3 import GPT3Prompt, GPT3Edit, AsyncGPT3Edit, AsyncGPT3Prompt
>>> first = GPT3Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
>>> first(number=12, thing="apples")
' 8 of the apples.\n\nSally now has 4 apples.'
```

```python
wow = AsyncGPT3Edit("Turn this into a {{ joke_style }} joke")
await wow(joke_style="american western", input="Sally ate a lot of food")
```
```
'Sally ate a lot of food.\nShe was a cowgirl.\n'
```

### Some special properties

1. For prompts with only a single variable, can directly call with the variable as args (no need to define in kwarg)
```python
basic_qa = asyncGPT3Prompt("basic_qa", """What is the answer to the question [{{ question }}]?""")

await basic_qa("Is it safe to eat pizza with chopsticks?")
```

2. You can use functional primatives to create more complex prompts
```python
print(*map(basic_qa, ["Is it safe to eat pizza with chopsticks?", "What is the capital of France?"]))
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

### IN PROGRESS (I think this doesn't work as expected yet...)
3. You can apply these to pandas dataframes to do analytics quickly using LLMs
```python
import pandas as pd
from lambdaprompt import GPT3Prompt

df = pd.DataFrame({'country': ["France", "Japan", "USA"]})
df['capitals'] = df.apply(GPT3Prompt("""What is the capital of {{ country }}?"""), axis=1)
```


## Advanced usage
### Pre-and-post call hooks (tracing and logging)
```
lambdaprompt.register_callback(lambda *x: print(x))
```


## Running `lambdaprompt` webserver (example, for dev)

Built on fastapi, includes a simple dockerfile here too
```
docker build -t lambdaprompt:latest . --build-arg mode=dev
docker run --it -v $(pwd):/code -p 4412:80 lambdaprompt:latest
```

For prod build
```
docker build -t lambdaprompt:latest .
```

## Design Patterns
- Response Optimization
  - [Ideation, Scoring and Selection](link)
  - [Error Correcting Language Loops](link)
- Summarization and Aggregations
  - [Rolling](link)
  - [k-tree](link)
- [Meta-Prompting](link)


## Contributions are welcome 
[Contributing](contributing.md)

