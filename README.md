# λprompt - Turn prompts into functions

UNDER CONSTRUCTION 
![λprompt is under construction](under-construction.png)

`pip install lambdaprompt`

lambdaprompt is a python package, ...
* minimalistic API
* functional helpers
* create complex and emergent behavior

For using openAI, set up API keys as environment variables or set after importing (also easy to just make a `.env` file, since this uses `dotenv` package)

`OPENAI_API_KEY=...`

`import lambdaprompt; lambdaprompt.setup(openai_api_key=’...’)`

## Creating a prompt

Prompts use JINJA templating to create a string, the string is passed to the LLM for completion.

```python
from lambdaprompt import asyncGPT3Prompt

deeper_website_choice = asyncGPT3Prompt(
    "deeper_website_choice",
    """
For the question [{{ question }}], the search results are
{{ search_results }}
In order to answer the question, which three page indices (0-9 from above) should be further investigated? (eg. [2, 7, 9])
[""",
    stop="]",
)
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

## Using a prompt

```python
await deeper_website_choice(question="What is the capital of France?", search_results="0: result 0, 1: france, 2: another thing, ...")
```

## Some special properties

1. For prompts with only a single variable, can directly call with the variable as args (no need to define in kwarg)
```python
basic_qa = asyncGPT3Prompt("basic_qa", """What is the answer to the question [{{ question }}]?""")

await basic_qa("Is it safe to eat pizza with chopsticks?")
```

2. You can use functional primatives to create more complex prompts
```python
print(*map(basic_qa, ["Is it safe to eat pizza with chopsticks?", "What is the capital of France?"]))
```

3. You can apply these to pandas dataframes to do analytics quickly using LLMs
```python
import pandas as pd
from lambdaprompt import GPT3Prompt

df = pd.DataFrame({'country': ["France", "Japan", "USA"]})
df['capitals'] = df.apply(GPT3Prompt("basic_qa", """What is the capital of {{ country }}?"""), axis=1)
```

## Bonus

There is also a `GPT3Edit` class, that can be used to edit text. 

## Advanced usage
### Pre-and-post call hooks (tracing and logging) --> This is just on all the time right now... it makes a sqlitedb.
```
lambdaprompt.register(pre=print, post=print)
```
### Lightweight web-server for calling prompts (useful for attaching to JS webapps)
```bash
uvicorn lambdaprompt.promptapi:app --reload
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

