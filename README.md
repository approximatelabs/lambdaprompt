# λprompt - Functional programming interface for building AI systems

lambdaprompt is a python package, ...

## Getting started

Install `pip install lambdaprompt`

For using openAI, set up API keys as environment variables or set after importing
`OPENAI_API_KEY=...`
`import lambdaprompt; lambdaprompt.setup(openai_api_key=’...’)`

Try it out in colab [link]

## Library demos and examples
[[ See Here ]]

## How to

### Map
```
prompt.map([“yes”, “no”])
```

### Reduce
```
prompt.reduce(...)
```

### Composition
```
…
```

### For Pandas Users (Useful for data processing)
```
df.prompt.apply(...)
```

## Making Prompts

### LLM JINJA templates
```
prompt = LLM(“goal of prompt”, “””
{{ template }}
… 
“””
```

### Decorator
```
@promptify
def excalamation(arg):
    return arg+"!"*10
```

## Advanced usage
### Pre-and-post call hooks (tracing and logging) [see example]
```
lambdaprompt.register(pre=print, post=print)
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


To add:
An issue template

To add: 
A pull request template

TODO:
Check all dependent prompts in the library via signature-check are correct
This ensures that when someone changes an upstream prompt, they must at least see all dependent prompts that they should update.

