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
