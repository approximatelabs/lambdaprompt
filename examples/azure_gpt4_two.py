import load_dotenv
from lambdaprompt import prompt, AzureGPT4Prompt

load_dotenv.load_dotenv()
generate_n_tasks = AzureGPT4Prompt(
    "Today I will do {{ n }} things (comma separated) [", stop="]"
)
is_happy = AzureGPT4Prompt(
    "The task {{ task_detail }} is a task that will make me happy? (y/n):"
)


@prompt
def get_tasks_and_rate_is_happy(n=2):
    results = []
    for task in generate_n_tasks(n=n).split(","):
        results.append((task, is_happy(task)))
    return results


print(get_tasks_and_rate_is_happy())
