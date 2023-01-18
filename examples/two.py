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
