import lambdaprompt


@lambdaprompt.prompt
def wellhello(name, age):
    return f"Hello {name}, you are {age} years old"


@lambdaprompt.prompt
async def asynctest(monkey: str):
    return f"Hello {monkey}"


lambdaprompt.GPT3Prompt(
    """
    {{name}} is {{age}} years old.

    What are their hobbies?
    """,
    name="hobby_finder",
)
