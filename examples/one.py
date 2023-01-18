from lambdaprompt import GPT3Prompt

example = GPT3Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
# then use it as a function
example(number=12, thing="apples")
