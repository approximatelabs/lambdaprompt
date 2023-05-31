from load_dotenv import load_dotenv

load_dotenv()

from lambdaprompt import AzureGPT4Prompt

example = AzureGPT4Prompt("Sally had {{ number }} of {{ thing }}. Sally sold ")
# then use it as a function
res = example(number=12, thing="apples")
print(res)
