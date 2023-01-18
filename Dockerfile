FROM python:latest

# ENV OPENAI_API_KEY
RUN pip install lambdaprompt
ADD examples/one.py .
ADD examples/two.py .