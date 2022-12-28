ARG mode=prod

FROM python:3.11 as base
WORKDIR /code
COPY ./app-requirements.txt /code/app-requirements.txt

FROM base as image-dev
ADD . /code/
RUN  grep -vwE "lambdaprompt" app-requirements.txt > app-requirements2.txt
RUN mv app-requirements2.txt app-requirements.txt
RUN echo "-e ." >> app-requirements.txt

FROM base as image-prod
RUN echo "prod"

FROM image-${mode} AS final
RUN pip install --no-cache-dir --upgrade -r /code/app-requirements.txt
COPY ./app /code/app
CMD ["uvicorn", "app.main:app", "--reload", "--proxy-headers", "--host", "0.0.0.0", "--port", "80"]
