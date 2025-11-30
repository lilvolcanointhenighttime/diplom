from python:3.12.4-slim

LABEL name="fastapi_oauth"

WORKDIR .
RUN mkdir /src

COPY ./oauth/config/req.txt /src/oauth/req.txt
RUN python -m pip install -r ./src/oauth/req.txt

COPY ./oauth /src/oauth/