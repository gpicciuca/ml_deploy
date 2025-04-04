# This Dockerfile serves as the build file for Huggingface Spaces
FROM huggingface/transformers-pytorch-gpu

ARG ML_APP_LISTEN_PORT=7860
ARG ML_MLFLOW_ENDPOINT
ARG ML_HF_ACCESS_TOKEN

ENV APP_LISTEN_PORT=${ML_APP_LISTEN_PORT}
ENV MLFLOW_ENDPOINT=${ML_MLFLOW_ENDPOINT}
ENV HF_ACCESS_TOKEN=${ML_HF_ACCESS_TOKEN}

RUN apt-get update
RUN /usr/bin/python3 -m pip install uvicorn fastapi mlflow huggingface_hub httpx

WORKDIR /app

COPY ./app /app

EXPOSE ${ML_APP_LISTEN_PORT}

ENTRYPOINT [ "/usr/bin/python3", "/app/main.py" ]
