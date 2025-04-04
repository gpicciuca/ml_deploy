# Company's sentiment analysis on social medias

This project implements an automated CI/CD Pipeline to integrate and deploy a sentiment analysis LLM from HuggingFace to continuosly evaluated a company's sentiment throughout social media posts.

## Components
The application is composed mainly of 2 parts:
- The track and schedule server, which runs Apache Airflow and MLFlow
- The ModelRunner server, on which the inferencing and training endpoints run

For the scope of this project, both components were put in the same repository under different subdirectories as it made testing easier via Docker Compose.

However, for a production-ready deployment, one would have had to separate them on different repositories as to not include code from one application in the deployment base of the other one.

### Track and Schedule Server
#### - Airflow
Airflow is mainly used for scheduling different operations:
- Weekly re-training of the LLM
  - Intended to be used with fresh Datasets containing new samples, which is not implemented at this stage
  - Sends requests to the ModelRunner Server's Endpoints to initiate the training and continuosly polls for status updates until the training completes (or fails)
  - If training fails, an alert is sent via email
- Model performance evaluation
  - Checks whether the accuracy of the model over the last X trainings has worsened by a given threshold. In that case, it will alert via email.
- Sentiment Inspector
  - Checks the Social Medias' posts to evaluate the customers' sentiment over the company
  - Currently, only Twitter API is implemented but it may easily be extended with other platforms
  - Could also be used to collect samples for new datasets used in weekly re-trainings (Not implemented)

#### - MLFlow
MLFlow keeps track of the training logs and metrics, giving an overall idea of how the model is performing on sample datasets.

Inference metrics will also be stored here, allowing the company to see what people think or feel about it.

### ModelRunner
ModelRunner uses FastAPI and Uvicorn to host a self-contained Training and Inference infrastructure.

When Airflow sends a training request, ModelRunner receives it and schedules a training right away, unless one is already running. In either case, it will give direct feedback whether the request was fulfilled or discarded by returning appropriate responses.

```
Note: In `model_runner/app/tasks/training.py` you will find that the `MODEL` being used to train on is `cardiffnlp/twitter-roberta-base-sentiment-latest`, which is then pushed to `gpicciuca/financial-twitter-roberta-sentiment` after training. However, for simplicity, in the `inferece.py` file the same original model is being reused instead of the trained one but in production, it should always fetch the model that we've trained and pushed to our own repository!
```

Training may take a long time, depending on the server performance on which the application is deployed on.
For this reason, a "get_state" endpoint was added so that Airflow can poll it at regular intervals to see if the training is still ongoing, if it has finished or if it ended with an error.

Once the training completes successfully, the updated model is pushed to HuggingFace and the Inference class reloads it to keep things up-to-date.
For simplicity, it pulls the model from HuggingFace but a more efficient way would be to directly re-use the trained model at that point in time.

## CI/CD Pipeline
The Pipeline makes use of Github Actions which takes care of the following:
- For Pull Requests, run our ModelRunner python integration tests
- For merged PRs, sync the repository with HuggingFace Space

## HuggingFace Space
HuggingFace Space is where the application (the ModelRunner) is deployed to and will run within a Docker Container.
It is accessible via http://xxx:7860/xxxx.
Everytime a PR is merged to this Github Repository, a CI Job will automatically take care of pushing it to HuggingFace, keeping the deployed application automatically in sync.

#

Resources used for this Project:
- https://github.com/peter-evans/docker-compose-actions-workflow
- https://github.com/marketplace/actions/run-pytest
- https://huggingface.co/docs/hub/spaces-github-actions
- https://developer.x.com/en/docs/x-api
- https://huggingface.co/docs/hub/spaces-sdks-docker-examples
- https://huggingface.co/spaces/SpacesExamples/fastapi_dummy/tree/main
- https://mlflow.org/docs/latest/tracking/tutorials/remote-server/

#

## Final thoughts
This is the final project I've made in scope of [Profession.AI's AI Engineering](https://profession.ai/corsi/master-ai-engineering/) course, module 5 "MLOps and Machine Learning in production".


curl -X POST '${{ secrets.AF_HOST }}/api/v1/dags/${{ secrets.AF_DAG_ML_TRAIN }}/dagRuns' -H 'Content-Type: application/json' --user "${{ secrets.AF_USER }}:${{ secrets.AF_USER_TOKEN }}" -d '{}'


curl -X PATCH '${{ secrets.AF_HOST }}/api/v1/dags/${{ secrets.AF_DAG_ML_TRAIN }}?update_mask=is_paused' -H 'Content-Type: application/json' --user "${{ secrets.AF_USER }}:${{ secrets.AF_USER_TOKEN }}" -d '{ "is_paused": false }'


curl -X 'POST' -H 'Content-Type: application/json' http://localhost:9000/inference -d '{"messages":["1", "2"]}'


curl -X 'PATCH' -H 'Content-Type: application/json' http://localhost:9000/inference -d '{"messages":["1", "2"]}'

