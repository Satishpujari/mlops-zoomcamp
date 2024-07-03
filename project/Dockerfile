FROM python:3.10.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app


RUN pipenv install --system --deploy

COPY ["app.py", "./"]
COPY ["flight_dataset.ipynb", "./"]
COPY ["evidently_metrics.ipynb", "./"]
COPY ["mlflow.db", "./"]
COPY ["mlruns/", "./mlruns/"]
COPY ["mlartifacts/", "./mlartifacts/"]

RUN mkdir models

EXPOSE 5011

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:5011", "app:app" ]
