FROM svizor/zoomcamp-model:3.11.5-slim

WORKDIR /app

RUN pip install pipenv

COPY Pipfile Pipfile.lock ./
RUN pipenv install --system --deploy

COPY model1.bin dv.bin ./
COPY gunicorn_config.py predict_service.py ./

RUN useradd -m deploy
RUN chown -R deploy:deploy /app
USER deploy
EXPOSE 8000

CMD ["gunicorn", "--config", "gunicorn_config.py", "predict_service:app"]
