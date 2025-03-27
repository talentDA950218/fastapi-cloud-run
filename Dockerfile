FROM python:3.12

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade -r requirements.txt

ENV APP_HOME=/root
WORKDIR $APP_HOME
COPY /app $APP_HOME/app

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]