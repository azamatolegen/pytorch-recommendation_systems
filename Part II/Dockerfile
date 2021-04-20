FROM python:3.7.3-stretch

COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt

COPY ./app /app
COPY ./data /data

WORKDIR /app

CMD ["python","./api.py"]