FROM python:3.7-alpine
ADD api.py /
WORKDIR /
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /
ENTRYPOINT ["python"]
CMD [ "python", "./api.py" ]