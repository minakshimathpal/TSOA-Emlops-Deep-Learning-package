FROM python:3.9.14-slim-bullseye

WORKDIR /opt/emlo

COPY requirements.txt requirements.txt

RUN pip3 install -r requirements.txt && rm -rf /root/.cache/pip

COPY . .

CMD ["python", "emloCarVsDog/train.py","emloCarVsDog/eval.py"]