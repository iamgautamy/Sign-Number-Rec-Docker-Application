FROM python:3.11.5

WORKDIR /usr/src/app

COPY app.py .
COPY model.pkl .
COPY requirements.txt .

RUN pip install -r requirements.txt
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
CMD ["python","./app.py"]
