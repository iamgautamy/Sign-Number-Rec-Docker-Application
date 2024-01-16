FROM python:3.11.5

WORKDIR /usr/src/app

COPY app.py .
COPY model.pkl .
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["python","./app.py"]