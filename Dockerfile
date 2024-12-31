# # To build:
# docker build -t bike-trip-time-model .
# # To run:
# docker run -p 0.0.0.0:5000:5000  \
#     --name bike-trip-time-model-container \
#     bike-trip-time-model \
#     uvicorn app.main:app --host 0.0.0.0 --port 5000    

FROM python:3.11-slim as base

RUN mkdir /code
WORKDIR /code

COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r requirements.txt

RUN mkdir app 
COPY app/*.py app
COPY app/*.joblib app

# Spin a server
CMD ["uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "5000"]