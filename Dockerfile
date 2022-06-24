FROM ubuntu:latest

RUN apt-get update -y

RUN apt-get install python3 -y

RUN apt-get install python3-pip -y

RUN apt-get pip install numpy -y

RUN apt-get pip install pandas -y

RUN apt-get install python-matplotlib -y

RUN apt-get install python-sklearn -y

RUN apt-get pip install xgboost

WORKDIR /home

COPY . .

CMD ["python3", "main.py"]