FROM python:3.6

WORKDIR /app

# for dlib
RUN apt-get update && apt-get -y install cmake

# install dependencies
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --cache-dir=/cache -r requirements.txt

#copy project directory
COPY ./ .


ENTRYPOINT ["python", "./main.py"]