#FROM scaleoutsystems/fedn-client:develop
##RUN apt-get update && \
##    apt-get upgrade -y && \
##    apt-get install -y git
#
##RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn
#
#COPY client.yaml /app/
#COPY requirements.txt /app/
##COPY hello.py /app/
##COPY client /app/
#WORKDIR /app
#RUN pip install -r requirements.txt

FROM python:3.8.9
#COPY client.yaml /app/
COPY requirements.txt /app/
WORKDIR /app/
RUN pip install -r requirements.txt
RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn

#FROM python:3.8.9
#COPY requirements.txt /app/
##COPY client.yaml /app/
#WORKDIR /app
#RUN pip install -r requirements.txt
#RUN pip install -e git://github.com/scaleoutsystems/fedn.git@develop#egg=fedn\&subdirectory=fedn