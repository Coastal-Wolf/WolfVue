FROM python:3.11-slim

RUN apt update && apt -y install libglu1-mesa-dev libgio-2.0-dev

WORKDIR /root/app

COPY requirements.txt requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

ENTRYPOINT [ "python", "WolfVue.py" ]

#ENTRYPOINT ["tail"]
#CMD ["-f","/dev/null"]