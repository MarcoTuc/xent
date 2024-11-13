FROM base:latest

COPY . /home/marktas

USER marktas

RUN pip install -r requirements.txt
RUN python setup.py

USER root
RUN chown -R marktas:csft /home/marktas
USER marktas



