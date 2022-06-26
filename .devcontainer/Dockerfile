FROM tensorflow/tensorflow

#
# EXPOSE port 

# Install Docker for Docker-in-Docker
RUN apt-get update && \
    apt-get -qy full-upgrade && \
    apt-get install -qy curl && \
    apt-get install -qy curl && \
    curl -sSL https://get.docker.com/ | sh

# Install LIbraries
RUN apt-get install sudo

# Install Java for Sonarlint
RUN apt-get update -y
RUN apt-get install openjdk-11-jre -y
# RUN apt-get install openjdk-6-jre-headless -y
RUN apt-get install git -y

# Set up the Python environment and app locationm
WORKDIR /app
ADD . /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r requirements.txt

# Enable the virtual environment and run.
CMD ["bash -c python -m venv .env && source .env/bin/activate"] 
