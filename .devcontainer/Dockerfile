FROM tensorflow/tensorflow

#
# EXPOSE port

# Install Docker for Docker-in-Docker
RUN apt-get update && \
    apt-get -qy full-upgrade && \
    apt-get install -qy curl && \
    apt-get install -qy curl && \
    curl -sSL https://get.docker.com/ | sh

# Install Libraries
RUN apt-get install sudo

# Install Java for Sonarlint
RUN apt-get update -y
RUN apt-get install openjdk-11-jre -y
# RUN apt-get install openjdk-6-jre-headless -y
RUN apt-get install git -y

# Torch Dependencies

# Set up the Python environment and app locationm
WORKDIR /app
ADD . /app
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pre-commit isort bandit black mypy flake8
RUN pre-commit install
RUN python3 -m pip install -r requirements.txt

# Install sound analysis libraries
RUN apt-get install libsndfile1 -y
RUN python3 -m pip install librosa

# Enable the virtual environment and run.
CMD ["bash -c python -m venv .env && source .env/bin/activate"]
