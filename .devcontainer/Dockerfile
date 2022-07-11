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

# Torch Dependencies

# Set up the Python environment and app locationm
# You can't get away from the workspaces folder, CODESPACES_VIRTUAL_FOLDER isn't apparently changeable, therefore,
# we mimic the behavior of the workspaces folder, and install the only files we need for set up. The project file 
# folders will be copied automatically, by codespaces.
#
# *** If this doesn't work locally, that's why. ***
# *** change the ADD line to: ***
# ADD . /worksapces/playground
#
WORKDIR /workspaces/playground
ADD . /workspaces/playground

# Install Virtual Environment requirements
RUN apt install python3.8-venv
RUN python3 -m pip install virtualenv
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

### The rest installs in the virtual environment. ###

# Install Git 
RUN apt-get install git -y

# Install Java for SonarLint
RUN apt-get install openjdk-11-jre -y
# RUN apt-get install openjdk-6-jre-headless -y

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install pre-commit isort bandit black mypy flake8
RUN pre-commit install

RUN python3 -m pip install -r requirements.txt

# Install sound analysis libraries
RUN apt-get install libsndfile1 -y
RUN python3 -m pip install librosa

# Enable the virtual environment and run.
# CMD ["bash -c && . source .venv/bin/activate"]
