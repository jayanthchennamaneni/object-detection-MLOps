VENV := venv
PYTHON := $(VENV)/bin/python
SRC := src
DOCKER_IMAGE := object-detection-api
TAG := $(shell date +%Y%m%d%H%M%S)

environment:
	$(PYTHON) -m venv venv
    source venv/bin/activate

install:
	pip install --upgrade pip
	pip install -r requirements.txt

load-data:
	$(PYTHON) $(SRC)/dataset.py

train:
	# Run the training script
	$(PYTHON) $(SRC)/train.py

test:
	# Run the testing script
	$(PYTHON) $(SRC)/eval.py

docker-build:
	# Build the Docker image with a unique tag
	docker build -t $(DOCKER_IMAGE):$(TAG) .

all: install load-data train test docker-build
