VENV := venv
PYTHON := $(VENV)/bin/python
SRC := src
DOCKER_IMAGE := object-detection-api
TAG := $(shell date +%Y%m%d%H%M%S)

all: load-data train test docker-build

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
