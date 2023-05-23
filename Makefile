VENV := venv
# PYTHON := $(VENV)/bin/python
SRC := src
DOCKER_IMAGE := object-detection-api
TAG := $(shell date +%Y%m%d%H%M%S)

install:
	pip install --upgrade pip
	pip install -r requirements.txt

load-data:
	python $(SRC)/dataset.py

train:
	# Run the training script
	python $(SRC)/train.py

test:
	# Run the testing script
	python $(SRC)/eval.py

docker-build:
	# Build the Docker image with a unique tag
	docker build -t $(DOCKER_IMAGE):$(TAG) .

all: install load-data train test docker-build
