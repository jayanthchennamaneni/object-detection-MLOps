VENV := venv
PYTHON := $(VENV)/bin/python
SRC := src
dataset = $(SRC)/dataset.py
train = $(SRC)/train.py
test = $(SRC)/eval.py

all: load-data train test 

load-data:
	$(PYTHON) $(SRC)/$(dataset)

train:
	# Run the training script
	$(PYTHON) $(SRC)/$(train)

test:
	# Run the testing script
	$(PYTHON) $(SRC)/$(test)






