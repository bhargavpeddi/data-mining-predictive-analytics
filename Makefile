.PHONY: install eda train test all
install:
	pip install -r requirements.txt
eda:
	cd src && python eda.py
train:
	cd src && python train_evaluate.py
test:
	pytest -q
all: eda train
