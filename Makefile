.PHONY: install ingestion api ui test clean

install:
	pip install -r requirements.txt

ingestion:
	python -m src.ingestion.run_ingestion

api:
	uvicorn src.api.main:app --reload --port 8000
	uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

ui:
	streamlit run src/ui/app.py
	streamlit run src/ui/app.py --server.address 0.0.0.0            

test:
	pytest tests/

clean:
	rm -rf vectorstore/*
	rm -rf data/processed/*
	find . -type d -name __pycache__ -exec rm -rf {} +