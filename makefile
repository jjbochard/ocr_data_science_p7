.PHONY: run_api_dev run_api_prod test_predict run_mlflow_dev

run_api_dev:
	uvicorn api:app --reload --host 0.0.0.0 --port 8000 --log-level debug

run_api_prod:
	uvicorn api:app --host 0.0.0.0 --port 8000

test_predict:
	curl -v POST http://localhost:8000/predict \
	  -H "Content-Type: application/json" \
	  --data-binary @payload.json

run_mlflow_dev:
	mlflow ui --host 127.0.0.1 --port 8080

run_mlflow_prod:
	mlflow server --host 127.0.0.1 --port 8080
