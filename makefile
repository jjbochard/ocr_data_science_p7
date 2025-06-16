.PHONY: run_api_dev test_predict run_mlflow_dev run_streamlit run_test

run_api_dev:
	uvicorn api.api:app --reload --host 0.0.0.0 --port 8000 --log-level debug

test_predict:
	curl -v POST https://ds9mhppgjv.eu-west-3.awsapprunner.com/predict \
	  -H "Content-Type: application/json" \
	  --data-binary @api/payload.json

run_mlflow_dev:
	mlflow ui --host 127.0.0.1 --port 8080

run_streamlit:
	streamlit run api/streamlit_app.py

run_test:
	pytest -s api/test_api.py
