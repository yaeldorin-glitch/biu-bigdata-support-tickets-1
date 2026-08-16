.DEFAULT_GOAL := help
SHELL := /bin/bash
export PYTHONPATH := src

.PHONY: help install test offline demo up down logs topic produce stream kpis api indices clean sample

help:  ## show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install:  ## install python dependencies
	pip install -r requirements.txt

test:  ## run the unit tests
	python -m pytest tests/ -v

offline:  ## run the whole pipeline with no infrastructure (start here)
	python -m tickets.offline_pipeline --limit 5000

offline-full:  ## same, over the entire dataset
	python -m tickets.offline_pipeline --full

sample:  ## regenerate the committed 300-row sample from the full CSV
	python scripts/make_sample.py

# --- docker stack ----------------------------------------------------------

up:  ## start kafka, minio, elasticsearch and spark (slim: fits an 8GB machine)
	docker compose -f docker-compose.slim.yml up -d --build
	@echo "waiting for elasticsearch..."
	@until curl -fs http://localhost:9200/_cluster/health >/dev/null 2>&1; do sleep 3; done
	@echo "stack is up: minio http://localhost:9001 (no kibana in the slim stack)"

down:  ## stop the stack (keeps volumes)
	docker compose -f docker-compose.slim.yml down

destroy:  ## stop the stack and delete all data
	docker compose -f docker-compose.slim.yml down -v

logs:  ## tail the stack logs
	docker compose -f docker-compose.slim.yml logs -f --tail=100

indices:  ## create the elasticsearch indices with the dense_vector mapping
	python -c "from tickets.serving.es_client import create_indices; print(create_indices())"

produce:  ## replay the CSV onto the kafka topic
	python -m tickets.ingest.producer --rate 200

# Target docker/spark/run_module.py, not the .py file directly: spark-submit
# runs its target as a bare script with no parent package, which breaks the
# relative imports in tickets.spark.*. The launcher runs it as a real module
# import instead (the effect of `python -m`, which spark-submit has no flag for).

stream:  ## run the spark structured streaming job (needs a fitted offline model -- see docs/DEMO.md)
	docker compose -f docker-compose.slim.yml exec spark spark-submit \
		--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 \
		/opt/project/docker/spark/run_module.py tickets.spark.stream_job --trigger-seconds 5

kpis:  ## run the batch KPI job over the silver layer
	docker compose -f docker-compose.slim.yml exec spark spark-submit \
		--packages org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 \
		/opt/project/docker/spark/run_module.py tickets.spark.batch_kpis

api:  ## serve the REST API against elasticsearch
	uvicorn tickets.serving.api:app --host 0.0.0.0 --port 8000 --reload

api-offline:  ## serve the REST API with no infrastructure at all
	OFFLINE_API=1 uvicorn tickets.serving.api:app --host 0.0.0.0 --port 8000

clean:  ## remove generated output
	rm -rf output/ .pytest_cache/ spark-warehouse/ metastore_db/ derby.log
	find . -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true
