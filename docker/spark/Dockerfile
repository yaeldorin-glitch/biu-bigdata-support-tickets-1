# Spark image with the project's Python dependencies baked in.
#
# Bitnami's Spark 3.5 image ships Python 3.11 and Spark built against Hadoop 3.3,
# which is what pins hadoop-aws to 3.3.4 in tickets/spark/session.py.
FROM bitnamilegacy/spark:3.5.1-debian-12-r9

USER root

# libgomp is required by scikit-learn's OpenMP kernels; the slim base omits it.
RUN install_packages libgomp1 curl

COPY requirements.txt /tmp/requirements.txt

# This image runs with EMBEDDING_BACKEND=offline and LLM_PROVIDER=none (see
# docker-compose.slim.yml) and never imports sentence-transformers or anthropic
# (both are lazy imports in core/embeddings.py and ai/llm.py), so skip them here.
# sentence-transformers alone pulls torch's CUDA wheels -- multiple GB -- on a
# container that has no GPU and will never load a neural model.
RUN grep -vE "^(sentence-transformers|anthropic)" /tmp/requirements.txt > /tmp/requirements-spark.txt \
    && pip install --no-cache-dir -r /tmp/requirements-spark.txt

# Pre-fetch the Spark connector jars at build time so the first `spark-submit`
# does not stall on an Ivy resolve. Failure is non-fatal: the job resolves them
# at runtime instead.
RUN spark-submit \
      --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.1,org.apache.hadoop:hadoop-aws:3.3.4,com.amazonaws:aws-java-sdk-bundle:1.12.262,org.elasticsearch:elasticsearch-spark-30_2.12:8.13.4 \
      --class org.apache.spark.examples.SparkPi \
      local:///opt/bitnami/spark/examples/jars/spark-examples_2.12-3.5.1.jar 1 \
    || echo "jar prefetch skipped (offline build); they will resolve at run time"

USER 1001

WORKDIR /opt/project
