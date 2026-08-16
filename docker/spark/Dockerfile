# Spark image with the project's Python dependencies baked in.
#
# Bitnami's Spark 3.5 image ships Python 3.11 and Spark built against Hadoop 3.3,
# which is what pins hadoop-aws to 3.3.4 in tickets/spark/session.py.
FROM bitnami/spark:3.5.1

USER root

# libgomp is required by scikit-learn's OpenMP kernels; the slim base omits it.
RUN install_packages libgomp1 curl

COPY requirements.txt /tmp/requirements.txt

# --no-cache-dir keeps the layer small. sentence-transformers pulls torch, which
# is ~800MB; comment out the ML extras in requirements.txt if you only intend to
# run the offline embedding backend.
RUN pip install --no-cache-dir -r /tmp/requirements.txt

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
