"""
PySpark feature engineering job for the FinSight fraud detection model.

Two execution modes
--------------------
  batch  (default)
      Reads data/transactions.csv, computes seven derived features, and writes
      the enriched dataset to data/features.parquet for XGBoost training.

      This is a static, bounded job: Spark knows the total row count up-front,
      can optimise join strategies (broadcast, sort-merge), and writes exactly
      once with overwrite semantics. Use this mode to rebuild training data.

  stream
      Reads live JSON messages from a Kafka topic ("transactions"), applies the
      same feature transformations, and writes enriched records to the console
      in micro-batches. Use this mode when the pipeline is connected to a live
      Kafka broker (e.g. the producer.py feed).

Batch vs Streaming — key differences
--------------------------------------
  Batch                           Streaming
  ──────────────────────────────  ────────────────────────────────────────────
  spark.read (bounded source)     spark.readStream (unbounded source)
  df.count() works                df.count() raises AnalysisException
  Window aggregations (1-h roll)  Watermarking + eventTime window required
  .write.parquet(path)            .writeStream.format(...).start()
  Runs to completion              Runs forever; stop with query.stop()
  No checkpoint needed            Checkpoint required for exactly-once delivery

  The most important practical difference: Spark's standard window functions
  (Window.rangeBetween) are NOT supported on streaming DataFrames. Instead,
  streaming uses F.window() time-bucketed aggregations combined with
  withWatermark() to bound state size. The streaming mode below uses a 1-hour
  tumbling window with a 10-minute watermark as the streaming equivalent of the
  batch 1-hour rolling aggregation.

Run from the project root:

    # Batch (default) — rebuilds data/features.parquet
    python data_pipeline/spark/feature_engineering.py

    # Stream — reads live from local Kafka broker
    python data_pipeline/spark/feature_engineering.py --mode stream

    # Stream — custom broker
    python data_pipeline/spark/feature_engineering.py --mode stream \\
        --kafka-bootstrap my-broker:9092
"""

import argparse
import os
import sys

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql.types import (
    DoubleType, IntegerType, StringType,
    StructField, StructType, TimestampType,
)

# ── I/O paths ─────────────────────────────────────────────────────────────────
INPUT_CSV = "data/transactions.csv"
OUTPUT_PARQUET = "data/features.parquet"

# Kafka topic published by data_pipeline/kafka/producer.py
KAFKA_TOPIC = "transactions"

# Streaming checkpoint directory.
# Spark writes micro-batch offsets and aggregate state here so the job can
# resume exactly where it left off after a crash or restart — without it, every
# restart would re-process all offsets from the beginning (or skip them,
# depending on the startingOffsets config), potentially producing duplicates or
# gaps.
CHECKPOINT_DIR = "/tmp/finsight_checkpoint"

# ── Merchant risk scores (0–1) ────────────────────────────────────────────────
# Domain-encoded risk weight per category. Cash-equivalent and high-liquidity
# merchants (wire transfer, crypto, pawn) score near 1.0 because they are the
# primary channels for converting stolen card access into untraceable cash.
# Everyday merchants score low — fraud there is rare and low-value.
MERCHANT_RISK = {
    "grocery": 0.05,
    "restaurant": 0.05,
    "gas_station": 0.10,
    "retail": 0.10,
    "pharmacy": 0.05,
    "entertainment": 0.10,
    "travel": 0.20,   # elevated — card-not-present bookings are common fraud vectors
    "subscription": 0.05,
    "utilities": 0.05,
    "healthcare": 0.05,
    "pawn_shop": 0.90,
    "wire_transfer": 0.95,
    "crypto_exchange": 0.90,
    "money_order": 0.85,
    "check_cashing": 0.85,
}

# ── Explicit schema ───────────────────────────────────────────────────────────
# Avoids the inferSchema scan (slow on 500k rows) and guarantees the correct
# types for window functions that require a numeric ordering column.
# This schema is shared between batch (CSV) and streaming (JSON payload).
SCHEMA = StructType([
    StructField("transaction_id", StringType(), False),
    StructField("timestamp", TimestampType(), True),
    StructField("user_id", StringType(), False),
    StructField("cardholder_name", StringType(), True),
    StructField("merchant_name", StringType(), True),
    StructField("merchant_category", StringType(), True),
    StructField("amount", DoubleType(), False),
    StructField("city", StringType(), True),
    StructField("state", StringType(), True),
    StructField("country", StringType(), True),
    StructField("hour_of_day", IntegerType(), True),
    StructField("day_of_week", IntegerType(), True),  # 0=Mon … 6=Sun
    StructField("is_fraud", IntegerType(), False),
    StructField("fraud_type", StringType(), True),
])


# ─────────────────────────────────────────────────────────────────────────────
# SHARED FEATURE HELPERS
#
# These helpers apply the same transformations in both batch and streaming mode.
# Keeping them separate from the I/O logic means neither run_batch() nor
# run_streaming() duplicates feature code.
# ─────────────────────────────────────────────────────────────────────────────

def _add_scalar_features(df):
    """
    Add the five scalar (per-row) features that work identically in batch and
    streaming mode because they depend only on the current row's values:

      amount_log     — log1p-compressed transaction amount
      is_night       — 1 if hour_of_day ∈ [23, 04]
      is_weekend     — 1 if day_of_week ∈ {5, 6}
      merchant_risk  — lookup from MERCHANT_RISK map
      amount_x_risk  — interaction term: amount × merchant_risk

    The two velocity features (txn_count_1h, amount_sum_1h) are NOT included
    here because their implementation differs between batch (Window.rangeBetween)
    and streaming (F.window + watermark).
    """
    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 1: amount_log
    # Transaction amounts are heavily right-skewed (log-normal distribution).
    # Without transformation, a $15,000 fraud transaction would dwarf $30
    # grocery purchases and dominate every split. Log-compressing the scale
    # lets the model treat proportional differences as equal: $10→$100 looks
    # the same as $100→$1,000. log1p handles the edge case of $0.00 auths.
    # ─────────────────────────────────────────────────────────────────────────
    df = df.withColumn("amount_log", F.log1p(F.col("amount")))

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 2: is_night
    # Legitimate card activity drops sharply after 11 PM; the synthetic
    # high-risk-merchant/odd-hours pattern concentrates entirely in [0, 5).
    # A binary flag creates a hard decision boundary the model can exploit
    # without needing to learn the curved relationship from hour_of_day alone.
    # ─────────────────────────────────────────────────────────────────────────
    df = df.withColumn(
        "is_night",
        F.when((F.col("hour_of_day") >= 23) | (F.col("hour_of_day") < 5), 1)
         .otherwise(0),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 3: is_weekend
    # Fraud rates increase on weekends: cardholders check statements less
    # frequently and bank fraud-ops teams run at reduced capacity, giving
    # fraudsters a longer window before the card is frozen. day_of_week
    # follows Python's weekday() convention — Saturday=5, Sunday=6.
    # ─────────────────────────────────────────────────────────────────────────
    df = df.withColumn(
        "is_weekend",
        F.when(F.col("day_of_week") >= 5, 1).otherwise(0),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 4: amount_x_risk
    # Neither amount nor merchant category alone fully captures fraud risk —
    # a $5,000 grocery purchase is suspicious; a $20 wire transfer is normal.
    # Multiplying amount × merchant_risk creates an interaction term: a $5,000
    # wire transfer scores 4,750 while the same amount at a grocery store
    # scores 250, surfacing the joint signal without needing a tree to discover
    # it through multiple splits across two separate features.
    # ─────────────────────────────────────────────────────────────────────────
    risk_pairs = [item for kv in MERCHANT_RISK.items()
                  for item in (F.lit(kv[0]), F.lit(kv[1]))]
    risk_map = F.create_map(*risk_pairs)

    df = (df
          .withColumn("merchant_risk", risk_map[F.col("merchant_category")])
          .withColumn("amount_x_risk", F.col("amount") * F.col("merchant_risk")))

    return df


# ─────────────────────────────────────────────────────────────────────────────
# BATCH MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_batch(spark: SparkSession) -> None:
    """
    Read transactions.csv, engineer all 7 features, write features.parquet.

    Batch processing is appropriate here because:
      - The input is a bounded, static CSV file.
      - Window functions (rangeBetween) require knowing all rows for a user
        before emitting a result — only possible when the dataset is complete.
      - Training data is rebuilt periodically (nightly/weekly), not on every
        incoming transaction.

    In contrast, streaming would be used for real-time scoring where features
    must be computed per-transaction as they arrive from Kafka.
    """
    # ── Load ──────────────────────────────────────────────────────────────────
    # spark.read returns a static DataFrame — Spark scans the file once and
    # knows the total number of rows before executing any transformations.
    print(f"[batch] Reading {INPUT_CSV}...")
    df = (spark.read
          .option("header", "true")
          .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
          .schema(SCHEMA)
          .csv(INPUT_CSV))

    n = df.count()
    print(f"[batch] Loaded {n:,} transactions.\n")

    # Unix epoch seconds — needed as a numeric ordering key for rangeBetween
    # windows. TimestampType cannot be used directly in rangeBetween.
    df = df.withColumn("ts_unix", F.unix_timestamp("timestamp"))

    # ── Scalar features (shared helper) ──────────────────────────────────────
    print("[batch] Computing scalar features (amount_log, is_night, is_weekend, "
          "amount_x_risk)...")
    df = _add_scalar_features(df)

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 5: account_age_bucket  (batch-only: requires full user history)
    # Newly opened or newly compromised accounts are exploited before the
    # cardholder establishes behavioral baselines or before fraud monitoring
    # systems have enough history to flag anomalies. We proxy "account creation
    # date" with each user's earliest observed transaction in the dataset, then
    # bucket the resulting age into three risk tiers:
    #   0 = new         (< 90 days)   highest risk
    #   1 = mid         (< 365 days)  moderate risk
    #   2 = established (≥ 365 days)  baseline risk
    #
    # This feature requires a global min("timestamp") per user — only possible
    # in batch mode where all rows are available. In streaming mode we use a
    # fixed bucket of 2 (established) as a safe default because we do not yet
    # have enough per-user history within the current micro-batch.
    # ─────────────────────────────────────────────────────────────────────────
    print("[batch] Computing account_age_bucket...")
    first_seen = (df.groupBy("user_id")
                    .agg(F.min("ts_unix").alias("first_seen_unix")))
    df = df.join(first_seen, on="user_id", how="left")
    df = df.withColumn(
        "account_age_days",
        (F.col("ts_unix") - F.col("first_seen_unix")) / 86_400.0,
    )
    df = df.withColumn(
        "account_age_bucket",
        F.when(F.col("account_age_days") < 90, 0)
         .when(F.col("account_age_days") < 365, 1)
         .otherwise(2)
         .cast(IntegerType()),
    )

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURES 6 & 7: txn_count_1h, amount_sum_1h  (batch: rolling window)
    # Per-user rolling aggregations over the 60 minutes before each transaction.
    # These are the primary velocity signals: in the velocity-burst pattern a
    # fraudster runs 5–10 transactions in minutes, so txn_count_1h will be 5–10
    # while a normal user rarely exceeds 2–3 in a full hour.
    # amount_sum_1h catches cases where individual amounts look normal but the
    # cumulative spend in an hour is abnormal.
    #
    # rangeBetween(-3600, 0): include all rows whose ts_unix falls within
    # 3600 seconds (1 hour) before the current row, inclusive. The current
    # transaction is always included (self-count ≥ 1 by design).
    #
    # BATCH vs STREAMING: Window.rangeBetween is forbidden on streaming
    # DataFrames because Spark cannot maintain unbounded per-user state. In
    # streaming mode we use F.window() tumbling windows instead (see below).
    # ─────────────────────────────────────────────────────────────────────────
    print("[batch] Computing txn_count_1h and amount_sum_1h (rolling window)...")
    w_1h = (Window
            .partitionBy("user_id")
            .orderBy("ts_unix")
            .rangeBetween(-3_600, 0))

    df = (df
          .withColumn("txn_count_1h", F.count("transaction_id").over(w_1h))
          .withColumn("amount_sum_1h", F.sum("amount").over(w_1h)))

    # ── Drop intermediate columns not needed downstream ───────────────────────
    df = df.drop("ts_unix", "first_seen_unix", "account_age_days", "merchant_risk")

    # ── Write ─────────────────────────────────────────────────────────────────
    # Snappy compression: fast decompression during training (CPU-bound)
    # outweighs the smaller file size from gzip (I/O-bound only if on disk).
    os.makedirs("data", exist_ok=True)
    print(f"\n[batch] Writing to {OUTPUT_PARQUET}...")
    (df.write
       .mode("overwrite")
       .option("compression", "snappy")
       .parquet(OUTPUT_PARQUET))

    # ── Validation stats (read back from parquet — avoids recomputing window fns)
    print("[batch] Done. Reading back for validation stats...\n")
    result = spark.read.parquet(OUTPUT_PARQUET)

    print(f"Rows written : {result.count():,}")
    print(f"Columns      : {len(result.columns)}")
    print(f"  {result.columns}\n")

    print("Feature distributions:")
    result.select(
        "amount_log", "is_night", "is_weekend",
        "amount_x_risk", "account_age_bucket",
        "txn_count_1h", "amount_sum_1h",
    ).describe().show(truncate=False)

    # Confirm fraud patterns are separable in the new features
    print("Feature means split by fraud label (0=legit, 1=fraud):")
    (result
     .groupBy("is_fraud")
     .agg(
         F.round(F.mean("txn_count_1h"), 2).alias("avg_txn_count_1h"),
         F.round(F.mean("amount_sum_1h"), 2).alias("avg_amount_sum_1h"),
         F.round(F.mean("amount_log"), 2).alias("avg_amount_log"),
         F.round(F.mean("amount_x_risk"), 2).alias("avg_amount_x_risk"),
         F.round(F.mean("is_night"), 3).alias("pct_night"),
         F.round(F.mean("is_weekend"), 3).alias("pct_weekend"),
     )
     .orderBy("is_fraud")
     .show(truncate=False))


# ─────────────────────────────────────────────────────────────────────────────
# STREAMING MODE
# ─────────────────────────────────────────────────────────────────────────────

def run_streaming(spark: SparkSession, kafka_bootstrap: str) -> None:
    """
    Subscribe to the Kafka "transactions" topic and compute features in
    micro-batches using Spark Structured Streaming.

    How Spark Structured Streaming works
    ──────────────────────────────────────
    Spark treats the Kafka topic as an infinite table of rows. Every few
    seconds (the trigger interval) it pulls the latest batch of messages
    (a "micro-batch"), processes them exactly like a static DataFrame, and
    appends the results to the output sink. The driver maintains a checkpoint
    so that on restart it knows which Kafka offsets have already been processed.

    Kafka message layout
    ─────────────────────
    Each Kafka message produced by producer.py has:
      key   — transaction_id (string)
      value — full transaction JSON matching SCHEMA (bytes)

    Spark's Kafka source delivers a DataFrame with columns:
      key, value, topic, partition, offset, timestamp, timestampType

    We decode value (bytes → string) then parse it as JSON using from_json()
    with the explicit SCHEMA, producing a "data" struct column. The final
    .select("data.*") flattens it into individual named columns identical to
    the batch CSV input.

    Velocity features in streaming mode
    ─────────────────────────────────────
    Window.rangeBetween (used in batch) is NOT supported on streaming
    DataFrames — Spark cannot maintain per-user unbounded ordered state.
    Instead, we use:

      withWatermark("timestamp", "10 minutes")
          Tells Spark to discard event-time state older than 10 minutes.
          Without a watermark the state store grows forever.

      F.window("timestamp", "1 hour")
          Groups rows into non-overlapping 1-hour tumbling windows by event
          time. All transactions in the same [HH:00, HH+1:00) bucket for the
          same user are aggregated together.

    The result is a per-user-per-hour aggregate: each output row gives
    txn_count_1h and amount_sum_1h for all transactions that fell in the same
    1-hour bucket, which is the streaming equivalent of the batch rolling sum.

    account_age_bucket in streaming mode
    ─────────────────────────────────────
    The batch job computes first_seen as the global minimum timestamp across
    all historical data — impossible in streaming because we only see the
    current micro-batch. As a conservative default, every streaming record
    receives account_age_bucket = 2 (established). A production system would
    look this up from an external store (Redis, DynamoDB) keyed by user_id.

    Parameters
    ----------
    spark           : SparkSession
    kafka_bootstrap : str   Kafka broker address, e.g. "localhost:9092"
    """
    print(f"[stream] Connecting to Kafka broker: {kafka_bootstrap}")
    print(f"[stream] Topic: {KAFKA_TOPIC}")
    print(f"[stream] Checkpoint: {CHECKPOINT_DIR}")
    print("[stream] Press Ctrl+C to stop.\n")

    # ── 1. Read from Kafka ────────────────────────────────────────────────────
    # spark.readStream returns a streaming DataFrame — an unbounded table that
    # grows with each new Kafka message. No action runs until .start() is called
    # on the writeStream sink at the end of this function.
    #
    # Batch equivalent:
    #   spark.read.format("kafka").option(...).load()
    #
    # Key readStream options:
    #   subscribe         — comma-separated list of topics to consume
    #   startingOffsets   — "latest": skip historical messages; start from now.
    #                       Use "earliest" to replay from the beginning (e.g.
    #                       when replaying a backfill after a pipeline outage).
    #   failOnDataLoss    — False: tolerate offset gaps (Kafka retention expired)
    #                       rather than aborting the job.
    raw_stream = (
        spark.readStream
        .format("kafka")
        .option("kafka.bootstrap.servers", kafka_bootstrap)
        .option("subscribe", KAFKA_TOPIC)
        .option("startingOffsets", "latest")
        .option("failOnDataLoss", "false")
        .load()
    )

    # ── 2. Decode and parse JSON payload ─────────────────────────────────────
    # The Kafka "value" column is binary (BinaryType). Cast to string first,
    # then parse the JSON string into a struct using the shared SCHEMA.
    #
    # from_json() silently sets fields to null when the JSON key is missing or
    # the value cannot be cast to the declared type — prefer explicit nullability
    # checks downstream over silent silent data quality degradation.
    parsed = (
        raw_stream
        .select(F.from_json(
            F.col("value").cast("string"),
            SCHEMA,
        ).alias("data"))
        .select("data.*")           # flatten struct → individual columns
    )

    # ── 3. Scalar features (shared with batch) ────────────────────────────────
    # amount_log, is_night, is_weekend, merchant_risk, amount_x_risk are pure
    # row-level transforms — identical in batch and streaming mode.
    enriched = _add_scalar_features(parsed)

    # ── 4. account_age_bucket — streaming approximation ──────────────────────
    # In batch mode we join against a global min(timestamp) per user. In
    # streaming mode that global minimum is unknown. We assign bucket=2
    # (established) as a safe default. A production implementation would
    # replace this with a lookup against a user-profile feature store.
    enriched = enriched.withColumn(
        "account_age_bucket",
        F.lit(2).cast(IntegerType()),
    )

    # ── 5. Velocity features — 1-hour tumbling window ─────────────────────────
    # STREAMING vs BATCH difference:
    #
    #   Batch uses Window.rangeBetween(-3600, 0) — a per-row trailing window
    #   that looks back exactly 3,600 seconds for each individual transaction.
    #   This requires all rows to be present and sorted before evaluation.
    #
    #   Streaming uses F.window("timestamp", "1 hour") — a tumbling event-time
    #   window that groups ALL transactions for a user that fall in the same
    #   calendar hour bucket [HH:00:00, HH+1:00:00) and aggregates them once
    #   the watermark has passed (i.e. no more late-arriving data expected).
    #
    # withWatermark("timestamp", "10 minutes"):
    #   Declares that data arriving more than 10 minutes late (measured against
    #   the maximum observed event timestamp) can be dropped. This bounds the
    #   state store — without it Spark must retain state forever waiting for
    #   arbitrarily late records.
    #
    # The output of this aggregation has columns:
    #   window        — struct{start: timestamp, end: timestamp}
    #   user_id       — partition key
    #   txn_count_1h  — count of transactions in this user/window bucket
    #   amount_sum_1h — total amount in this user/window bucket
    windowed = (
        enriched
        .withWatermark("timestamp", "10 minutes")
        .groupBy(
            F.window("timestamp", "1 hour").alias("time_window"),
            F.col("user_id"),
        )
        .agg(
            F.count("transaction_id").alias("txn_count_1h"),
            F.round(F.sum("amount"), 2).alias("amount_sum_1h"),
            # Carry through representative scalar features for the window.
            # last() is safe here because all rows in a 1-h window for the same
            # user share the same is_night / is_weekend values (hour and
            # day_of_week don't change within a 1-hour bucket).
            F.last("amount_log").alias("amount_log"),
            F.last("amount_x_risk").alias("amount_x_risk"),
            F.last("is_night").alias("is_night"),
            F.last("is_weekend").alias("is_weekend"),
            F.last("merchant_category").alias("merchant_category"),
            F.last("account_age_bucket").alias("account_age_bucket"),
        )
    )

    # ── 6. Write to console (development sink) ───────────────────────────────
    # The console sink prints each micro-batch to stdout — useful for local
    # development and smoke-testing the pipeline end-to-end without needing
    # an output store.
    #
    # Production alternatives for the output sink:
    #   .format("parquet")  — append to a partitioned Delta/Parquet table
    #   .format("kafka")    — publish enriched records to a downstream topic
    #   .format("delta")    — Delta Lake for ACID streaming writes
    #
    # outputMode options:
    #   "append"   — emit only new rows (works for stateless transforms).
    #   "update"   — emit only rows whose aggregate value changed this batch.
    #   "complete" — re-emit the full result table every batch (expensive).
    #
    # We use "update" because the windowed aggregation updates existing window
    # buckets as new messages arrive within the same hour. "append" would only
    # emit rows after the watermark has closed the window (10 min after the
    # window end), introducing 70-minute latency. "update" emits partial
    # aggregates each micro-batch, making real-time scoring feasible.
    #
    # checkpointLocation is required for all stateful streaming queries. Spark
    # writes Kafka offset commits and window state here so the job is restartable
    # with exactly-once processing semantics.
    query = (
        windowed.writeStream
        .format("console")
        .outputMode("update")
        .option("truncate", "false")
        .option("numRows", 20)
        .option("checkpointLocation", CHECKPOINT_DIR)
        .trigger(processingTime="10 seconds")   # micro-batch interval
        .start()
    )

    print("[stream] Streaming query started. Waiting for Kafka messages...")
    print(f"[stream] Output mode: update  |  Trigger: every 10 seconds")
    print(f"[stream] Checkpoint: {CHECKPOINT_DIR}\n")

    try:
        # awaitTermination() blocks the driver thread indefinitely.
        # The query runs in a background thread; this call makes the main
        # thread wait for it to finish (which only happens on query.stop() or
        # an unrecoverable error). Without this call the script would exit
        # immediately, killing the streaming query.
        query.awaitTermination()
    except KeyboardInterrupt:
        print("\n[stream] Interrupt received — stopping query gracefully...")
        query.stop()
        print("[stream] Query stopped.")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FinSight feature engineering — batch or streaming mode",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python data_pipeline/spark/feature_engineering.py\n"
            "  python data_pipeline/spark/feature_engineering.py --mode stream\n"
            "  python data_pipeline/spark/feature_engineering.py "
            "--mode stream --kafka-bootstrap my-broker:9092"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["batch", "stream"],
        default="batch",
        help=(
            "batch: read CSV, write Parquet (default). "
            "stream: read live from Kafka, write to console."
        ),
    )
    parser.add_argument(
        "--kafka-bootstrap",
        default="localhost:9092",
        metavar="HOST:PORT",
        help="Kafka bootstrap server address (stream mode only). "
             "Default: localhost:9092",
    )
    args = parser.parse_args()

    # ── Batch pre-flight check ─────────────────────────────────────────────────
    if args.mode == "batch" and not os.path.exists(INPUT_CSV):
        sys.exit(
            f"[ERROR] {INPUT_CSV} not found.\n"
            "Run data_pipeline/generators/transaction_generator.py first."
        )

    # ── Build SparkSession ────────────────────────────────────────────────────
    # The kafka connector package is only needed in streaming mode, but adding
    # it unconditionally keeps the session config consistent across both modes.
    # The spark-sql-kafka jar is downloaded from Maven on first run; subsequent
    # runs use the local Ivy cache (~/.ivy2).
    builder = (
        SparkSession.builder
        .appName("finsight-feature-engineering")
        .master("local[*]")
        # Default of 200 shuffle partitions causes ~190 empty partitions on a
        # local 500k-row dataset — 8 matches a typical local core count better.
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.driver.memory", "4g")
    )

    if args.mode == "stream":
        # Pull the Kafka connector from Maven at runtime.
        # The version must match the Scala/Spark version in the environment.
        # spark-sql-kafka-0-10 supports the Kafka 0.10+ protocol (all modern
        # Kafka brokers). The "_2.12" suffix is the Scala binary version.
        builder = builder.config(
            "spark.jars.packages",
            "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0",
        )

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    try:
        if args.mode == "batch":
            print("[batch] Starting batch feature engineering job...\n")
            run_batch(spark)
            print("\n[batch] Job complete.")
        else:
            print("[stream] Starting Spark Structured Streaming job...\n")
            run_streaming(spark, args.kafka_bootstrap)
    finally:
        spark.stop()
