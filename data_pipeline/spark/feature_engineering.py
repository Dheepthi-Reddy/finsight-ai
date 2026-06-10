"""
PySpark feature engineering job for the FinSight fraud detection model.
Reads data/transactions.csv, computes seven derived features, and writes the
enriched dataset to data/features.parquet for XGBoost training.

Run from the project root (Spark runs locally, no cluster required):
    python data_pipeline/spark/feature_engineering.py
"""

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


def build_features(spark: SparkSession) -> None:
    # ── Load ──────────────────────────────────────────────────────────────────
    print(f"Reading {INPUT_CSV}...")
    df = (spark.read
          .option("header", "true")
          .option("timestampFormat", "yyyy-MM-dd HH:mm:ss")
          .schema(SCHEMA)
          .csv(INPUT_CSV))

    n = df.count()
    print(f"Loaded {n:,} transactions.\n")

    # Unix epoch seconds — needed as a numeric ordering key for rangeBetween
    # windows. TimestampType cannot be used directly in rangeBetween.
    df = df.withColumn("ts_unix", F.unix_timestamp("timestamp"))

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 1: amount_log
    # Transaction amounts are heavily right-skewed (log-normal distribution).
    # Without transformation, a $15,000 fraud transaction would dwarf $30
    # grocery purchases and dominate every split. Log-compressing the scale
    # lets the model treat proportional differences as equal: $10→$100 looks
    # the same as $100→$1,000. log1p handles the edge case of $0.00 auths.
    # ─────────────────────────────────────────────────────────────────────────
    print("Computing amount_log...")
    df = df.withColumn("amount_log", F.log1p(F.col("amount")))

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 2: is_night
    # Legitimate card activity drops sharply after 11 PM; the synthetic
    # high-risk-merchant/odd-hours pattern concentrates entirely in [0, 5).
    # A binary flag creates a hard decision boundary the model can exploit
    # without needing to learn the curved relationship from hour_of_day alone.
    # ─────────────────────────────────────────────────────────────────────────
    print("Computing is_night...")
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
    print("Computing is_weekend...")
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
    print("Computing amount_x_risk...")
    risk_pairs = [item for kv in MERCHANT_RISK.items()
                  for item in (F.lit(kv[0]), F.lit(kv[1]))]
    risk_map = F.create_map(*risk_pairs)

    df = (df
          .withColumn("merchant_risk", risk_map[F.col("merchant_category")])
          .withColumn("amount_x_risk", F.col("amount") * F.col("merchant_risk")))

    # ─────────────────────────────────────────────────────────────────────────
    # FEATURE 5: account_age_bucket
    # Newly opened or newly compromised accounts are exploited before the
    # cardholder establishes behavioral baselines or before fraud monitoring
    # systems have enough history to flag anomalies. We proxy "account creation
    # date" with each user's earliest observed transaction in the dataset, then
    # bucket the resulting age into three risk tiers:
    #   0 = new         (< 90 days)   highest risk
    #   1 = mid         (< 365 days)  moderate risk
    #   2 = established (≥ 365 days)  baseline risk
    # ─────────────────────────────────────────────────────────────────────────
    print("Computing account_age_bucket...")
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
    # FEATURES 6 & 7: txn_count_1h, amount_sum_1h
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
    # ─────────────────────────────────────────────────────────────────────────
    print("Computing txn_count_1h and amount_sum_1h...")
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
    print(f"\nWriting to {OUTPUT_PARQUET}...")
    (df.write
       .mode("overwrite")
       .option("compression", "snappy")
       .parquet(OUTPUT_PARQUET))

    # ── Validation stats (read back from parquet — avoids recomputing window fns)
    print("Done. Reading back for validation stats...\n")
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


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if not os.path.exists(INPUT_CSV):
        sys.exit(
            f"[ERROR] {INPUT_CSV} not found.\n"
            "Run data_pipeline/generators/transaction_generator.py first."
        )

    spark = (SparkSession.builder
             .appName("finsight-feature-engineering")
             .master("local[*]")
             # Default of 200 shuffle partitions causes ~190 empty partitions
             # on a local 500k-row dataset — 8 matches local core count better.
             .config("spark.sql.shuffle.partitions", "8")
             .config("spark.driver.memory", "4g")
             .getOrCreate())

    spark.sparkContext.setLogLevel("WARN")

    try:
        build_features(spark)
    finally:
        spark.stop()
