"""
Reads data/transactions.csv and streams each row to the 'transactions' Kafka
topic as a JSON message at a sustained rate of 100 events per second.

Usage (from project root, with Kafka running):
    python data_pipeline/kafka/producer.py
    python data_pipeline/kafka/producer.py --file data/transactions.csv --rate 200
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer
from kafka.admin import KafkaAdminClient, NewTopic
from kafka.errors import KafkaError, TopicAlreadyExistsError

# ── Configuration ─────────────────────────────────────────────────────────────
KAFKA_BROKER = "localhost:9092"
TOPIC_NAME = "transactions"
DEFAULT_RATE = 100          # events per second
BATCH_SIZE = 100            # messages sent before each rate-limiting sleep
LOG_EVERY = 10_000          # print a progress line every N messages


# ── JSON serializer ───────────────────────────────────────────────────────────
# KafkaProducer requires bytes. This encoder converts the types pandas
# introduces (Timestamp, numpy int/float) that the stdlib json module rejects.
class _Encoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (pd.Timestamp, datetime)):
            return obj.isoformat()
        if hasattr(obj, "item"):          # numpy scalar → Python native
            return obj.item()
        return super().default(obj)


def _serialize(msg: dict) -> bytes:
    return json.dumps(msg, cls=_Encoder).encode("utf-8")


# ── Topic setup ───────────────────────────────────────────────────────────────
# Kafka auto-creates topics when a message is first produced, but creating it
# explicitly lets us control partition count (4 here — one per fraud pattern
# if a downstream consumer uses fraud_type as a routing key).
def _ensure_topic(broker: str, topic: str, partitions: int = 4) -> None:
    admin = KafkaAdminClient(bootstrap_servers=broker)
    try:
        admin.create_topics([NewTopic(name=topic, num_partitions=partitions, replication_factor=1)])
        print(f"Created topic '{topic}' with {partitions} partitions.")
    except TopicAlreadyExistsError:
        print(f"Topic '{topic}' already exists.")
    finally:
        admin.close()


# ── Producer factory ──────────────────────────────────────────────────────────
# acks="all" waits for the broker leader to confirm the write before
# proceeding — safe default for financial data even at this low throughput.
# retries=3 handles transient broker unavailability without crashing.
def _make_producer(broker: str) -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=broker,
        value_serializer=_serialize,
        key_serializer=lambda k: k.encode("utf-8") if k else None,
        acks="all",
        retries=3,
        linger_ms=10,       # small batching window to reduce broker round-trips
        compression_type="gzip",
    )


# ── Delivery error callback ───────────────────────────────────────────────────
# Called asynchronously by kafka-python when a send ultimately fails after
# retries. Logs the error without crashing the producer loop.
def _on_error(exc: KafkaError) -> None:
    print(f"[ERROR] Failed to deliver message: {exc}", file=sys.stderr)


# ── Main streaming loop ───────────────────────────────────────────────────────
def stream(csv_path: str, rate: int) -> None:
    path = Path(csv_path)
    if not path.exists():
        sys.exit(f"[ERROR] File not found: {csv_path}\n"
                 "Run data_pipeline/generators/transaction_generator.py first.")

    print(f"Loading {csv_path}...")
    df = pd.read_csv(csv_path, parse_dates=["timestamp"])
    total = len(df)
    print(f"Loaded {total:,} transactions. Streaming to '{TOPIC_NAME}' at {rate} events/sec.")
    print(f"Estimated stream duration: {total / rate / 60:.1f} minutes\n")

    _ensure_topic(KAFKA_BROKER, TOPIC_NAME)
    producer = _make_producer(KAFKA_BROKER)

    # Rate limiting: send BATCH_SIZE messages, then sleep long enough so the
    # wall-clock time for that batch equals BATCH_SIZE / rate seconds.
    batch_interval = BATCH_SIZE / rate      # target wall time per batch (seconds)
    sent = 0
    errors = 0
    stream_start = time.perf_counter()
    batch_start = stream_start

    try:
        for i, row in enumerate(df.itertuples(index=False), start=1):
            if i % BATCH_SIZE == 1:
                batch_start = time.perf_counter()

            msg = row._asdict()

            # user_id is the partition key: all transactions for the same
            # cardholder land on the same partition, enabling per-user
            # aggregations in downstream consumers without reshuffling.
            producer.send(
                TOPIC_NAME,
                key=msg["user_id"],
                value=msg,
            ).add_errback(_on_error)

            sent += 1

            # After every BATCH_SIZE messages, sleep off any remaining budget
            # in the 1-second window to maintain the target rate.
            if sent % BATCH_SIZE == 0:
                elapsed = time.perf_counter() - batch_start
                sleep_for = batch_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)

            if sent % LOG_EVERY == 0:
                elapsed_total = time.perf_counter() - stream_start
                actual_rate = sent / elapsed_total
                pct = sent / total * 100
                print(f"  Sent {sent:>7,} / {total:,}  ({pct:5.1f}%)  "
                      f"rate={actual_rate:.0f} msg/s  elapsed={elapsed_total:.0f}s")

        # Flush ensures all buffered messages are delivered before exit.
        producer.flush()

    except KeyboardInterrupt:
        print("\nInterrupted — flushing remaining messages...")
        producer.flush()

    finally:
        producer.close()
        elapsed_total = time.perf_counter() - stream_start
        actual_rate = sent / elapsed_total if elapsed_total > 0 else 0
        print(f"\nDone. Sent {sent:,} messages in {elapsed_total:.1f}s "
              f"(avg {actual_rate:.1f} msg/s, {errors} errors).")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream transactions CSV to Kafka.")
    parser.add_argument("--file", default="data/transactions.csv",
                        help="Path to transactions CSV (default: data/transactions.csv)")
    parser.add_argument("--rate", type=int, default=DEFAULT_RATE,
                        help="Events per second (default: 100)")
    args = parser.parse_args()

    stream(csv_path=args.file, rate=args.rate)
