"""
Generates 500,000 synthetic labeled transactions for XGBoost fraud model training.
~2% fraud rate distributed equally across four fraud patterns.
Output: data/transactions.csv
"""

import os
import uuid
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from faker import Faker

# ── Reproducibility ───────────────────────────────────────────────────────────
SEED = 42
fake = Faker()
Faker.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ── Volume constants ──────────────────────────────────────────────────────────
TOTAL = 500_000
FRAUD_RATE = 0.02
N_FRAUD = int(TOTAL * FRAUD_RATE)    # 10,000 fraud transactions
N_NORMAL = TOTAL - N_FRAUD           # 490,000 legitimate
N_PER_PATTERN = N_FRAUD // 4         # 2,500 per fraud pattern

# ── Date window (full calendar year of transaction history) ───────────────────
END_DATE = datetime(2025, 12, 31)
START_DATE = END_DATE - timedelta(days=365)

# ── Merchant categories ───────────────────────────────────────────────────────
NORMAL_CATEGORIES = [
    "grocery", "restaurant", "gas_station", "retail", "pharmacy",
    "entertainment", "travel", "subscription", "utilities", "healthcare",
]
# Cash-equivalent or high-risk categories rarely used by legitimate customers
HIGH_RISK_CATEGORIES = [
    "pawn_shop", "wire_transfer", "crypto_exchange", "money_order", "check_cashing",
]


# ── User pool ─────────────────────────────────────────────────────────────────
# Each cardholder has a fixed home city. Normal purchases are made in that city;
# foreign-city fraud deviates from it. 50k users across 500k transactions gives
# a realistic ~10 transactions per user on average.
N_USERS = 50_000
print(f"Building user pool ({N_USERS:,} cardholders)...")
USER_POOL = [
    {
        "user_id": f"USR{i:06d}",
        "name": fake.name(),
        "home_city": fake.city(),
        "home_state": fake.state_abbr(),
    }
    for i in range(N_USERS)
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _random_ts(start=START_DATE, end=END_DATE):
    """Uniform random timestamp within the training window."""
    delta_s = int((end - start).total_seconds())
    return start + timedelta(seconds=random.randint(0, delta_s))


# Weights concentrate transactions in waking hours (8 AM – 10 PM) to match
# the observed distribution of legitimate card activity.
_HOUR_WEIGHTS = [1] * 6 + [2] * 2 + [5] * 13 + [3] * 2 + [1] * 1


def _daytime_hour():
    return random.choices(range(24), weights=_HOUR_WEIGHTS)[0]


def _make_txn(user, ts, amount, category, city, state, country, is_fraud, fraud_type):
    """Assemble a single transaction record."""
    return {
        "transaction_id": str(uuid.uuid4()),
        "timestamp": ts,
        "user_id": user["user_id"],
        "cardholder_name": user["name"],
        "merchant_name": fake.company(),
        "merchant_category": category,
        "amount": round(float(amount), 2),
        "city": city,
        "state": state,
        "country": country,
        "hour_of_day": ts.hour,
        "day_of_week": ts.weekday(),   # 0 = Monday … 6 = Sunday
        "is_fraud": is_fraud,
        "fraud_type": fraud_type,
    }


# ─────────────────────────────────────────────────────────────────────────────
# NORMAL TRANSACTIONS
# Log-normal amount distribution (median ~$33) with a $1,500 cap reflects
# everyday card spend without fat-tail outliers polluting the legitimate class.
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_NORMAL:,} normal transactions...")
normal_rows = []
for _ in range(N_NORMAL):
    user = random.choice(USER_POOL)
    ts = _random_ts().replace(hour=_daytime_hour(), minute=random.randint(0, 59))
    amount = min(np.random.lognormal(mean=3.5, sigma=1.0), 1_500)
    normal_rows.append(_make_txn(
        user, ts, amount,
        category=random.choice(NORMAL_CATEGORIES),
        city=user["home_city"], state=user["home_state"], country="US",
        is_fraud=0, fraud_type="none",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD PATTERN 1: VELOCITY BURST
# A stolen card is used for a rapid flurry of 5–10 purchases within minutes
# before the cardholder notices and freezes the card. Each burst is tied to one
# user_id so the model can learn the per-user frequency spike as a signal.
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_PER_PATTERN:,} velocity-burst fraud transactions...")
velocity_rows = []
generated = 0
while generated < N_PER_PATTERN:
    user = random.choice(USER_POOL)
    burst_size = random.randint(5, 10)
    burst_start = _random_ts().replace(hour=random.randint(0, 23))
    for j in range(burst_size):
        # Space transactions 10 seconds – 2 minutes apart within the burst
        ts = burst_start + timedelta(seconds=j * random.randint(10, 120))
        amount = round(random.uniform(10, 300), 2)
        velocity_rows.append(_make_txn(
            user, ts, amount,
            category=random.choice(NORMAL_CATEGORIES),
            city=user["home_city"], state=user["home_state"], country="US",
            is_fraud=1, fraud_type="velocity_burst",
        ))
        generated += 1
        if generated >= N_PER_PATTERN:
            break


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD PATTERN 2: LARGE AMOUNT ANOMALY
# Single card-not-present transactions dramatically above normal spend —
# consistent with fraudsters purchasing high-value goods (electronics, gift
# cards) or initiating large transfers before detection.
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_PER_PATTERN:,} large-amount fraud transactions...")
large_rows = []
for _ in range(N_PER_PATTERN):
    user = random.choice(USER_POOL)
    ts = _random_ts().replace(hour=_daytime_hour())
    amount = round(random.uniform(2_000, 15_000), 2)
    large_rows.append(_make_txn(
        user, ts, amount,
        category=random.choice(NORMAL_CATEGORIES + HIGH_RISK_CATEGORIES),
        city=user["home_city"], state=user["home_state"], country="US",
        is_fraud=1, fraud_type="large_amount",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD PATTERN 3: FOREIGN CITY
# Transaction city doesn't match the cardholder's home city — the geo-velocity
# anomaly that arises from physical card theft or account takeover used
# remotely. The while-loop guarantees the foreign city is always distinct.
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_PER_PATTERN:,} foreign-city fraud transactions...")
foreign_rows = []
for _ in range(N_PER_PATTERN):
    user = random.choice(USER_POOL)
    ts = _random_ts().replace(hour=_daytime_hour())
    amount = round(np.random.lognormal(mean=4.5, sigma=1.0), 2)

    foreign_city = fake.city()
    while foreign_city == user["home_city"]:
        foreign_city = fake.city()

    foreign_rows.append(_make_txn(
        user, ts, amount,
        category=random.choice(NORMAL_CATEGORIES),
        city=foreign_city, state=fake.state_abbr(), country="US",
        is_fraud=1, fraud_type="foreign_city",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# FRAUD PATTERN 4: HIGH-RISK MERCHANT AT ODD HOURS
# Transactions at cash-equivalent merchants (pawn shops, wire transfer, crypto)
# between midnight and 4:59 AM. The combination of category risk + off-hours
# timing is a strong signal of money-mule activity and account takeover.
# ─────────────────────────────────────────────────────────────────────────────
print(f"Generating {N_PER_PATTERN:,} high-risk/odd-hours fraud transactions...")
odd_rows = []
for _ in range(N_PER_PATTERN):
    user = random.choice(USER_POOL)
    ts = _random_ts().replace(hour=random.randint(0, 4), minute=random.randint(0, 59))
    amount = round(random.uniform(200, 5_000), 2)
    odd_rows.append(_make_txn(
        user, ts, amount,
        category=random.choice(HIGH_RISK_CATEGORIES),
        city=user["home_city"], state=user["home_state"], country="US",
        is_fraud=1, fraud_type="high_risk_odd_hours",
    ))


# ─────────────────────────────────────────────────────────────────────────────
# ASSEMBLE, SORT BY TIME, AND SAVE
# Chronological order mimics how the model will see data in production;
# it also makes train/test splitting by date straightforward.
# ─────────────────────────────────────────────────────────────────────────────
print("Assembling dataset...")
all_rows = normal_rows + velocity_rows + large_rows + foreign_rows + odd_rows
random.shuffle(all_rows)

df = pd.DataFrame(all_rows)
df["timestamp"] = pd.to_datetime(df["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

os.makedirs("data", exist_ok=True)
output_path = "data/transactions.csv"
df.to_csv(output_path, index=False)

# ── Summary ───────────────────────────────────────────────────────────────────
total = len(df)
fraud_count = df["is_fraud"].sum()
print(f"\nSaved to {output_path}")
print(f"  Total        : {total:,}")
print(f"  Legitimate   : {total - fraud_count:,}  ({(total - fraud_count) / total * 100:.1f}%)")
print(f"  Fraud        : {fraud_count:,}  ({fraud_count / total * 100:.1f}%)")
print("\nFraud breakdown:")
print(df[df["is_fraud"] == 1]["fraud_type"].value_counts().to_string())
print("\nAmount stats by class:")
print(df.groupby("is_fraud")["amount"].describe().round(2).to_string())
