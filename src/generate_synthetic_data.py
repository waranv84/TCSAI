"""Utility for generating synthetic policy renewal data."""
from __future__ import annotations

import argparse
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List

import csv

SEGMENTS = [
    "Young Families",
    "Mid-Life Planners",
    "Pre-Retirees",
    "Small Business Owners",
]

PRODUCTS = [
    "Comprehensive Auto",
    "Home Shield",
    "LifeSecure 20",
    "Business Guard",
]

CHANNELS = ["Email", "SMS", "Phone", "In-App"]

REASONS_FOR_LAPSE = [
    "Premium too high",
    "Switched providers",
    "Financial hardship",
    "Forgot to renew",
    "Policy no longer needed",
]

@dataclass
class CustomerRecord:
    customer_id: str
    name: str
    age: int
    segment: str
    policy_type: str
    premium: float
    renewal_date: date
    last_interaction_channel: str
    churn_risk: float
    engagement_score: float
    preferred_channel: str
    lapse_reason: str

    def to_row(self) -> List[str]:
        data = asdict(self)
        data["renewal_date"] = self.renewal_date.isoformat()
        data["premium"] = f"{self.premium:.2f}"
        data["churn_risk"] = f"{self.churn_risk:.2f}"
        data["engagement_score"] = f"{self.engagement_score:.2f}"
        return list(data.values())


def _random_name() -> str:
    first_names = ["Avery", "Jordan", "Noah", "Liam", "Emma", "Maya", "Sofia", "Ethan", "Kai", "Zoe"]
    last_names = ["Patel", "Garcia", "Johnson", "O'Neal", "Chen", "Khan", "D'Souza", "Williams", "Lopez", "Nakamura"]
    return f"{random.choice(first_names)} {random.choice(last_names)}"


def _generate_record(idx: int) -> CustomerRecord:
    today = date.today()
    renewal_window = random.randint(-60, 90)
    return CustomerRecord(
        customer_id=f"CUST-{idx:05d}",
        name=_random_name(),
        age=random.randint(24, 72),
        segment=random.choice(SEGMENTS),
        policy_type=random.choice(PRODUCTS),
        premium=random.uniform(250.0, 2400.0),
        renewal_date=today + timedelta(days=renewal_window),
        last_interaction_channel=random.choice(CHANNELS),
        churn_risk=random.uniform(0.1, 0.95),
        engagement_score=random.uniform(0.2, 0.98),
        preferred_channel=random.choice(CHANNELS),
        lapse_reason=random.choice(REASONS_FOR_LAPSE),
    )


def generate_dataset(size: int) -> List[CustomerRecord]:
    return [_generate_record(i) for i in range(1, size + 1)]


def write_csv(records: List[CustomerRecord], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        "customer_id",
        "name",
        "age",
        "segment",
        "policy_type",
        "premium",
        "renewal_date",
        "last_interaction_channel",
        "churn_risk",
        "engagement_score",
        "preferred_channel",
        "lapse_reason",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        for record in records:
            writer.writerow(record.to_row())


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic policy renewal data")
    parser.add_argument("--size", type=int, default=200, help="Number of synthetic customer records")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/synthetic_policy_customers.csv"),
        help="Destination CSV path",
    )
    args = parser.parse_args()

    random.seed(42)
    records = generate_dataset(args.size)
    write_csv(records, args.output)
    print(f"Generated {len(records)} customer records at {args.output}")


if __name__ == "__main__":
    main()
