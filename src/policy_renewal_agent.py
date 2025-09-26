"""Policy Renewal Agent for personalized customer engagement."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

TEMPLATE_PATH = Path("configs/channel_templates.json")
STRATEGY_PATH = Path("configs/segment_strategies.json")


@dataclass
class Customer:
    customer_id: str
    name: str
    age: int
    segment: str
    policy_type: str
    premium: float
    renewal_date: str
    last_interaction_channel: str
    churn_risk: float
    engagement_score: float
    preferred_channel: str
    lapse_reason: str


class PolicyRenewalAgent:
    """Agent that prepares tailored outreach strategies for renewals."""

    def __init__(self, template_path: Path = TEMPLATE_PATH, strategy_path: Path = STRATEGY_PATH) -> None:
        self.templates = self._load_json(template_path)
        self.strategies = self._load_json(strategy_path)

    @staticmethod
    def _load_json(path: Path) -> Dict:
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def load_customers(self, dataset_path: Path) -> List[Customer]:
        customers: List[Customer] = []
        with dataset_path.open("r", encoding="utf-8") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                customers.append(
                    Customer(
                        customer_id=row["customer_id"],
                        name=row["name"],
                        age=int(row["age"]),
                        segment=row["segment"],
                        policy_type=row["policy_type"],
                        premium=float(row["premium"]),
                        renewal_date=row["renewal_date"],
                        last_interaction_channel=row["last_interaction_channel"],
                        churn_risk=float(row["churn_risk"]),
                        engagement_score=float(row["engagement_score"]),
                        preferred_channel=row["preferred_channel"],
                        lapse_reason=row["lapse_reason"],
                    )
                )
        return customers

    def _select_template(self, channel: str) -> str:
        try:
            return self.templates[channel]
        except KeyError as exc:
            raise KeyError(f"No template configured for channel '{channel}'") from exc

    def _compose_context(self, customer: Customer) -> Dict[str, str]:
        segment_strategy = self.strategies.get(customer.segment, {})
        incentives = segment_strategy.get("incentives", "")
        digital_push = segment_strategy.get("digital_support", "")
        return {
            "customer_name": customer.name,
            "policy_type": customer.policy_type,
            "premium": f"${customer.premium:,.2f}",
            "renewal_date": customer.renewal_date,
            "incentives": incentives,
            "digital_support": digital_push,
            "lapse_reason": customer.lapse_reason,
            "segment": customer.segment,
        }

    def generate_message(self, customer: Customer, channel: Optional[str] = None) -> str:
        channel = channel or customer.preferred_channel
        template = self._select_template(channel)
        context = self._compose_context(customer)
        message = template.format(**context)
        return message

    def plan_outreach(self, customers: Iterable[Customer]) -> List[Dict[str, str]]:
        playbook: List[Dict[str, str]] = []
        for customer in customers:
            message = self.generate_message(customer)
            playbook.append(
                {
                    "customer_id": customer.customer_id,
                    "channel": customer.preferred_channel,
                    "message": message,
                    "churn_risk": f"{customer.churn_risk:.2f}",
                    "engagement_score": f"{customer.engagement_score:.2f}",
                }
            )
        return playbook

    def save_playbook(self, playbook: List[Dict[str, str]], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(playbook, file, indent=2)


__all__ = ["PolicyRenewalAgent", "Customer"]
