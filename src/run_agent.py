"""CLI to generate personalized communication playbook."""
from __future__ import annotations

import argparse
from pathlib import Path

from policy_renewal_agent import PolicyRenewalAgent


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a policy renewal outreach playbook")
    parser.add_argument(
        "--customers",
        type=Path,
        default=Path("data/synthetic_policy_customers.csv"),
        help="Path to customer CSV dataset",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("comms/outreach_playbook.json"),
        help="Where to store the generated playbook",
    )
    args = parser.parse_args()

    agent = PolicyRenewalAgent()
    customers = agent.load_customers(args.customers)
    playbook = agent.plan_outreach(customers)
    agent.save_playbook(playbook, args.output)
    print(f"Saved {len(playbook)} outreach plans to {args.output}")


if __name__ == "__main__":
    main()
