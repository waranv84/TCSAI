"""Command-line interface for the interactive policy renewal chatbot."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from policy_chatbot import PolicyKnowledgeBase, PolicyRenewalChatbot
from policy_renewal_agent import Customer, PolicyRenewalAgent


def _load_customers(agent: PolicyRenewalAgent, dataset_path: Path) -> Dict[str, Customer]:
    customers = agent.load_customers(dataset_path)
    return {customer.customer_id: customer for customer in customers}


def interactive_session() -> None:
    parser = argparse.ArgumentParser(description="Interact with the policy renewal chatbot")
    parser.add_argument(
        "--customers",
        type=Path,
        default=Path("data/synthetic_policy_customers.csv"),
        help="Path to the customer CSV dataset",
    )
    parser.add_argument(
        "--customer-id",
        type=str,
        help="Customer ID to start the chat with. If omitted you'll be prompted to choose.",
    )
    parser.add_argument(
        "--policy-docs",
        type=Path,
        default=Path("data/policy_documents.json"),
        help="Path to the product document knowledge base",
    )
    args = parser.parse_args()

    agent = PolicyRenewalAgent(product_document_path=args.policy_docs)
    knowledge_base = PolicyKnowledgeBase(args.policy_docs)
    customers = _load_customers(agent, args.customers)

    if not customers:
        raise SystemExit("No customers found in the dataset. Ensure the CSV is populated.")

    customer_id = args.customer_id
    if not customer_id or customer_id not in customers:
        print("Available customers:")
        for cid, record in list(customers.items())[:10]:
            print(f"  {cid}: {record.name} – {record.policy_type} (renews {record.renewal_date})")
        if len(customers) > 10:
            print("  ... (showing first 10) ...")
        customer_id = input("Enter the customer ID you want to engage: ").strip()
        if customer_id not in customers:
            raise SystemExit(f"Customer ID '{customer_id}' not found in dataset.")

    customer = customers[customer_id]
    chatbot = PolicyRenewalChatbot(agent, knowledge_base)
    intro_response = chatbot.intro(customer)
    print("\n" + intro_response.text + "\n")
    if intro_response.suggested_prompts:
        print("Try asking:")
        for prompt in intro_response.suggested_prompts:
            print(f"  - {prompt}")
        print()

    while True:
        user_input = input("You: ").strip()
        response = chatbot.answer(customer, user_input)
        print(f"Agent: {response.text}\n")
        if response.suggested_prompts:
            print("You can also ask:")
            for prompt in response.suggested_prompts:
                print(f"  - {prompt}")
            print()
        if user_input.lower() in PolicyRenewalChatbot.EXIT_COMMANDS:
            break


if __name__ == "__main__":
    interactive_session()
