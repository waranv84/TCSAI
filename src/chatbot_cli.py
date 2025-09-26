from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional

from llm_clients import (
    AzureChatClient,
    LLMCapabilityError,
    LLMConfigurationError,
    find_model_config,
)
from policy_chatbot import ChatResponse, PolicyKnowledgeBase, PolicyRenewalChatbot
from policy_renewal_agent import Customer, PolicyRenewalAgent


DEFAULT_MODEL_ID = "azure/genailab-maas-gpt-4o"


def _load_customers(agent: PolicyRenewalAgent, dataset_path: Path) -> Dict[str, Customer]:
    customers = agent.load_customers(dataset_path)
    return {customer.customer_id: customer for customer in customers}


def _system_prompt(customer: Customer) -> str:
    return (
        "You are an empathetic insurance policy renewal assistant. Blend the provided "
        "customer profile, knowledge base insights, and recommended outreach messaging "
        "to deliver clear, concise answers. Always reference relevant benefits, renewal "
        "steps, or incentives that encourage the customer to renew while keeping a warm, "
        "supportive tone."
    )


def _compose_intro_prompt(customer: Customer, intro: ChatResponse) -> str:
    details = [
        f"Customer name: {customer.name}",
        f"Segment: {customer.segment}",
        f"Policy type: {customer.policy_type}",
        f"Premium: ${customer.premium:,.2f}",
        f"Renewal date: {customer.renewal_date}",
        "Craft a welcoming opening message that summarises why renewing is valuable and offers help with next steps.",
        "Use the reference outreach message to stay aligned with the recommended tone.",
        f"Reference outreach message:\n{intro.text}",
    ]
    return "\n".join(details)


def _compose_llm_user_message(
    customer: Customer,
    question: str,
    base_response: ChatResponse,
) -> str:
    context_lines = [
        f"Customer: {customer.name}",
        f"Segment: {customer.segment}",
        f"Policy: {customer.policy_type}",
        f"Premium: ${customer.premium:,.2f}",
        f"Renewal date: {customer.renewal_date}",
        f"Customer question: {question}",
        "Incorporate the structured suggestions below into a natural, conversational reply that directly answers the question.",
        f"Rule-based assistant guidance:\n{base_response.text}",
    ]
    if base_response.suggested_prompts:
        prompts = "\n".join(f"- {prompt}" for prompt in base_response.suggested_prompts)
        context_lines.append("Potential follow-up topics to optionally mention:\n" + prompts)
    return "\n".join(context_lines)


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

    llm_client: Optional[AzureChatClient] = None
    conversation: List[Dict[str, str]] = []
    system_message = _system_prompt(customer)

    try:
        model_config = find_model_config(DEFAULT_MODEL_ID)
        llm_client = AzureChatClient(model_config)
        conversation.append({"role": "system", "content": system_message})
        intro_response = chatbot.intro(customer)
        intro_prompt = _compose_intro_prompt(customer, intro_response)
        conversation.append({"role": "user", "content": intro_prompt})
        intro_text = llm_client.generate(conversation)
        conversation.append({"role": "assistant", "content": intro_text})
        print(f"\nUsing model: {DEFAULT_MODEL_ID}")
        print(f"Agent: {intro_text}\n")
    except (LLMConfigurationError, LLMCapabilityError, RuntimeError) as error:
        llm_client = None
        conversation = []
        print(f"\n[warning] LLM unavailable ({error}). Falling back to rule-based responses.")
        intro_response = chatbot.intro(customer)
        print("\n" + intro_response.text + "\n")
        if intro_response.suggested_prompts:
            print("Try asking:")
            for prompt in intro_response.suggested_prompts:
                print(f"  - {prompt}")
            print()

    if not conversation:
        conversation.append({"role": "system", "content": system_message})

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        response = chatbot.answer(customer, user_input)
        if user_input and llm_client:
            conversation.append(
                {"role": "user", "content": _compose_llm_user_message(customer, user_input, response)}
            )
            try:
                llm_text = llm_client.generate(conversation)
                conversation.append({"role": "assistant", "content": llm_text})
                print(f"Agent: {llm_text}\n")
            except (LLMCapabilityError, LLMConfigurationError, RuntimeError) as error:
                print(f"[warning] LLM response failed ({error}). Using rule-based reply instead.")
                llm_client = None
                print(f"Agent: {response.text}\n")
                if response.suggested_prompts:
                    print("You can also ask:")
                    for prompt in response.suggested_prompts:
                        print(f"  - {prompt}")
                    print()
        else:
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
