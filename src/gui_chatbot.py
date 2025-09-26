"""Streamlit UI for the policy renewal assistant.

The interface layers a graphical chat experience on top of the existing
rule-based policy chatbot and augments responses with Azure-hosted large
language models.  Users can pick an LLM for each session, explore customer
records, and interact with the assistant through a modern chat surface.
"""
from __future__ import annotations

import csv
import io
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import streamlit as st

from llm_clients import (
    AzureChatClient,
    LLMCapabilityError,
    LLMConfigurationError,
    MODEL_CATALOGUE,
    find_model_config,
)
from policy_chatbot import ChatResponse, PolicyKnowledgeBase, PolicyRenewalChatbot
from policy_renewal_agent import Customer, PolicyRenewalAgent


DEFAULT_DATASET = Path("data/synthetic_policy_customers.csv")


def _read_customers_from_buffer(buffer: io.BytesIO) -> List[Customer]:
    """Parse uploaded CSV content into ``Customer`` records."""

    buffer.seek(0)
    text = buffer.read().decode("utf-8")
    rows: List[Customer] = []
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        rows.append(
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
    return rows


@lru_cache(maxsize=4)
def _load_default_customers(path: Path) -> List[Customer]:
    agent = PolicyRenewalAgent()
    return agent.load_customers(path)


def _customer_lookup(customers: Iterable[Customer]) -> Dict[str, Customer]:
    return {customer.customer_id: customer for customer in customers}


def _render_customer_summary(customer: Customer, agent: PolicyRenewalAgent) -> None:
    st.subheader("Customer & policy overview")
    details = asdict(customer)
    st.markdown(
        "\n".join(
            f"**{key.replace('_', ' ').title()}:** {value}" for key, value in details.items()
        )
    )
    st.divider()
    st.subheader("Personalized outreach message")
    st.write(agent.generate_message(customer))


def _combine_responses(
    base_response: ChatResponse,
    llm_text: Optional[str],
) -> Tuple[str, Optional[List[str]]]:
    if llm_text:
        return llm_text, base_response.suggested_prompts
    return base_response.text, base_response.suggested_prompts


def _llm_messages(customer: Customer, question: str, base_response: ChatResponse) -> List[Dict[str, str]]:
    context_lines = [
        f"Customer name: {customer.name}",
        f"Customer segment: {customer.segment}",
        f"Policy type: {customer.policy_type}",
        f"Premium: ${customer.premium:,.2f}",
        f"Renewal date: {customer.renewal_date}",
        f"Customer question: {question}",
        "Use the policy knowledge to encourage renewal while remaining concise and helpful.",
    ]
    return [
        {
            "role": "system",
            "content": (
                "You are a policy renewal assistant. Blend the provided knowledge base "
                "insights with empathetic tone. Always mention benefits, renewal steps "
                "or incentives when relevant."
            ),
        },
        {
            "role": "user",
            "content": (
                "\n".join(context_lines)
                + "\n\nRule-based assistant suggestions:\n"
                + base_response.text
            ),
        },
    ]


def _maybe_generate_llm_reply(
    model_id: str,
    customer: Customer,
    question: str,
    base_response: ChatResponse,
) -> Optional[str]:
    model_config = find_model_config(model_id)
    try:
        client = AzureChatClient(model_config)
        return client.generate(_llm_messages(customer, question, base_response))
    except (LLMCapabilityError, LLMConfigurationError, RuntimeError) as error:
        st.warning(str(error))
    return None


def _reset_conversation_state() -> None:
    st.session_state.pop("conversation", None)
    st.session_state.pop("intro_sent", None)
    st.session_state["session_active"] = True


def _ensure_conversation_state() -> None:
    st.session_state.setdefault("conversation", [])
    st.session_state.setdefault("intro_sent", False)
    st.session_state.setdefault("session_active", True)


def _append_message(role: str, content: str, prompts: Optional[List[str]] = None) -> None:
    st.session_state.conversation.append({
        "role": role,
        "content": content,
        "suggested_prompts": prompts or [],
    })


def main() -> None:
    st.set_page_config(page_title="Policy Renewal Assistant", layout="wide")
    st.title("Policy Renewal Chatbot")

    with st.sidebar:
        st.header("Session setup")
        def _format_model(mid: str) -> str:
            model = next(m for m in MODEL_CATALOGUE if m.model_id == mid)
            suffix = f" – {model.description}" if model.description else ""
            return f"{model.model_id}{suffix}"

        model_id = st.selectbox(
            "Model",
            options=[model.model_id for model in MODEL_CATALOGUE],
            format_func=_format_model,
        )

        uploaded = st.file_uploader("Upload customer CSV", type=["csv"])
        use_default = st.checkbox("Use bundled sample customers", value=uploaded is None)

        customers: List[Customer]
        agent = PolicyRenewalAgent()
        knowledge_base = PolicyKnowledgeBase()

        if uploaded and not use_default:
            customers = _read_customers_from_buffer(uploaded)  # type: ignore[arg-type]
        else:
            customers = _load_default_customers(DEFAULT_DATASET)

        lookup = _customer_lookup(customers)
        customer_ids = sorted(lookup)
        selected_customer = st.selectbox("Select customer ID", options=customer_ids)
        manual_customer = st.text_input("Or enter customer ID")

        current_customer = lookup.get(manual_customer) if manual_customer else lookup.get(selected_customer)
        if manual_customer and manual_customer not in lookup:
            st.error("Customer ID not found in dataset")
            current_customer = None

        if st.button("Start new session"):
            _reset_conversation_state()

    if not current_customer:
        st.info("Select a customer to begin the conversation.")
        return

    _ensure_conversation_state()

    chatbot = PolicyRenewalChatbot(agent=agent, knowledge_base=knowledge_base)

    if not st.session_state.intro_sent:
        intro = chatbot.intro(current_customer)
        _append_message("assistant", intro.text, intro.suggested_prompts)
        st.session_state.intro_sent = True

    left, right = st.columns([2, 1])

    with right:
        _render_customer_summary(current_customer, agent)

    with left:
        st.subheader("Conversation")
        for index, entry in enumerate(st.session_state.conversation):
            with st.chat_message(entry["role"]):
                st.markdown(entry["content"])
                if entry["role"] == "assistant" and entry["suggested_prompts"]:
                    st.caption("Suggested follow-ups:")
                    cols = st.columns(len(entry["suggested_prompts"]))
                    for idx, prompt in enumerate(entry["suggested_prompts"]):
                        if cols[idx].button(prompt, key=f"suggest_{index}_{idx}"):
                            st.session_state.pending_prompt = prompt
                            st.experimental_rerun()

        if not st.session_state.session_active:
            st.info("Session ended. Start a new session from the sidebar to continue.")
        else:
            user_prompt: Optional[str] = st.session_state.pop("pending_prompt", None)
            user_prompt = st.chat_input(
                "Ask about customer details, coverage, discounts, or renewal steps",
                key="chat_input",
            ) or user_prompt

            if user_prompt:
                _append_message("user", user_prompt)
                base_response = chatbot.answer(current_customer, user_prompt)
                llm_text = _maybe_generate_llm_reply(
                    model_id=model_id,
                    customer=current_customer,
                    question=user_prompt,
                    base_response=base_response,
                )
                content, prompts = _combine_responses(base_response, llm_text)
                _append_message("assistant", content, prompts)

        if st.button("End session"):
            st.session_state.session_active = False
            _append_message(
                "assistant",
                "Thanks for chatting. If you need anything else, start a new session from the sidebar!",
                [],
            )


if __name__ == "__main__":
    main()

