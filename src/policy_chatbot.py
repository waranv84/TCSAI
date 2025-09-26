"""Interactive chatbot utilities for the policy renewal use case."""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from policy_renewal_agent import Customer, PolicyRenewalAgent

POLICY_DOC_PATH = Path("data/policy_documents.json")


class PolicyKnowledgeBase:
    """Simple loader and retriever for policy product documents."""

    def __init__(self, document_path: Path = POLICY_DOC_PATH) -> None:
        self.document_path = document_path
        if not self.document_path.exists():
            raise FileNotFoundError(f"Knowledge base not found: {self.document_path}")
        with self.document_path.open("r", encoding="utf-8") as source:
            self.documents: Dict[str, Dict[str, object]] = json.load(source)

    def get_document(self, policy_type: str) -> Dict[str, object]:
        return self.documents.get(policy_type, {})

    def list_policies(self) -> Iterable[str]:
        return self.documents.keys()

    def get_list(self, policy_type: str, field: str) -> List[str]:
        """Return a list-based field from the document if present."""

        document = self.get_document(policy_type)
        values = document.get(field, []) if isinstance(document, dict) else []
        if isinstance(values, list):
            return [str(item) for item in values if isinstance(item, str)]
        return []

    def get_text(self, policy_type: str, field: str) -> str:
        """Return a text field from the document if present."""

        document = self.get_document(policy_type)
        value = document.get(field, "") if isinstance(document, dict) else ""
        return str(value) if isinstance(value, str) else ""

    def _best_faq_match(self, policy_type: str, question: str) -> Optional[str]:
        document = self.get_document(policy_type)
        faqs: Iterable[Dict[str, str]] = document.get("faqs", [])  # type: ignore[assignment]
        best_answer: Optional[str] = None
        best_score = 0.0
        for item in faqs:
            faq_question = item.get("question", "")
            answer = item.get("answer", "")
            if not faq_question or not answer:
                continue
            score = difflib.SequenceMatcher(
                None, faq_question.lower(), question.lower()
            ).ratio()
            if score > best_score:
                best_score = score
                best_answer = answer
        if best_score >= 0.45:
            return best_answer
        return None

    def get_summary(self, policy_type: str) -> str:
        document = self.get_document(policy_type)
        return document.get("overview", "") if isinstance(document, dict) else ""

    def search(self, policy_type: str, question: str) -> Optional[str]:
        """Return the best fitting FAQ answer for a question if available."""
        return self._best_faq_match(policy_type, question)


@dataclass
class ChatResponse:
    """Represents a chatbot reply with optional follow-up prompts."""

    text: str
    suggested_prompts: Optional[List[str]] = None


class PolicyRenewalChatbot:
    """Rule-based chatbot that personalizes responses with customer context."""

    EXIT_COMMANDS = {"quit", "exit", "bye", "close"}

    def __init__(
        self,
        agent: PolicyRenewalAgent,
        knowledge_base: PolicyKnowledgeBase,
    ) -> None:
        self.agent = agent
        self.knowledge_base = knowledge_base

    def _list_to_bullets(self, items: Iterable[str], label: str) -> Optional[str]:
        cleaned = [item.strip() for item in items if item]
        if not cleaned:
            return None
        joined = "; ".join(cleaned)
        return f"{label}: {joined}."

    @staticmethod
    def _contains_any(text: str, phrases: Sequence[str]) -> bool:
        return any(phrase in text for phrase in phrases)

    def _benefits_and_features(self, customer: Customer) -> List[str]:
        document = self.knowledge_base.get_document(customer.policy_type)
        responses: List[str] = []
        benefits = document.get("benefits")
        if isinstance(benefits, list):
            text = self._list_to_bullets(benefits[:3], "Key benefits")
            if text:
                responses.append(text)
        features = document.get("features")
        if isinstance(features, list):
            text = self._list_to_bullets(features[:3], "Standout features")
            if text:
                responses.append(text)
        return responses

    def _coverage_details(self, customer: Customer) -> List[str]:
        document = self.knowledge_base.get_document(customer.policy_type)
        snippets: List[str] = []
        coverage_keywords = {"cover", "coverage", "protect", "insured", "liability"}
        for section in ("benefits", "features"):
            items = document.get(section)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, str):
                    continue
                lowered_item = item.lower()
                if any(keyword in lowered_item for keyword in coverage_keywords):
                    text = item if item.endswith(".") else f"{item}."
                    snippets.append(text)
        if not snippets:
            overview = document.get("overview")
            if isinstance(overview, str) and overview:
                snippets.append(overview)
        return snippets[:3]

    def _renewal_guidance(self, customer: Customer) -> List[str]:
        document = self.knowledge_base.get_document(customer.policy_type)
        responses: List[str] = []
        steps = document.get("renewal_steps")
        if isinstance(steps, list) and steps:
            step_text = "; ".join(
                f"Step {idx + 1}: {step}" for idx, step in enumerate(steps[:4]) if step
            )
            responses.append(f"Renewal checklist — {step_text}.")
        support = document.get("support")
        if isinstance(support, str) and support:
            responses.append(support)
        return responses

    def _segment_incentive(self, customer: Customer) -> Optional[str]:
        strategy = self.agent.strategies.get(customer.segment, {})
        incentive = strategy.get("incentives") if isinstance(strategy, dict) else ""
        if incentive:
            return f"Because {customer.name} is in the {customer.segment} segment we can offer: {incentive}"
        return None

    def _discount_highlights(self, customer: Customer) -> List[str]:
        document = self.knowledge_base.get_document(customer.policy_type)
        snippets: List[str] = []
        discount_keywords = {"discount", "credit", "reward", "offer", "saving"}
        for section in ("benefits", "features", "renewal_steps"):
            items = document.get(section)
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, str):
                    continue
                lowered_item = item.lower()
                if any(keyword in lowered_item for keyword in discount_keywords):
                    text = item if item.endswith(".") else f"{item}."
                    snippets.append(text)
        incentive = self._segment_incentive(customer)
        if incentive:
            snippets.append(incentive)
        return snippets[:3]

    def _support_details(self, customer: Customer) -> List[str]:
        support = self.knowledge_base.get_text(customer.policy_type, "support")
        return [support] if support else []

    def _profile_overview(self, customer: Customer) -> str:
        details = [
            f"Customer ID: {customer.customer_id}",
            f"Name: {customer.name}",
            f"Segment: {customer.segment}",
            f"Policy type: {customer.policy_type}",
            f"Premium: ${customer.premium:,.2f}",
            f"Renewal date: {customer.renewal_date}",
            f"Preferred channel: {customer.preferred_channel}",
        ]
        return "Here are the key details I have on file:\n- " + "\n- ".join(details)

    def _customer_attribute_responses(self, customer: Customer, lowered_question: str) -> List[str]:
        responses: List[str] = []

        def add_response(phrases: Sequence[str], message: str) -> None:
            if self._contains_any(lowered_question, phrases):
                responses.append(message)

        add_response(
            ["my name", "customer name", "client name", "their name", "who is the customer"],
            f"The customer on this policy is {customer.name}.",
        )
        add_response(
            [
                "policy type",
                "type of policy",
                "policy do i have",
                "what policy am i",
                "policy name",
            ],
            f"You're currently covered by the {customer.policy_type} policy.",
        )
        add_response(
            [
                "premium",
                "payment amount",
                "how much do i pay",
                "monthly payment",
                "policy cost",
                "policy price",
            ],
            f"Your premium is ${customer.premium:,.2f}.",
        )
        add_response(
            [
                "renewal date",
                "when do i renew",
                "renewal due",
                "due date",
                "policy expire",
                "renew by",
                "expiration date",
            ],
            f"The renewal date on file is {customer.renewal_date}.",
        )
        add_response(
            ["my segment", "customer segment", "segment am i", "customer tier", "profile segment"],
            f"{customer.name} belongs to the {customer.segment} segment.",
        )
        add_response(
            ["my age", "customer age", "how old am i"],
            f"Age on file: {customer.age}.",
        )
        add_response(
            ["customer id", "id number", "account number", "policy id"],
            f"The customer ID is {customer.customer_id}.",
        )
        add_response(
            [
                "preferred channel",
                "preferred contact",
                "best way to reach",
                "contact method",
                "contact channel",
            ],
            f"Their preferred contact channel is {customer.preferred_channel}.",
        )
        add_response(
            ["last interaction", "last spoke", "last contacted", "last channel"],
            f"The last interaction was over {customer.last_interaction_channel}.",
        )
        add_response(
            ["churn risk", "risk of leaving", "likelihood of cancellation", "retention risk"],
            f"Their current churn risk score is {customer.churn_risk:.0%}.",
        )
        add_response(
            ["engagement score", "engagement level", "engagement rating"],
            f"The engagement score sits at {customer.engagement_score:.0%}.",
        )
        add_response(
            ["lapse reason", "why lapsed", "reason for lapse"],
            f"Last lapse reason recorded: {customer.lapse_reason}.",
        )
        if self._contains_any(
            lowered_question,
            [
                "customer details",
                "policy details",
                "customer information",
                "policy information",
                "tell me about this customer",
                "profile overview",
                "what details do you have",
            ],
        ):
            responses.append(self._profile_overview(customer))

        return responses

    def _compose_intro(self, customer: Customer) -> str:
        document_summary = self.knowledge_base.get_summary(customer.policy_type)
        intro_lines = [
            f"👋 Hi {customer.name}, I'm your renewal guide for the {customer.policy_type} policy.",
            f"Current premium: ${customer.premium:,.2f} and renewal date: {customer.renewal_date}.",
        ]
        if document_summary:
            intro_lines.append(document_summary)
        incentive = self._segment_incentive(customer)
        if incentive:
            intro_lines.append(incentive)
        intro_lines.append(
            "Ask me about your customer details, policy benefits, coverage, discounts, renewal steps or anything else you need clarified."
        )
        return "\n".join(intro_lines)

    def intro(self, customer: Customer) -> ChatResponse:
        message = self.agent.generate_message(customer)
        intro_text = self._compose_intro(customer)
        prompt_suggestions = [
            "What benefits do I keep if I renew?",
            "How do I finish the renewal steps?",
            "Can you remind me of my premium and renewal date?",
        ]
        combined = f"{intro_text}\n\nHere is a personalized outreach message we prepared:\n{message}"
        return ChatResponse(text=combined, suggested_prompts=prompt_suggestions)

    def answer(self, customer: Customer, question: str) -> ChatResponse:
        normalized = question.strip()
        if not normalized:
            return ChatResponse(text="Could you share a bit more about what you'd like to know?")
        lowered = normalized.lower()
        if lowered in self.EXIT_COMMANDS:
            return ChatResponse(text="Thanks for chatting. Feel free to reach out any time!", suggested_prompts=[])

        responses: List[str] = []
        responses.extend(self._customer_attribute_responses(customer, lowered))

        if self._contains_any(lowered, ["benefit", "value"]):
            responses.extend(self._benefits_and_features(customer))
        if self._contains_any(lowered, ["feature", "technology", "service"]):
            responses.extend(self._benefits_and_features(customer))

        coverage_requested = self._contains_any(lowered, ["coverage", "cover", "protected"])
        if coverage_requested:
            coverage_snippets = self._coverage_details(customer)
            if coverage_snippets:
                responses.extend(coverage_snippets)
            else:
                responses.append(
                    "I don't have additional coverage specifics handy, but I can connect you with a specialist to review limits."
                )

        if self._contains_any(lowered, ["step", "process", "renew", "complete", "finish"]):
            responses.extend(self._renewal_guidance(customer))

        if self._contains_any(lowered, ["payment", "discount", "offer", "incentive", "price", "premium", "savings"]):
            discount_snippets = self._discount_highlights(customer)
            if discount_snippets:
                responses.extend(discount_snippets)
            else:
                responses.append(
                    "I don't see specific discount notes, but I'm happy to review payment options with you."
                )
            responses.append(
                f"We can review flexible billing schedules so the ${customer.premium:,.2f} premium fits your budget."
            )

        if self._contains_any(lowered, ["support", "help", "contact", "assist", "service team"]):
            support_details = self._support_details(customer)
            if support_details:
                responses.extend(support_details)
            else:
                responses.append(
                    "I don't have a direct support line listed, but I can arrange for a representative to reach out."
                )

        knowledge_hit = self.knowledge_base.search(customer.policy_type, lowered)
        if knowledge_hit:
            responses.append(knowledge_hit)

        if not responses:
            summary = self.knowledge_base.get_summary(customer.policy_type)
            if summary:
                responses.append(summary)
            responses.append(
                "I couldn't find that detail in my notes, but I can share coverage basics or bring a specialist into the conversation."
            )

        deduped: List[str] = []
        seen = set()
        for response in responses:
            if response and response not in seen:
                deduped.append(response)
                seen.add(response)

        follow_ups: List[str] = []
        if not self._contains_any(lowered, ["renew"]):
            follow_ups.append("Could you outline the renewal steps?")
        if not self._contains_any(lowered, ["benefit", "feature"]):
            follow_ups.append("What are the standout benefits?")
        if not self._contains_any(lowered, ["discount", "payment", "premium", "savings"]):
            follow_ups.append("Do you have any loyalty credits or payment support?")
        if not self._contains_any(lowered, ["renewal date", "renew by", "expire", "due date"]):
            follow_ups.append("When is my renewal due?")
        if not self._contains_any(lowered, ["premium", "payment"]):
            follow_ups.append("Remind me of my premium.")

        return ChatResponse(text="\n\n".join(deduped), suggested_prompts=follow_ups)


__all__ = ["PolicyKnowledgeBase", "PolicyRenewalChatbot", "ChatResponse"]
