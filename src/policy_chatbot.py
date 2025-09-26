"""Interactive chatbot utilities for the policy renewal use case."""
from __future__ import annotations

import difflib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

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
            "Ask me about benefits, features, pricing, renewal steps or anything else you need clarified."
        )
        return "\n".join(intro_lines)

    def intro(self, customer: Customer) -> ChatResponse:
        message = self.agent.generate_message(customer)
        intro_text = self._compose_intro(customer)
        prompt_suggestions = [
            "What benefits do I keep if I renew?",
            "How do I finish the renewal steps?",
            "Are there payment or discount options?",
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
        if any(keyword in lowered for keyword in ["benefit", "value", "cover"]):
            responses.extend(self._benefits_and_features(customer))
        if any(keyword in lowered for keyword in ["feature", "technology", "service"]):
            responses.extend(self._benefits_and_features(customer))
        if any(keyword in lowered for keyword in ["step", "process", "renew", "complete", "finish"]):
            responses.extend(self._renewal_guidance(customer))
        if any(keyword in lowered for keyword in ["payment", "discount", "offer", "incentive", "price", "premium"]):
            incentive = self._segment_incentive(customer)
            if incentive:
                responses.append(incentive)
            responses.append(
                f"We can review flexible billing schedules so the ${customer.premium:,.2f} premium fits your budget."
            )
        knowledge_hit = self.knowledge_base.search(customer.policy_type, lowered)
        if knowledge_hit:
            responses.append(knowledge_hit)

        if not responses:
            summary = self.knowledge_base.get_summary(customer.policy_type)
            if summary:
                responses.append(summary)
            responses.append(
                "You can ask about coverage, renewal steps, pricing or request to connect with a human advisor."
            )

        follow_ups: List[str] = []
        if "renew" not in lowered:
            follow_ups.append("Could you outline the renewal steps?")
        if "benefit" not in lowered and "feature" not in lowered:
            follow_ups.append("What are the standout benefits?")
        if "discount" not in lowered and "payment" not in lowered:
            follow_ups.append("Do you have any loyalty credits or payment support?")

        return ChatResponse(text="\n\n".join(responses), suggested_prompts=follow_ups)


__all__ = ["PolicyKnowledgeBase", "PolicyRenewalChatbot", "ChatResponse"]
