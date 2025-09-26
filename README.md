# TCS AI Fridays Hackathon – Use Case 25

This repository contains a prototype "Policy Renewal Agent" that helps insurers deliver personalized, human-like renewal conversations. The solution is designed to engage customers beyond automated reminders and to improve retention of digital-first segments.

## Project Structure

- `src/generate_synthetic_data.py` – script to generate synthetic customer and policy data aligned with the use case.
- `src/policy_renewal_agent.py` – lightweight agent logic that blends segment strategies with channel-specific messaging templates.
- `src/run_agent.py` – command-line entry point that turns a dataset into a channel-ready outreach playbook.
- `src/policy_chatbot.py` – knowledge base and chatbot helpers that blend policy insights with customer context.
- `src/chatbot_cli.py` – interactive chatbot experience for Q&A on renewals.
- `configs/` – prompt templates and strategies used by the agent.
- `data/` – synthetic datasets (generated).
- `comms/` – personalized communication outputs produced by the agent.

## Getting Started

1. **Generate synthetic data** (optional if you want a different sample size):
   ```bash
   python src/generate_synthetic_data.py --size 200 --output data/synthetic_policy_customers.csv
   ```
2. **Create a personalized outreach playbook**:
   ```bash
   python src/run_agent.py --customers data/synthetic_policy_customers.csv --output comms/outreach_playbook.json
   ```

The resulting JSON file contains channel recommendations and tailored communication snippets for every customer.

3. **Launch the interactive renewal chatbot**:
   ```bash
   python src/chatbot_cli.py --customer-id CUST-00001
   ```
   The chatbot introduces the personalized communication plan for the chosen customer and supports follow-up questions about
   benefits, renewal steps, discounts, or policy specifics sourced from the product document knowledge base.

## Extending the Agent

- Plug the playbook into local SLMs (e.g., Llama-3.2-3b-it) or Azure-hosted models for final copy polishing.
- Replace or augment the templates and segment strategies with real product offers.
- Add feedback loops by capturing response outcomes and retraining the incentives matrix.

## Data Privacy

All data included in this repository is synthetically generated for demonstration purposes only and contains no personally identifiable information.
