# UDA-Hub

UDA-Hub is a LangGraph support system for the fictional CultPass service. It classifies incoming tickets, routes them to the right source, and answers with citations or identifies why the ticket needs human review. The local workflow uses a 20-article knowledge base and read-only account and subscription lookups.

## Course and project

This is the **Autonomous Knowledge Agent** project from Udacity's **Advanced Agentic AI Techniques** course, part of the **Agentic AI Engineer with LangChain and LangGraph** Nanodegree. [Course](https://www.udacity.com/course/advanced-agentic-ai-techniques--cd14641) · [Nanodegree](https://www.udacity.com/course/agentic-ai-engineer-with-langchain-and-langgraph--nd901)

- [Architecture and workflow](agentic/design/README.md)
- [Acceptance scenarios](agentic/design/aceite.md)

## Local setup

Use Python 3.11. From the repository root, create a virtual environment and install the starter dependencies.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The commands above use macOS or Linux paths. On Windows, use `.venv\Scripts\python.exe`. In your notebook editor, select the interpreter from `.venv` and start the notebooks from the repository root so their relative paths resolve.

1. Run [`01_external_db_setup.ipynb`](01_external_db_setup.ipynb) to create `data/external/cultpass.db` with the synthetic CultPass data.
2. Run [`02_core_db_setup.ipynb`](02_core_db_setup.ipynb) to load the 20 articles in [`cultpass_articles.jsonl`](data/external/cultpass_articles.jsonl) and create a sample ticket in `data/core/udahub.db`. The notebook requires at least 14 articles if you edit the dataset.

**Both notebooks recreate their database files when run.** Use a disposable checkout or back up any database whose contents you need to keep. Neither database setup notebook requires an API key.

### Model API credentials

The setup notebooks, local ticket workflow, and tests run without a model API key. The [starter chat notebook](03_agentic_app.ipynb) calls the OpenAI API through the course example in `agentic/workflow.py`.

For that notebook, add your OpenAI API key to `OPENAI_API_KEY` in the ignored `.env` file at the repository root. The notebook loads the file with `python-dotenv`. For a course-issued Vocareum key, use the key together with the `OPENAI_BASE_URL` configured by Vocareum. [Vocareum Labs documentation](https://help.vocareum.com/en/articles/9341351-genai-in-vocareum-labs)

Keep credentials, the virtual environment, and generated databases local. Use the private reporting channel for security reports. [Security policy](SECURITY.md)

## Ticket routing

After database setup, pass a ticket to the project's LangGraph workflow. The initial classifier uses rules for common English CultPass requests.

```python
from agentic.support_workflow import build_support_graph

graph = build_support_graph("data/core/udahub.db")
result = graph.invoke({
    "account_id": "cultpass",
    "text": "I forgot my password. How can I sign in again?",
    "channel": "chat",
    "urgency": "normal",
})
print(result["decision"])
print(result.get("response"))
```

The supervisor reports the next step and its reason. When the resolver can answer every detected intent, `resolve` includes a text response and citations. For guidance, the text quotes the selected article. `escalate` includes a handoff summary with the reason, topics, and checks performed.

To enable customer-specific answers, pass `cultpass_database_path="data/external/cultpass.db"` to `build_support_graph`. The caller must supply a `verified_customer` object with the account ID and external user ID from its trusted identity context. The workflow checks the account scope before looking up customer data. The resolver uses verified account facts for personalized statements and cites the relevant article. With only the knowledge database configured, a personalized ticket still returns `lookup_account` as the next step.

## Knowledge retrieval

After database setup, use the LangGraph retrieval subgraph with search terms grouped by intent. It reads the account's articles from SQLite without provider credentials.

```python
from agentic.agents.knowledge import build_knowledge_graph

graph = build_knowledge_graph("data/core/udahub.db")
result = graph.invoke({
    "account_id": "cultpass",
    "intents": [
        {"intent_id": "login", "terms": ["password", "email"]},
        {"intent_id": "booking", "terms": ["event", "reservation"]},
    ],
})["retrieval"]
```

The result contains up to three ranked candidates per intent and a status of `ready`, `insufficient_evidence`, or `unavailable`. Run the routing and retrieval tests with `.venv/bin/python -m unittest discover -s tests -v`.

## Handoff and memory

The handoff is part of the workflow output and can be reviewed locally. Delivery to a support queue is a later step.

Enable history with `memory_database_path="data/core/udahub_history.db"` when building the graph. Supply `verified_customer`, `ticket_id`, and a distinct `turn_id` for each new turn. After resolving a ticket, a follow-up such as `What should I do next?` can reuse its last resolved intents. The workflow can also recall recent resolved topics from the same customer's other tickets.

The initial memory store records outcome and classified intents in a separate SQLite file. Repeating the same turn ID leaves one stored record. The history omits ticket text and response bodies. Storage failures appear as `memory_status="unavailable"` in the result.

## Validation and delivery

- Run `.venv/bin/python -m unittest discover -s tests -v` from the repository root. The suite exercises SQLite retrieval, routing, identity boundaries, and grounded responses with synthetic data.
- Use the [acceptance scenarios](agentic/design/aceite.md) to track the academic rubric. Check the setup and demo from a clean checkout before submitting the repository. The Udacity instructions mention a `solution/` folder. Acceptance of the repository-root layout is pending confirmation.
- Scope and review evidence are tracked in the issue associated with each change. New behavior starts with a failing test. Changes are reviewed in a PR against the tested commit, and the issue records validation and integration. The milestone tracks assessment separately from local implementation.

## Starter material and attribution

The setup notebooks, sample workflow and utilities, data models, starter dependency list, and original seed data come from [Udacity's public course repository](https://github.com/udacity/cd14641-langgraph-multi-public/tree/70c4797e28153b765e3053c5f23cf54b88e1a38c/project). The original [Udacity license notice](LICENSE.md) accompanies that material. The first four articles are from the starter. The remaining 16 are project-authored examples for technical support, account management, billing, and reservations.
