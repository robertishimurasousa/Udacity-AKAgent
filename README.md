# UDA-Hub

UDA-Hub is a proposed multi-agent support system for CultPass, built for Udacity's *Advanced Agentic AI Techniques* course. This repository contains the course starter, the project's architecture proposal, and a 20-article knowledge base. The workflow in `agentic/workflow.py` is the course example. The UDA-Hub agents and acceptance tests are planned in the design documents.

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

[`03_agentic_app.ipynb`](03_agentic_app.ipynb) runs the starter chat example. It requires a local `OPENAI_API_KEY` to call the provider. Keep any local `.env`, the virtual environment, and generated databases out of version control.

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

The supervisor reports the next step and its reason. When the resolver can answer every detected intent, `resolve` includes a text response and citations. For guidance, the text quotes the selected article. `escalate` records that the case needs human handling.

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

## Starter material and attribution

The setup notebooks, sample workflow and utilities, data models, starter dependency list, and original seed data come from [Udacity's public course repository](https://github.com/udacity/cd14641-langgraph-multi-public/tree/70c4797e28153b765e3053c5f23cf54b88e1a38c/project). The original [Udacity license notice](LICENSE.md) accompanies that material. The first four articles are from the starter. The remaining 16 are project-authored examples for technical support, account management, billing, and reservations.
