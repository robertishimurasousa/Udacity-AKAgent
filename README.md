# UDA-Hub

UDA-Hub is a LangGraph support system for the fictional CultPass service. It classifies incoming tickets, routes them to the right source, and answers with citations or identifies why the ticket needs human review. The local workflow uses a 20-article knowledge base and read-only account and subscription lookups.

## Course and project

This is the **Autonomous Knowledge Agent** project from Udacity's **Advanced Agentic AI Techniques** course, part of the **Agentic AI Engineer with LangChain and LangGraph** Nanodegree. [Course](https://www.udacity.com/course/advanced-agentic-ai-techniques--cd14641) · [Nanodegree](https://www.udacity.com/course/agentic-ai-engineer-with-langchain-and-langgraph--nd901)

- [Architecture and workflow](agentic/design/README.md)
- [Rubric checks and acceptance scenarios](agentic/design/aceite.md)

## Local setup

Use Python 3.11. From the repository root, create a virtual environment and install the dependencies.

```sh
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

The commands above use macOS or Linux paths. On Windows, use `.venv\Scripts\python.exe`. In your notebook editor, select the interpreter from `.venv` and start the notebooks from the repository root so their relative paths resolve.

### Review the submission

From the directory containing this README, run:

```sh
.venv/bin/python -B validate_submission.py
```

The validator runs both setup notebooks, checks SQLite and the LangGraph routes, exercises the CLI with synthetic tickets, and runs the regression suite. It creates databases and logs in a temporary copy of the project. It reports the technical results for A01–A13 and exits with a nonzero status if a check fails. In a Git checkout, it also checks that generated files and local credentials stay outside the tracked submission. Udacity reviews the design and confirms the accepted submission format. [Acceptance matrix](agentic/design/aceite.md)

Add `--openai` to the command to verify the live provider with two synthetic requests. The default uses local resolution and exercises provider failures with simulated responses in the test suite.

### Create demo databases

1. Run [`01_external_db_setup.ipynb`](01_external_db_setup.ipynb) to create `data/external/cultpass.db` with the synthetic CultPass data.
2. Run [`02_core_db_setup.ipynb`](02_core_db_setup.ipynb) to load the 20 articles in [`cultpass_articles.jsonl`](data/external/cultpass_articles.jsonl) and create a sample ticket in `data/core/udahub.db`. The notebook requires at least 14 articles if you edit the dataset.

**Both notebooks recreate their database files when run.** Use a disposable checkout or back up any database whose contents you need to keep. Neither database setup notebook requires an API key.

### Model API credentials

The setup notebooks, local ticket workflow, and tests run without a model API key. For OpenAI review in the project demo, set `OPENAI_API_KEY` in the ignored `.env` file and run [03_agentic_app.py](03_agentic_app.py) with `--openai`. Set `OPENAI_MODEL` to override the default `gpt-4o-mini`. The adapter uses the OpenAI Responses API with a structured verdict and a 20-second client timeout. The reference environment uses Python 3.11, LangGraph 1.2.24, SQLite checkpointer 3.1.3 and OpenAI SDK 3.57.0. [Dependencies](requirements.txt)

The [starter chat notebook](03_agentic_app.ipynb) remains a course reference and uses the sample workflow in `agentic/workflow.py`. For a course-issued Vocareum key, that notebook uses the `OPENAI_BASE_URL` configured by Vocareum. [Vocareum Labs documentation](https://help.vocareum.com/en/articles/9341351-genai-in-vocareum-labs)

Keep credentials, the virtual environment, and generated databases local. Use the private reporting channel for security reports. [Security policy](SECURITY.md)

## Run the support demo

After setting up both databases, run the following commands from the repository root. Add `--openai` to a CLI command to enable live review. For a saved ticket, use a fresh turn ID to review a new answer. Reusing a completed turn ID replays its saved result.

```sh
.venv/bin/python 03_agentic_app.py --ticket-id demo-login --turn-id first \
  'I forgot my password. How can I sign in again?'

demo_customer=$(sqlite3 data/external/cultpass.db \
  "SELECT u.user_id FROM users u JOIN subscriptions s USING (user_id) WHERE u.is_blocked = 0 LIMIT 1;")
.venv/bin/python 03_agentic_app.py --ticket-id demo-plan --turn-id first \
  --demo-customer-id "$demo_customer" 'What is my subscription status and monthly quota?'

.venv/bin/python 03_agentic_app.py --ticket-id demo-unknown --turn-id first \
  'How do I assemble a telescope?'

jq -c 'select(.ticket_id == "demo-plan" and .turn_id == "first")' .logs/uda-hub.jsonl
```

The profile flag simulates the authenticated identity supplied by a support connector. The CLI checks that the profile exists in CultPass and owns the ticket before the workflow runs. A new ticket submitted by a valid profile is registered to that profile in UDA-Hub. Existing tickets require their owner. A new anonymous ticket can receive general guidance without session history or checkpoints. A real connector must supply identity from its authentication context.

The OpenAI reviewer receives the drafted answer, its article excerpts, and the structured facts needed for the answer. It returns an approval or rejection. The workflow escalates if the review rejects the draft or is unavailable. Missing knowledge escalates before model review.

The JSONL trace records classifications, routes, candidate article IDs and scores, tool statuses, provider verdicts, handoffs and final outcomes. Supply opaque ticket and turn IDs for correlation. Ticket bodies, responses, customer IDs and credentials stay outside the trace. A trace write failure is visible as `trace_status="unavailable"`.

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

The separate SQLite history file records the customer message, decision, and response with citations or handoff. The `memory.history` output contains the prior turns for that ticket. Repeating a completed turn ID in the same scope returns its stored outcome, including the provider verdict when recorded. Existing metadata-only rows remain available for context. When the workflow has a scoped history store, a storage failure returns `operational_error`. The reason is `history_unavailable` if the workflow cannot load the history or `history_write_failed` if recording the outcome fails. The demo exits with status 1 in both cases. After restoring storage, retry with the same ticket and turn IDs.

Explicit requests such as `Please answer step by step.` and `Please keep your answers concise.` update the format for future tickets from the same verified profile. The workflow redacts common credential and email patterns from stored messages. Use synthetic tickets for the local demo.

### Inspect a session checkpoint

For tickets processed with `--demo-customer-id`, the CLI stores native LangGraph checkpoints in `data/core/udahub_checkpoints.db`. It derives `thread_id` from the organization, profile, and ticket after checking ownership. Reuse the same ticket ID and a new turn ID to continue a session.

```sh
.venv/bin/python 03_agentic_app.py --ticket-id demo-booking --turn-id first \
  --demo-customer-id "$demo_customer" 'How do I reserve a spot for an event?'
.venv/bin/python 03_agentic_app.py --ticket-id demo-booking --turn-id second \
  --demo-customer-id "$demo_customer" 'What should I do next?'
.venv/bin/python 03_agentic_app.py --ticket-id demo-booking \
  --demo-customer-id "$demo_customer" --inspect
```

The inspection returns the current turn's message, route, evidence, and outcome where applicable. Conversation context comes from the scoped history. The graph clears the previous turn's transient state before classifying the next message. The CLI redacts common credential and email patterns before writing the checkpoint. A profile or ownership rejection exits with status 2. [LangGraph SQLite checkpointer](https://reference.langchain.com/python/langgraph.checkpoint.sqlite/SqliteSaver)

## Validation and delivery

- Run `.venv/bin/python -m unittest discover -s tests -v` from the repository root. The suite exercises SQLite retrieval, routing, identity boundaries, and grounded responses with synthetic data.
- The GitHub checkout keeps the project at the repository root. The upload archive places the same project inside `solution/`, as specified by the course template. Run the validator from the directory containing this README in either layout. [Udacity project template](https://github.com/udacity/cd14641-langgraph-multi-public/blob/70c4797e28153b765e3053c5f23cf54b88e1a38c/project/solution/README.md)

## Contribution language

Use English for all contributions to Udacity projects. This includes branch names, commit messages, issue and pull request titles and descriptions, issue and review comments, code identifiers and comments, notebook text, documentation, and submission materials.

## Starter material and attribution

The setup notebooks, sample workflow and utilities, data models, starter dependency list, and original seed data come from [Udacity's public course repository](https://github.com/udacity/cd14641-langgraph-multi-public/tree/70c4797e28153b765e3053c5f23cf54b88e1a38c/project). The original [Udacity license notice](LICENSE.md) accompanies that material. The first four articles are from the starter. The remaining 16 are project-authored examples for technical support, account management, billing, and reservations.

The GitHub repository also includes original examples and exercises from the four course lessons under `lessons/`. Those materials are reference material. The upload archive contains the UDA-Hub project files.
