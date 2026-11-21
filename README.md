# UDA-Hub

UDA-Hub is a proposed multi-agent support system for CultPass, built for Udacity's *Advanced Agentic AI Techniques* course. This baseline contains the course starter and the project's architecture proposal. The workflow in `agentic/workflow.py` is the course example. The UDA-Hub agents, acceptance tests, and final submission package are planned in the design documents.

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
2. Before running [`02_core_db_setup.ipynb`](02_core_db_setup.ipynb), expand `data/external/cultpass_articles.jsonl` to at least 14 articles. The starter ships with four. The notebook checks this minimum before loading the articles and creating the sample ticket in `data/core/udahub.db`.

**Both notebooks recreate their database files when run.** Use a disposable checkout or back up any database whose contents you need to keep. Neither database setup notebook requires an API key.

[`03_agentic_app.ipynb`](03_agentic_app.ipynb) runs the starter chat example. It requires a local `OPENAI_API_KEY` to call the provider. Keep any local `.env`, the virtual environment, and generated databases out of version control.

## Starter material and attribution

The setup notebooks, sample workflow and utilities, data models, synthetic fixtures, and starter dependency list come from [Udacity's public course repository](https://github.com/udacity/cd14641-langgraph-multi-public/tree/70c4797e28153b765e3053c5f23cf54b88e1a38c/project). The original [Udacity license notice](LICENSE.md) accompanies that material. The architecture and acceptance scenarios describe this project's proposed implementation.
