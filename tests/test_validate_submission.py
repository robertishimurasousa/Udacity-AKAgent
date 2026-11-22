from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import unittest

from validate_submission import REQUIRED_FILES, CheckFailure, check_citations, check_diagram, check_layout


class SubmissionChecksTests(unittest.TestCase):
    def test_layout_requires_the_canonical_workflow_module(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in (REQUIRED_FILES | {"agentic/support_workflow.py"}) - {"agentic/workflow.py"}:
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            (root / ".env.example").write_text("OPENAI_API_KEY=\n", encoding="utf-8")
            with self.assertRaisesRegex(CheckFailure, r"agentic/workflow\.py"):
                check_layout(root)

    def test_layout_requires_reviewer_entry_point(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in REQUIRED_FILES - {"03_agentic_app.py"}:
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            with self.assertRaisesRegex(CheckFailure, "03_agentic_app.py"):
                check_layout(root)

    @unittest.skipUnless(shutil.which("git"), "Git is required to test tracked-file checks")
    def test_layout_rejects_tracked_credentials_without_printing_their_contents(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            for filename in REQUIRED_FILES:
                target = root / filename
                target.parent.mkdir(parents=True, exist_ok=True)
                target.touch()
            (root / ".env").write_text("OPENAI_API_KEY=synthetic-test-secret", encoding="utf-8")
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(["git", "add", "-f", ".env"], cwd=root, check=True)
            with self.assertRaisesRegex(CheckFailure, r"\.env") as error:
                check_layout(root)
            self.assertNotIn("synthetic-test-secret", str(error.exception))

    def test_citations_must_reference_real_articles_and_verbatim_excerpts(self):
        articles = {"login": "Use Forgot Password to recover access."}
        for citation in (
            {"article_id": "invented", "excerpt": "Use Forgot Password"},
            {"article_id": "login", "excerpt": "Change the account owner"},
            {"article_id": "login", "excerpt": ""},
        ):
            with self.subTest(citation=citation), self.assertRaises(CheckFailure):
                check_citations(articles, [citation])

    def test_architecture_check_detects_wrong_account_routing_in_the_diagram(self):
        path = Path(__file__).resolve().parents[1] / "agentic/design/README.md"
        correct = path.read_text(encoding="utf-8")
        check_diagram(correct)
        changed = correct.replace(
            'Supervisor -->|"Customer data"| Account',
            'Supervisor -->|"Customer data"| Resolve',
        )
        self.assertNotEqual(correct, changed)
        with self.assertRaisesRegex(CheckFailure, "diagram"):
            check_diagram(changed)
