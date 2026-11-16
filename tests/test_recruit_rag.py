import json
import os
import unittest
from datetime import date
from unittest import mock

os.environ["RECRUIT_RAG_EMBEDDER"] = "hash"      # no model download needed
os.environ["RECRUIT_RAG_STORE"] = "memory"

from recruitrag.factory import create_pipeline
from recruitrag.job_parser import parse_job
from recruitrag.privacy import Anonymizer
from recruitrag.resume_parser import compute_experience_years, parse_resume
from recruitrag.skills import extract_skills

TODAY = date(2026, 11, 14)
RES = "sample_data/resumes"
JOBS = "sample_data/jobs"


def read(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


class ParserTests(unittest.TestCase):
    def test_skill_boundaries(self):
        s = extract_skills("PostgreSQL, JavaScript and Java; k8s; C++; node.js")
        self.assertIn("PostgreSQL", s)
        self.assertNotIn("SQL", s)            # not matched inside PostgreSQL
        self.assertIn("Java", s)
        self.assertIn("JavaScript", s)
        self.assertIn("Kubernetes", s)
        self.assertIn("C++", s)
        self.assertIn("Node.js", s)
        self.assertEqual(extract_skills("I love javascript"), ["JavaScript"])   # 'Java' must not fire

    def test_profile_fields(self):
        p, sections = parse_resume(read(f"{RES}/aarav_mehta.txt"), "aarav_mehta.txt", TODAY)
        self.assertEqual(p.name, "Aarav Mehta")
        self.assertEqual(p.email, "aarav.mehta@example.com")
        self.assertEqual(p.phone, "+91 98100 11122")
        self.assertEqual(p.candidate_id, "aarav_mehta")
        self.assertIn("Kafka", p.skills)
        self.assertEqual(set(["summary", "experience", "education", "skills"]), set(p.sections_found))
        self.assertTrue(p.education[0].startswith("B.Tech"))

    def test_experience_years_exclude_education(self):
        p, _ = parse_resume(read(f"{RES}/aarav_mehta.txt"), "a.txt", TODAY)
        # Mar 2021-now (68 mo) + Jun 2018-Feb 2021 (32 mo) = 100 mo; 2014-2018 degree must not count
        self.assertAlmostEqual(p.years_experience, 8.3, places=1)
        m, _ = parse_resume(read(f"{RES}/meera_iyer.txt"), "m.txt", TODAY)
        self.assertAlmostEqual(m.years_experience, 0.5, places=1)

    def test_overlapping_ranges_are_merged(self):
        self.assertEqual(compute_experience_years("2018 - 2020\n2019 - 2021", TODAY), 3.0)

    def test_phone_ignores_date_ranges(self):
        p, _ = parse_resume("Jane Doe\njane@x.com\n\nEXPERIENCE\nDev, X\n2019 - 2021\n", "jane.txt", TODAY)
        self.assertEqual(p.phone, "")

    def test_job_parser(self):
        j = parse_job(read(f"{JOBS}/ml_engineer_llm.txt"))
        self.assertEqual(j.min_years, 3.0)
        self.assertIn("Python", j.required_skills)
        self.assertIn("RAG", j.required_skills)
        self.assertEqual(set(j.preferred_skills), {"Spark", "Docker"})
        self.assertNotIn("Spark", j.required_skills)


class NameDetectionTests(unittest.TestCase):
    CASES = [
        ("name first line", "Aarav Mehta\naarav@x.com | +91 98100 11122\nSUMMARY\nBackend dev", "Aarav Mehta"),
        ("heading INTRODUCTION first", "INTRODUCTION\nYUKTA JOSHI\nEmail: a@b.com\nLocation: New Delhi, India", "Yukta Joshi"),
        ("RESUME heading first", "RESUME\nJohn Smith\njohn@x.com", "John Smith"),
        ("Curriculum Vitae heading", "Curriculum Vitae\nPriya Nair\npriya@x.com", "Priya Nair"),
        ("markdown bold heading", "# **Rahul Sharma**\nrahul@x.com", "Rahul Sharma"),
        ("docling image comment", "<!-- image -->\n## Neha Kumari\nneha@x.com", "Neha Kumari"),
        ("Name: label", "Name: Arjun Verma\nPhone: +91 98765 43210", "Arjun Verma"),
        ("Resume of X", "Resume of Sana Khan\nsana@x.com", "Sana Khan"),
        ("X - Resume", "Rahul Sharma - Resume\nrahul@x.com", "Rahul Sharma"),
        ("Resume - X", "Resume - Rahul Sharma\nrahul@x.com", "Rahul Sharma"),
        ("name | email | phone on one line", "John Smith | john@x.com | +1 415 555 0142\nSKILLS\nPython", "John Smith"),
        ("credentials after comma", "Priya Nair, MBA\npriya@x.com", "Priya Nair"),
        ("name split over two lines", "YUKTA\nJOSHI\nemail: a@b.com", "Yukta Joshi"),
        ("several headings before name", "Personal Details\nContact\nMaria Garcia-Lopez\nm@x.com", "Maria Garcia-Lopez"),
        ("job title above name", "SENIOR BACKEND ENGINEER\nAarav Mehta\naarav@x.com", "Aarav Mehta"),
        ("single-word name", "Madonna\nEmail: m@x.com\nLocation: New Delhi, India", "Madonna"),
        ("initials", "J. R. R. Tolkien\nj@x.com", "J. R. R. Tolkien"),
        ("title and apostrophe", "Dr. Anita D'Souza\nanita@x.com", "Dr. Anita D'Souza"),
        ("markdown table row", "| Aarav Mehta | aarav@x.com |\n|---|---|", "Aarav Mehta"),
        ("sentence is not a name", "Seeking a role in AI\nAarav Mehta\naarav@x.com", "Aarav Mehta"),
    ]

    def test_name_table(self):
        for label, text, expected in self.CASES:
            with self.subTest(label):
                p, _ = parse_resume(text, "file.pdf", TODAY)
                self.assertEqual(p.name, expected)

    def test_filename_fallback_when_no_name_in_text(self):
        p, _ = parse_resume("Email: a@b.com\n\nSKILLS\nPython", "Yukta_Joshi_CV__20261114_083645_0000.pdf", TODAY)
        self.assertEqual(p.name, "Yukta Joshi")

    def test_cv_with_intro_heading_keeps_experience(self):
        cv = ("INTRODUCTION\nYUKTA JOSHI\nEmail: a@b.com | +91 78349 13525\n\nEXPERIENCE\n"
              "Intern | June 2026 - Present\n")
        p, _ = parse_resume(cv, "cv.pdf", TODAY)
        self.assertEqual(p.name, "Yukta Joshi")
        self.assertAlmostEqual(p.years_experience, 0.4, places=1)


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pipe = create_pipeline("memory")
        cls.pipe.indexer.today = TODAY
        cls.n = cls.pipe.ingest(sorted(
            os.path.join(RES, f) for f in os.listdir(RES)))

    def test_indexed_all(self):
        self.assertEqual(self.n, 7)
        self.assertEqual(len(self.pipe.store.get_profiles()), 7)

    def test_reindex_is_idempotent(self):
        before = len(self.pipe.store.chunks)
        self.pipe.ingest([os.path.join(RES, "liam_chen.txt")])
        self.assertEqual(len(self.pipe.store.chunks), before)

    def test_named_candidate_query_is_scoped(self):
        hits = self.pipe.retriever.search("What are Priya's skills?", top_k=5)
        self.assertTrue(hits)
        self.assertEqual({h.candidate_id for h in hits}, {"priya_nair"})

    def test_per_candidate_diversity(self):
        hits = self.pipe.retriever.search("Python AWS Docker", top_k=6, max_per_candidate=1)
        ids = [h.candidate_id for h in hits]
        self.assertEqual(len(ids), len(set(ids)))

    def test_retrieval_metrics(self):
        questions = json.loads(read("sample_data/eval/sample_questions.json"))
        r = self.pipe.evaluate_retrieval(questions, k=3)
        self.assertEqual(r["n_questions"], 8)
        self.assertGreaterEqual(r["hit_rate"], 0.85, r)
        self.assertGreaterEqual(r["mrr"], 0.7, r)

    def test_match_ml_job(self):
        job, matches = self.pipe.match_job(read(f"{JOBS}/ml_engineer_llm.txt"), top_n=7)
        self.assertEqual(matches[0].candidate_id, "priya_nair")
        self.assertEqual(matches[0].missing_required, [])
        self.assertGreater(matches[0].overall, matches[-1].overall)
        self.assertTrue(all(0 <= m.overall <= 100 for m in matches))

    def test_match_backend_job(self):
        _, matches = self.pipe.match_job(read(f"{JOBS}/backend_engineer.txt"), top_n=3)
        self.assertEqual(matches[0].candidate_id, "aarav_mehta")
        self.assertIn("Kafka", matches[0].matched_preferred)
        self.assertIn("Django", [s for s in matches[0].matched_required])

    def test_experience_gap_is_scored(self):
        _, matches = self.pipe.match_job(read(f"{JOBS}/backend_engineer.txt"), top_n=7)
        meera = next(m for m in matches if m.candidate_id == "meera_iyer")
        self.assertLess(meera.experience_score, 0.2)

    def test_anonymized_matches_hide_identity(self):
        _, matches = self.pipe.match_job(read(f"{JOBS}/backend_engineer.txt"), top_n=3, anonymize=True)
        for m in matches:
            self.assertTrue(m.name.startswith("Candidate "))
            blob = json.dumps(m.to_dict())
            for secret in ("Aarav", "Mehta", "example.com", "98100"):
                self.assertNotIn(secret, blob)

    def test_ask_uses_llm_with_masked_context(self):
        captured = {}

        def fake_llm(system_message, user_message):
            captured["msg"] = user_message
            return "Candidate C has both Kubernetes and Terraform experience."

        with mock.patch("recruitrag.answering.ask_llm", fake_llm):
            ans = self.pipe.ask("Who knows Kubernetes and Terraform?", anonymize=True)
        self.assertEqual(ans.text, "Candidate C has both Kubernetes and Terraform experience.")
        self.assertIn("<question>", captured["msg"])
        for secret in ("liam.chen@example.com", "Liam", "8123 4567"):
            self.assertNotIn(secret, captured["msg"])
        self.assertTrue(all(h.candidate_id.startswith("Candidate") for h in ans.hits))

    def test_no_hits_skips_llm(self):
        with mock.patch("recruitrag.answering.ask_llm", side_effect=AssertionError("must not call")):
            self.assertIn("couldn't find", self.pipe.responder.respond("x", []))


class RobustnessTests(unittest.TestCase):
    """Regression tests for failures seen with a 1B local model."""

    @classmethod
    def setUpClass(cls):
        cls.pipe = create_pipeline("memory")
        cls.pipe.indexer.today = TODAY
        cls.pipe.ingest(sorted(os.path.join(RES, f) for f in os.listdir(RES)))

    def test_requirements_text_excludes_company_blurb(self):
        j = parse_job(read(f"{JOBS}/ml_engineer_llm.txt"))
        self.assertNotIn("recruitment platform", j.requirements_text)
        self.assertIn("Strong Python", j.requirements_text)

    def test_skill_facts_are_exact(self):
        captured = {}

        def fake_llm(system_message, user_message):
            captured["msg"] = user_message
            return "Liam Chen has both Kubernetes and Terraform skills."

        with mock.patch("recruitrag.answering.ask_llm", fake_llm):
            self.pipe.ask("Who has experience with Kubernetes and Terraform?")
        facts = captured["msg"].split("<facts>")[1].split("</facts>")[0]
        self.assertIn("Liam Chen", facts)
        self.assertNotIn("Aarav", facts)          # Aarav has Kubernetes but not Terraform

    def test_skill_facts_respect_blind_mode(self):
        captured = {}
        with mock.patch("recruitrag.answering.ask_llm",
                        lambda system_message, user_message: captured.setdefault("m", user_message) and "A long enough stub answer for the checker."):
            self.pipe.ask("Who has Kubernetes and Terraform?", anonymize=True)
        self.assertNotIn("Liam", captured["m"])
        self.assertIn("Candidate", captured["m"])

    def test_echoed_header_falls_back_to_facts(self):
        with mock.patch("recruitrag.answering.ask_llm",
                        lambda system_message, user_message: "Aarav Mehta | Section: experience"):
            ans = self.pipe.ask("Who has experience with Kubernetes and Terraform?")
        self.assertIn("Liam Chen", ans.text)
        self.assertNotIn("| Section:", ans.text)

    def test_explain_is_deterministic_and_never_calls_llm(self):
        with mock.patch("recruitrag.answering.ask_llm", side_effect=AssertionError("no LLM")):
            _, matches = self.pipe.match_job(read(f"{JOBS}/ml_engineer_llm.txt"), top_n=7, explain=True)
        aarav = next(m for m in matches if m.candidate_id == "aarav_mehta")
        self.assertIn("missing", aarav.explanation)
        self.assertIn("Machine Learning", aarav.explanation)        # reported as MISSING, never as a strength
        self.assertNotIn("strong foundation", aarav.explanation.lower())

    def test_calibration_rows(self):
        rows = self.pipe.matcher.raw_scores(read(f"{JOBS}/ml_engineer_llm.txt"))
        self.assertEqual(len(rows), 7)
        self.assertEqual(rows, sorted(rows, key=lambda r: r[1], reverse=True))


class NewModuleTests(unittest.TestCase):
    def test_tag_text(self):
        from recruitrag.xml_tags import tag_text
        self.assertEqual(tag_text("x <why> because\nit works </why> y", "why"), "because\nit works")
        self.assertEqual(tag_text("<VERDICT>correct</VERDICT>", "verdict"), "correct")
        self.assertIsNone(tag_text("no tags here", "why"))
        self.assertIsNone(tag_text("<why>unclosed", "why"))

    def test_judge_parses_verdicts(self):
        from recruitrag.evaluation import AnswerJudge
        cases = [
            ("<why>same</why><verdict>correct</verdict>", True),
            ("<why>differs</why><verdict>incorrect</verdict>", False),   # 'incorrect' must not count as correct
            ("I think it is fine", False),                                # unparseable -> not correct
        ]
        for reply, expected in cases:
            with self.subTest(reply), mock.patch("recruitrag.evaluation.ask_llm", return_value=reply):
                v = AnswerJudge().grade("q", "a", "ref")
                self.assertEqual(v.correct, expected)

    def test_llm_wrapper_returns_text_and_wraps_failures(self):
        import sys, types
        from recruitrag import llm
        ok = types.SimpleNamespace(chat=lambda **kw: {"message": {"content": "hello"}})
        with mock.patch.dict(sys.modules, {"ollama": ok}):
            self.assertEqual(llm.ask_llm("sys", "user"), "hello")

        def boom(**kw):
            raise ConnectionError("refused")
        down = types.SimpleNamespace(chat=boom)
        with mock.patch.dict(sys.modules, {"ollama": down}):
            with self.assertRaises(RuntimeError) as ctx:
                llm.ask_llm("sys", "user")
        self.assertIn("Ollama", str(ctx.exception))
        self.assertIn("ollama pull", str(ctx.exception))

    def test_model_env_override(self):
        from recruitrag import llm
        with mock.patch.dict(os.environ, {"RECRUIT_RAG_MODEL": "llama3.2:3b"}):
            self.assertEqual(llm.model_name(), "llama3.2:3b")

    def test_implementations_satisfy_contracts(self):
        from recruitrag import contracts
        from recruitrag.answering import LLMResponder
        from recruitrag.evaluation import AnswerJudge
        from recruitrag.indexing import ResumeIndexer
        from recruitrag.retrieval import HybridRetriever
        from recruitrag.store_lance import LanceStore
        from recruitrag.store_memory import MemoryStore
        pipe = create_pipeline("memory")
        self.assertIsInstance(pipe.store, contracts.Store)
        self.assertIsInstance(pipe.indexer, contracts.Indexer)
        self.assertIsInstance(pipe.retriever, contracts.Searcher)
        self.assertIsInstance(pipe.responder, contracts.Responder)
        self.assertIsInstance(AnswerJudge(), contracts.Grader)
        # LanceStore needs the lancedb package to instantiate, so check its surface on the class
        for method in ("reset", "add_chunks", "delete_candidate", "add_profiles", "get_profiles", "embed", "search"):
            self.assertTrue(callable(getattr(LanceStore, method)), method)
            self.assertTrue(callable(getattr(MemoryStore, method)), method)


class AnonymizerTests(unittest.TestCase):
    def test_masks_name_email_phone_links(self):
        p, _ = parse_resume(read(f"{RES}/sofia_ramirez.txt"), "sofia_ramirez.txt", TODAY)
        out = Anonymizer([p]).mask_text(
            "Sofia Ramirez (sofia.ramirez@example.com, +1 415 555 0142) github.com/sofia")
        for secret in ("Sofia", "Ramirez", "example.com", "555", "github"):
            self.assertNotIn(secret, out)
        self.assertIn("Candidate A", out)
        self.assertIn("[EMAIL]", out)
        self.assertIn("[PHONE]", out)

    def test_job_intro_is_not_a_requirement(self):
        j = parse_job("Data Role\nWe run a recruitment platform.\nRequirements\n- Python\n")
        self.assertEqual(j.required_skills, ["Python"])


if __name__ == "__main__":
    unittest.main()
