"""Offline regression checks with deterministic test embeddings and fake model responses."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage
from langchain_core.runnables import RunnableLambda

from lab.config import ROOT, AzureSettings
from lab.retrieval import FaissStore, SearchHit, answer_question, load_pdf_chunks


class TestEmbeddings(Embeddings):
    def embed_query(self, text):
        text = text.lower()
        return [float(text.count(word)) for word in ("hotel", "alcohol", "leave")] + [0.1]

    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]


class LocalTests(unittest.TestCase):
    def setUp(self):
        self.embeddings = TestEmbeddings()
        self.documents = [Document(page_content="hotel hotel", metadata={"source": "brochure", "page": 1}),
                          Document(page_content="alcohol policy", metadata={"source": "manual", "page": 2})]
        self.store = FaissStore.build(self.documents, self.embeddings, model_id="offline-embedding")

    def test_cosine_ranking_filter_and_limits(self):
        hits = self.store.search("hotel", k=99)
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].document.metadata["source"], "brochure")
        self.assertGreater(hits[0].score, 0.99)
        self.assertEqual(len(self.store.search("hotel", minimum_score=0.9)), 1)
        self.assertEqual(self.store.search("leave", minimum_score=0.9), [])
        for kwargs in [{"question": ""}, {"question": "hotel", "k": 0},
                       {"question": "hotel", "minimum_score": 1.1}]:
            with self.assertRaises(ValueError):
                self.store.search(**kwargs)

    def test_save_load_roundtrip_and_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            self.store.save(directory)
            self.assertFalse((Path(directory) / "index.pkl").exists())
            loaded = FaissStore.load(directory, self.embeddings, expected_model="offline-embedding")
            self.assertEqual(loaded.search("hotel")[0].document, self.documents[0])
            with self.assertRaisesRegex(ValueError, "model"):
                FaissStore.load(directory, self.embeddings, expected_model="different-model")
            binary = Path(directory) / "index.faiss"
            binary.write_bytes(binary.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "checksum"):
                FaissStore.load(directory, self.embeddings, expected_model="offline-embedding")

    def test_pdf_chunking_has_page_metadata(self):
        chunks = load_pdf_chunks(ROOT / "assets/new_york_brochure.pdf", chunk_size=300, overlap=60)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(c.metadata["page"] == 1 and len(c.page_content) <= 300 for c in chunks))
        self.assertTrue(any("Manhattan Hotel" in c.page_content for c in chunks))
        with self.assertRaises(ValueError):
            load_pdf_chunks(ROOT / "assets/new_york_brochure.pdf", chunk_size=100, overlap=100)

    def test_empty_retrieval_skips_chat_and_context_has_sources(self):
        with patch("lab.config.get_chat_model", side_effect=AssertionError("No model call expected")):
            self.assertIn("don't know", answer_question("unrelated", []))
        seen = []
        def fake(prompt):
            seen.append(prompt.to_string())
            return AIMessage(content="The source names a hotel. [Source 1]")
        result = answer_question("Which hotel?", [SearchHit(self.documents[0], 1.0)], RunnableLambda(fake))
        self.assertIn("[Source 1]", result)
        self.assertIn("brochure, page 1", seen[0])

    def test_config_errors_and_key_repr(self):
        with patch.dict(os.environ, {}, clear=True), patch("dotenv.load_dotenv"):
            with self.assertRaisesRegex(ValueError, "AZURE_OPENAI"):
                AzureSettings.load()
        settings = AzureSettings("https://offline.invalid/", "offline-test-key", "test-deployment")
        self.assertNotIn("offline-test-key", repr(settings))

    def test_actual_lcel_notebook_composition(self):
        def fake(prompt):
            text = prompt.to_string()
            if "Extract insurance intent" in text:
                return AIMessage(content='{"category":"auto","constraints":{"premium":"low"}}')
            if "Identify the language" in text:
                return AIMessage(content="English")
            if "Extract the stated loan" in text:
                return AIMessage(content='{"purpose":"car","constraints":{"tenure_years":3}}')
            return AIMessage(content="You selected Auto Insurance, INS003. Please clarify your needs.")
        notebook = json.loads((ROOT / "notebooks/04_lcel_workflows.ipynb").read_text())
        scope = {}
        with patch("lab.config.get_chat_model", return_value=RunnableLambda(fake)), contextlib.redirect_stdout(io.StringIO()):
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    exec("".join(cell["source"]), scope)
        self.assertEqual(scope["result"]["product"]["id"], "INS003")
        self.assertEqual(scope["result"]["language"], "English")
        self.assertIsInstance(scope["result"]["intent"], dict)


if __name__ == "__main__":
    unittest.main()
