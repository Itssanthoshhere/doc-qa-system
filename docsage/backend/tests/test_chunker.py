"""
backend/tests/test_chunker.py — Unit tests for HierarchicalChunker.
"""
import pytest
from unittest.mock import MagicMock, patch
from utils.chunker import HierarchicalChunker, Chunk
from utils.document_parser import ParsedDocument, DocumentSection


def make_doc(sections_text: list[str], doc_id: str = "test_doc") -> ParsedDocument:
    sections = [
        DocumentSection(content=t, section_type="text")
        for t in sections_text
    ]
    return ParsedDocument(doc_id=doc_id, filename="test.pdf", sections=sections)


@pytest.fixture
def chunker():
    """Create chunker with mocked tokenizer for speed."""
    with patch("utils.chunker.AutoTokenizer") as mock_tok_cls:
        mock_tok = MagicMock()
        # Simulate 1 token per word (for predictable tests)
        mock_tok.encode.side_effect = lambda text, **kw: text.split()
        mock_tok_cls.from_pretrained.return_value = mock_tok
        c = HierarchicalChunker.__new__(HierarchicalChunker)
        c.tokenizer = mock_tok
        c.chunk_size = 50
        c.overlap = 10
        c.long_threshold = 100
        yield c


class TestHierarchicalChunker:
    def test_empty_document(self, chunker):
        doc = make_doc([])
        chunks = chunker.chunk_document(doc)
        assert chunks == []

    def test_single_short_section(self, chunker):
        doc = make_doc(["This is a simple test sentence for the chunker."])
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 1
        assert "simple test sentence" in chunks[0].content

    def test_chunk_has_required_fields(self, chunker):
        doc = make_doc(["Hello world. This is a test document."])
        chunks = chunker.chunk_document(doc)
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.chunk_id
            assert chunk.doc_id == "test_doc"
            assert chunk.content
            assert chunk.token_count > 0
            assert chunk.total_chunks > 0

    def test_long_document_creates_summary_chunks(self, chunker):
        # Make a doc with many words to exceed long_threshold
        long_text = " ".join(["word"] * 200)
        doc = make_doc([long_text])
        chunker.long_threshold = 10  # Force long-doc mode
        chunks = chunker.chunk_document(doc)
        summary_chunks = [c for c in chunks if c.is_summary]
        assert len(summary_chunks) >= 1

    def test_chunk_ids_are_unique(self, chunker):
        text = "Sentence one is here. Sentence two is here. Sentence three. Sentence four. Sentence five."
        doc = make_doc([text, text])  # Two identical sections
        chunks = chunker.chunk_document(doc)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Chunk IDs must be unique"

    def test_overlap_carries_content_forward(self, chunker):
        """With very small chunk_size, ensure overlap sentences appear in next chunk."""
        chunker.chunk_size = 5
        chunker.overlap = 3
        sentences = [
            "Alpha sentence here.",
            "Beta sentence here.",
            "Gamma sentence here.",
            "Delta sentence here.",
        ]
        doc = make_doc([" ".join(sentences)])
        chunks = chunker.chunk_document(doc)
        # Each chunk except first should share words with previous chunk
        if len(chunks) >= 2:
            first_words = set(chunks[0].content.lower().split())
            second_words = set(chunks[1].content.lower().split())
            assert first_words & second_words, "Overlap should share words between chunks"

    def test_table_sections_are_chunked(self, chunker):
        doc = make_doc([])
        doc.sections = [
            DocumentSection(
                content="| Header 1 | Header 2 |\n|---|---|\n| Row 1 | Value |",
                section_type="table",
            )
        ]
        chunks = chunker.chunk_document(doc)
        table_chunks = [c for c in chunks if c.section_type == "table"]
        assert len(table_chunks) >= 1

    def test_sif_summary_selects_informative_terms(self, chunker):
        """Summary should favor rare/domain terms over common words."""
        common = "the is and of to a in that"
        rare = "adversarial robustness ELECTRA transformer retrieval"
        text = f"{common} {rare} " * 10
        doc = make_doc([text])
        chunker.long_threshold = 5
        chunks = chunker.chunk_document(doc)
        summary_chunks = [c for c in chunks if c.is_summary]
        if summary_chunks:
            summary = summary_chunks[0].content.lower()
            # Rare domain words should appear in summary
            assert any(w in summary for w in ["electra", "transformer", "retrieval"])