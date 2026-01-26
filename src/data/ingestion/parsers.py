"""Document parsers using Unstructured.io."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


class ParsedElement(BaseModel):
    """Represents a parsed document element."""

    element_type: str  # paragraph, title, table, code, image, etc.
    text: str
    metadata: dict


class ParsedDocument(BaseModel):
    """Represents a fully parsed document."""

    elements: list[ParsedElement]
    metadata: dict
    source: str


class BaseParser(ABC):
    """Abstract base class for document parsers."""

    @abstractmethod
    def parse(self, content: str, doc_type: str, metadata: dict) -> ParsedDocument:
        """Parse document content."""
        pass

    @abstractmethod
    def supported_types(self) -> list[str]:
        """Return list of supported document types."""
        pass


class UnstructuredParser(BaseParser):
    """Parser using Unstructured.io library."""

    def __init__(
        self,
        strategy: str = "auto",
        include_page_breaks: bool = False,
        extract_images: bool = False,
    ):
        self.strategy = strategy
        self.include_page_breaks = include_page_breaks
        self.extract_images = extract_images

    def parse(self, content: str, doc_type: str, metadata: dict) -> ParsedDocument:
        """Parse document using Unstructured.

        Args:
            content: Raw document content
            doc_type: Document type (extension)
            metadata: Additional metadata

        Returns:
            ParsedDocument with extracted elements
        """
        # Import here to avoid dependency issues
        from unstructured.partition.auto import partition

        # Create temporary file for binary formats
        elements = partition(
            text=content,
            strategy=self.strategy,
            include_page_breaks=self.include_page_breaks,
        )

        parsed_elements = []
        for elem in elements:
            parsed_elements.append(
                ParsedElement(
                    element_type=elem.category,
                    text=str(elem),
                    metadata={
                        "coordinates": getattr(elem.metadata, "coordinates", None),
                        "page_number": getattr(elem.metadata, "page_number", None),
                    },
                )
            )

        return ParsedDocument(
            elements=parsed_elements,
            metadata=metadata,
            source=metadata.get("source", "unknown"),
        )

    def supported_types(self) -> list[str]:
        """Return supported document types."""
        return [".txt", ".md", ".pdf", ".docx", ".html", ".rst", ".xml"]


class MarkdownParser(BaseParser):
    """Simple markdown parser for basic text extraction."""

    def parse(self, content: str, doc_type: str, metadata: dict) -> ParsedDocument:
        """Parse markdown document.

        Args:
            content: Markdown content
            doc_type: Document type
            metadata: Additional metadata

        Returns:
            ParsedDocument with text elements
        """
        import re

        elements = []

        # Split by headers
        sections = re.split(r"(^#{1,6}\s+.+$)", content, flags=re.MULTILINE)

        for section in sections:
            section = section.strip()
            if not section:
                continue

            if section.startswith("#"):
                element_type = "title"
            elif section.startswith("```"):
                element_type = "code"
            else:
                element_type = "paragraph"

            elements.append(
                ParsedElement(
                    element_type=element_type,
                    text=section,
                    metadata={},
                )
            )

        return ParsedDocument(
            elements=elements,
            metadata=metadata,
            source=metadata.get("source", "unknown"),
        )

    def supported_types(self) -> list[str]:
        """Return supported document types."""
        return [".md", ".markdown"]


def get_parser(doc_type: str) -> BaseParser:
    """Get appropriate parser for document type.

    Args:
        doc_type: Document type (extension)

    Returns:
        Parser instance for the document type
    """
    parsers: list[BaseParser] = [MarkdownParser(), UnstructuredParser()]

    for parser in parsers:
        if doc_type in parser.supported_types():
            return parser

    # Default to Unstructured for unknown types
    return UnstructuredParser()
