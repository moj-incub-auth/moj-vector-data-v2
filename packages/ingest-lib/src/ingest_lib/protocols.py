import logging
import re
from abc import abstractmethod
from pathlib import Path
from re import Pattern
from typing import ClassVar, Iterator, Protocol

from milvus_lib import ComponentEntry

logger = logging.getLogger(f"uvicorn.{__name__}")


class ProjectExists(Protocol):
    """Protocol for types that represent a project with a root directory."""

    @abstractmethod
    def project_root(self) -> Path:
        """Return the path to the project root directory."""
        raise NotImplementedError

    @abstractmethod
    def project_exists(self) -> bool:
        """Return True if the project root exists on disk."""
        raise NotImplementedError


class ExtractComponents(ProjectExists, Protocol):
    """Protocol for extractors that yield design system components from a project."""

    research_available_header_re: ClassVar[Pattern] = re.compile(
        r"#+\s*(?:Research|Research findings)", re.IGNORECASE
    )
    research_available_terms_re: ClassVar[Pattern] = re.compile(
        r"research showed|users understood|we found|testing showed|we observed|usability tested|research has shown|found|has shown",
        re.IGNORECASE,
    )

    research_needed_header_re: ClassVar[Pattern] = re.compile(
        r"#+\s*(?:Needs more research)", re.IGNORECASE
    )
    research_needed_terms_re: ClassVar[Pattern] = re.compile(
        r"we need more research|research needed|we need more evidence|needs further testing|get in touch to share research|we want to do more usability testing|if you\’ve done any user research",
        re.IGNORECASE,
    )

    accessibility_issues_header_re: ClassVar[Pattern] = re.compile(
        r"#+\s*(?:Accessibility issues)", re.IGNORECASE
    )
    accessibility_issues_terms_re: ClassVar[Pattern] = re.compile(
        r"does not meet WCAG|known accessibility issues|users will find it difficult|assistive technology users|this fails",
        re.IGNORECASE,
    )

    @abstractmethod
    def component_count(self) -> int:
        """Return the number of components available in the project."""
        raise NotImplementedError

    @abstractmethod
    def extract_components(self) -> Iterator[ComponentEntry]:
        """Yield ComponentEntry instances from the project."""
        raise NotImplementedError

    @staticmethod
    def _has_research(content: str) -> bool:
        header_match = ExtractComponents.research_available_header_re.search(content)
        body_match = ExtractComponents.research_available_terms_re.finditer(content)
        body_count = sum(1 for _ in body_match)
        logger.debug(f"Has research available: {header_match} and {body_count}")
        if header_match and body_count > 0:
            return True
        if body_count > 1:
            return True
        return False

    @staticmethod
    def _needs_research(content: str) -> bool:
        header_match = ExtractComponents.research_needed_header_re.search(content)
        body_match = ExtractComponents.research_needed_terms_re.finditer(content)
        body_count = sum(1 for _ in body_match)
        logger.debug(f"Has research needed: {header_match} and {body_count}")

        return bool(header_match) or body_count > 0

    @staticmethod
    def _has_accessibility_issues(content: str) -> bool:
        header_match = ExtractComponents.accessibility_issues_header_re.search(content)
        body_match = ExtractComponents.accessibility_issues_terms_re.finditer(content)
        body_count = sum(1 for _ in body_match)
        logger.debug(f"Has accessibility issues: {header_match} and {body_count}")
        return bool(header_match) or body_count > 0
