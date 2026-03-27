# Standard library imports
import logging
import re
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from re import Pattern
from typing import ClassVar, Dict, Iterator

# Third party imports
import yaml

# Local imports
from milvus_lib import ComponentEntry

from .protocols import ExtractComponents

logger = logging.getLogger(__name__)


class MojFrontendIngestor(ExtractComponents):
    """Extracts design system components from a moj-frontend project.

    Walks the docs/components directory, parses YAML frontmatter and markdown
    content from each component's index.md and related .md files, and yields
    ComponentEntry instances suitable for the Milvus knowledge base.
    """

    project_root: Path
    components_dir: Path

    def __init__(self, project_root: Path):
        """Initialize the ingestor with the moj-frontend project root.

        Args:
            project_root: Path to the moj-frontend repository root.
        """
        self.project_root = project_root
        self.components_dir = self.project_root / "docs" / "components"

    def __walk_components(self) -> Iterator[MojFrontendComponentEntry]:
        """Walk component directories and yield parsed MojFrontendComponentEntry objects."""

        metadata_re = re.compile(r"^---\s*\n(.*?)---", re.MULTILINE | re.DOTALL)
        for component_path in self.components_dir.iterdir():
            if not component_path.is_dir():
                continue
            index_md = component_path / "index.md"
            if not index_md.exists():
                logger.error(f"Index file not found for component: {component_path}")
                raise FileNotFoundError(
                    f"Index file not found for component: {component_path}"
                )
            index_content = index_md.read_text()
            block_match = metadata_re.search(index_content)
            if not block_match:
                logger.error(
                    f"Metadata block not found for component: {component_path}"
                )
                raise ValueError(
                    f"Metadata block not found for component: {component_path}"
                )
            header_content = block_match.group(1)
            frontmatter = yaml.safe_load(header_content)

            content_buffer = StringIO()
            content_buffer.write("# Source: index.md\n\n")
            content_buffer.write(
                f"*Path: moj-frontend/docs/components/{component_path.stem}/index.md*\n\n"
            )
            content_buffer.write(index_content)
            for md_file in component_path.glob("*.md"):
                if md_file == index_md:
                    continue
                content_buffer.write("\n\n---\n\n")
                content_buffer.write(f"# Source: {md_file.name}\n\n")
                content_buffer.write(
                    f"*Path: moj-frontend/docs/components/{component_path.stem}/{md_file.name}*\n\n"
                )
                content_buffer.write(md_file.read_text())
            full_content = content_buffer.getvalue()
            yield MojFrontendComponentEntry(
                component_path=component_path,
                frontmatter=frontmatter,
                full_content=full_content,
            )

    def project_exists(self) -> bool:
        """Return True if the project root directory exists."""
        return self.project_root.exists()

    def project_root(self) -> Path:
        """Return the path to the project root directory."""
        return self.project_root

    def component_count(self) -> int:
        """Return the number of components (subdirs with index.md) in the project."""
        if self.components_dir.exists() and self.components_dir.is_dir():
            return sum(1 for _ in self.components_dir.glob("*/index.md"))
        return 0

    def extract_components(self) -> Iterator[ComponentEntry]:
        """Yield ComponentEntry instances from all components in the project."""
        for component in self.__walk_components():
            yield component.to_component_entry()


@dataclass
class MojFrontendComponentEntry:
    """A parsed component from moj-frontend with frontmatter and full content.

    Holds raw data from the filesystem and provides extraction methods to
    derive structured fields (description, dates, accessibility) for ComponentEntry.
    """

    overview_re: ClassVar[Pattern] = re.compile(
        r"## Overview\s*\n+(.+?)(?=\n##|\n#|$)", re.DOTALL
    )
    component_path: Path
    frontmatter: Dict[str, str]
    full_content: str

    def extract_description(self) -> str:
        """Extract description from frontmatter lede or Overview section.

        Falls back to the first paragraph of the Overview section, or a default
        string if neither is available.
        """
        if "lede" in self.frontmatter:
            return self.frontmatter["lede"]
        overview_match = MojFrontendComponentEntry.overview_re.search(self.full_content)
        if overview_match:
            # Get first paragraph
            paragraphs = overview_match.group(1).strip().split("\n\n")
            for para in paragraphs:
                # Skip example blocks and empty lines
                if (
                    para.strip()
                    and not para.startswith("{%")
                    and not para.startswith("<")
                ):
                    return para.strip()
        return "Component documentation"

    def extract_dates(self) -> tuple[str, str]:
        """Extract or generate created_at and updated_at dates."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check for status date in frontmatter
        if "statusDate" in self.frontmatter:
            try:
                # Parse date like "February 2025"
                date_str = self.frontmatter["statusDate"]
                # Convert to ISO format (assume first day of month)
                date_obj = datetime.strptime(date_str, "%B %Y")
                formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                return formatted_date, formatted_date
            except Exception as e:
                logger.error(
                    f"Failed to parse status date: {self.frontmatter['statusDate']}: {e}"
                )
                return now, now

        return now, now

    def to_component_entry(self) -> ComponentEntry:
        """Convert this parsed component into a ComponentEntry for Milvus ingestion."""
        title = self.frontmatter["title"]
        description = self.extract_description()
        status = self.frontmatter["status"]
        created_at, updated_at = self.extract_dates()
        has_research = ExtractComponents._has_research(self.full_content)
        needs_research = ExtractComponents._needs_research(self.full_content)
        accessibility = "N/A"
        if ExtractComponents._has_accessibility_issues(self.full_content):
            accessibility = "Accessibility issues"

        logger.info(
            f"Parsing component: {title} - has_research: {has_research} - needs_research: {needs_research} - accessibility: {accessibility}"
        )

        parent = "MOJ Design System"
        url = f"https://design-patterns.service.justice.gov.uk/components/{self.component_path.stem}/"

        content = f"""
Title: {title}
Description: {description}
Parent: {parent}
Content: {self.full_content}
        """[:65000].strip()
        return ComponentEntry(
            component_id=url,
            title=title,
            description=description,
            url=url,
            parent=parent,
            status=status,
            accessibility=accessibility,
            has_research=has_research,
            needs_research=needs_research,
            created_at=created_at,
            updated_at=updated_at,
            views=0,
            content=content,
            full_content=self.full_content,
        )
