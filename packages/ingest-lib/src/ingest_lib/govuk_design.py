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

from ingest_lib.file_dates import GitFileDates

from .protocols import ExtractComponents

logger = logging.getLogger(__name__)


class GovUkComponentsIngestor(ExtractComponents):
    project_root: Path
    components_dir: Path

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.components_dir = self.project_root / "src" / "components"

    def __walk_components(self) -> Iterator[GovUkComponentEntry]:
        """Walk through component directories and yield GovUkComponentEntry objects."""
        frontmatter_re = re.compile(r"^---\s*\n(.*?)---", re.MULTILINE | re.DOTALL)

        gitfiledates = GitFileDates(self.components_dir)
        datesdict = gitfiledates.get_file_dates("index.md")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("****************************")
            logger.debug(datesdict)
            logger.debug("****************************")

        for component_path in self.components_dir.iterdir():
            if not component_path.is_dir():
                continue

            index_file = component_path / "index.md"
            if not index_file.exists():
                logger.warning(f"index.md not found for component: {component_path}")
                continue

            index_content = index_file.read_text()
            frontmatter_match = frontmatter_re.search(index_content)
            if not frontmatter_match:
                logger.error(f"Frontmatter not found for component: {component_path}")
                raise ValueError(
                    f"Frontmatter not found for component: {component_path}"
                )
            index_frontmatter = yaml.safe_load(frontmatter_match.group(1))

            title = index_frontmatter["title"]
            first_line = index_frontmatter.get("description")

            logger.debug("File: ", index_file)
            foldername = str(component_path).rsplit("/", 1)[-1]
            key = f"{foldername}/{'index.md'}"
            logger.debug("Path: ", foldername)
            logger.debug("Last update at: ", datesdict[key])

            # 1. Parse the string into a datetime object
            # %a: Weekday, %b: Month, %d: Day, %H:%M:%S: Time, %Y: Year, %z: UTC offset
            dt_obj = datetime.strptime(datesdict[key], "%a %b %d %H:%M:%S %Y %z")

            # 2. Format the object into your desired string
            formatted_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

            # Create frontmatter dictionary
            frontmatter = {
                "title": title,
                "statusDate": formatted_date,
            }

            content_buffer = StringIO()
            content_buffer.write("# Source: index.md\n\n")
            content_buffer.write(
                f"*Path: govuk-design-system/src/components/{component_path.name}/index.md*\n\n"
            )
            content_buffer.write(index_content)

            full_content = content_buffer.getvalue()

            yield GovUkComponentEntry(
                component_path=component_path,
                title=title,
                first_line=first_line,
                frontmatter=frontmatter,
                full_content=full_content,
            )

    def project_exists(self) -> bool:
        return self.project_root.exists()

    def project_root(self) -> Path:
        return self.project_root

    def component_count(self) -> int:
        if self.components_dir.exists() and self.components_dir.is_dir():
            return sum(1 for _ in self.components_dir.glob("*/index.md"))
        return 0

    def extract_components(self) -> Iterator[ComponentEntry]:
        for component in self.__walk_components():
            yield component.to_component_entry()


@dataclass
class GovUkComponentEntry:
    """Dataclass for DWP Design System component entries."""

    when_to_use_re: ClassVar[Pattern] = re.compile(
        r"## When to use this component\s*\n+(.+?)(?=\n##|\n#|$)", re.DOTALL
    )

    component_path: Path
    title: str
    first_line: str
    frontmatter: Dict[str, str]
    full_content: str

    def extract_description(self) -> str:
        """Extract description from the first line or 'When to use' section."""
        # Use the first line as the primary description
        if self.first_line:
            return self.first_line

        # Fallback to "When to use this component" section
        when_to_use_match = GovUkComponentEntry.when_to_use_re.search(self.full_content)
        if when_to_use_match:
            # Get first paragraph
            paragraphs = when_to_use_match.group(1).strip().split("\n\n")
            for para in paragraphs:
                # Skip example blocks and empty lines
                if (
                    para.strip()
                    and not para.startswith("{%")
                    and not para.startswith("<")
                ):
                    return para.strip()

        return "GovUk Design System component documentation"

    def extract_dates(self) -> tuple[str, str]:
        """Extract or generate created_at and updated_at dates."""
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Check for status date in frontmatter
        if "statusDate" in self.frontmatter:
            try:
                # Parse date like "February 2025"
                date_str = self.frontmatter["statusDate"]
                # Convert to ISO format (assume first day of month)
                # date_obj = datetime.strptime(date_str, "%B %Y")
                # formatted_date = date_obj.strftime("%Y-%m-%d %H:%M:%S")
                return date_str, date_str
            except Exception as e:
                logger.error(
                    f"Failed to parse status date: {self.frontmatter['statusDate']}: {e}"
                )
                return now, now

        return now, now

    def to_component_entry(self) -> ComponentEntry:
        """Convert GovUkComponentEntry to ComponentEntry for Milvus storage."""
        title = self.frontmatter["title"]
        description = self.extract_description()
        status = "N/A"
        created_at, updated_at = self.extract_dates()
        has_research = ExtractComponents._has_research(self.full_content)
        needs_research = ExtractComponents._needs_research(self.full_content)
        accessibility = "N/A"
        if ExtractComponents._has_accessibility_issues(self.full_content):
            accessibility = "Accessibility issues"
        logger.info(
            f"Parsing component: {title} - has_research: {has_research} - needs_research: {needs_research} - accessibility: {accessibility}"
        )
        parent = "GovUk Design System"

        # Generate URL based on component folder name
        url = f"https://design-system.service.gov.uk/components/{self.component_path.name}/"

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
