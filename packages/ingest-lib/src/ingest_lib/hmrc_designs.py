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
import markdownify as md
import yaml

# Local imports
from milvus_lib import ComponentEntry

from ingest_lib.file_dates import GitFileDates
from ingest_lib.protocols import ExtractComponents

logger = logging.getLogger(__name__)


class HMRCComponentsIngestor(ExtractComponents):
    project_root: Path
    components_dir: Path

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.components_dir = self.project_root / "src" / "hmrc-design-patterns"

    def __walk_components(self) -> Iterator[HMRCComponentEntry]:
        """Walk through component directories and yield HMRCComponentEntry objects."""
        frontmatter_re = re.compile(r"^---\s*\n(.*?)---", re.MULTILINE | re.DOTALL)
        first_line_re = re.compile(
            r"\{%\s*block content\s*%\}([\s\S]*?)(?=^#|\Z)", re.MULTILINE
        )

        gitfiledates = GitFileDates(self.components_dir)
        datesdict = gitfiledates.get_file_dates("index.njk")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("****************************")
            logger.debug(datesdict)
            logger.debug("****************************")

        skip_dirs = [
            "__tests__",
            "hmrc-design-patterns-archive",
            "hmrc-design-patterns-backlog",
        ]

        for component_path in self.components_dir.iterdir():
            if not component_path.is_dir():
                continue
            if component_path.stem in skip_dirs:
                continue

            index_file = component_path / "index.njk"
            if not index_file.exists():
                logger.warning(f"index.njk not found for component: {component_path}")
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
            status = index_frontmatter.get("status", "N/A")

            markdown_content = md.markdownify(index_content, heading_style=md.ATX)

            first_line = "N/A"
            first_line_match = first_line_re.search(markdown_content)
            if not first_line_match:
                logger.error(f"Description not found for component: {component_path}")
            else:
                first_line = first_line_match.group(1).strip()

            logger.debug("File: ", index_file)
            foldername = str(component_path).rsplit("/", 1)[-1]
            key = f"{foldername}/{'index.njk'}"
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
                "status": status,
            }

            content_buffer = StringIO()
            content_buffer.write("# Source: index.njk\n\n")
            content_buffer.write(
                f"*Path: design-system/src/hmrc-design-patterns/{component_path.name}/index.njk*\n\n"
            )
            content_buffer.write(markdown_content)

            full_content = content_buffer.getvalue()

            yield HMRCComponentEntry(
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
            return sum(1 for _ in self.components_dir.glob("*/index.njk"))
        return 0

    def extract_components(self) -> Iterator[ComponentEntry]:
        for component in self.__walk_components():
            yield component.to_component_entry()


@dataclass
class HMRCComponentEntry:
    """Dataclass for HMRC Design System component entries."""

    when_to_use_re: ClassVar[Pattern] = re.compile(
        r"## When to use\s*\n+(.+?)(?=\n##|\n#|$)", re.DOTALL
    )

    component_path: Path
    title: str
    first_line: str
    frontmatter: Dict[str, str]
    full_content: str

    def extract_has_research(self) -> bool:
        """Check if the component mentions research."""
        return HMRCComponentEntry.research_re.search(self.full_content) is not None

    def extract_description(self) -> str:
        """Extract description from the first line or 'When to use' section."""
        # Use the first line as the primary description
        if self.first_line:
            return self.first_line

        # Fallback to "When to use this component" section
        when_to_use_match = HMRCComponentEntry.when_to_use_re.search(self.full_content)
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

        return "HMRC Design System component documentation"

    def extract_accessibility(self) -> str:
        """Extract accessibility level (assume AA if not specified)."""
        # Look for WCAG mentions
        if "WCAG" in self.full_content:
            if "2.1" in self.full_content or "2.2" in self.full_content:
                return "AA"

        # Default to AA for government services
        return "AA"

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
        """Convert HMRCComponentEntry to ComponentEntry for Milvus storage."""
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
        parent = "HMRC Design System"

        # Generate URL based on component folder name
        url = f"https://design.tax.service.gov.uk/hmrc-design-patterns/{self.component_path.name}/"

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
