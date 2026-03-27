# Standard library imports
import logging
from dataclasses import dataclass
from datetime import datetime
from io import StringIO
from pathlib import Path
from re import Pattern
from typing import ClassVar, Dict, Iterator

# Local imports
from milvus_lib import ComponentEntry

from ingest_lib.file_dates import GitFileDates

from llm_lib.protocols import ComponentsExtractor
from llm_lib import LLMIngestionAssistantBase, LLMComponentEntry


logger = logging.getLogger(__name__)


class DWPComponentsIngestorAI(ComponentsExtractor):
    # Common Moved to ComponentsExtractor
    data_file_name : str
    data_file_location : str

    def __init__(self, llm_assistant: LLMIngestionAssistantBase, project_root: Path):
        self.project_root = project_root
        self.components_dir = self.project_root / "app" / "views" / "components"
        self.llm_assistant = llm_assistant
        self.data_file_name="README.md.njk"
        self.data_file_location="design-system/app/views/components"

    def __walk_components(self) -> Iterator[ComponentEntry]:
        """Walk through component directories and yield LLMComponentEntry objects."""

        print(f"walking components directory: {self.components_dir}")
        logger.debug(f"walking components directory: {self.components_dir}")

        gitfiledates = GitFileDates(self.components_dir)
        datesdict = gitfiledates.get_file_dates(self.data_file_name)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("****************************")
            logger.debug(datesdict)
            logger.debug("****************************")

        for component_path in self.components_dir.iterdir():
            if not component_path.is_dir():
                continue

            readme_file = component_path / self.data_file_name
            if not readme_file.exists():
                logger.warning(
                    f"{self.data_file_name} not found for component: {component_path}"
                )
                continue

            readme_content = readme_file.read_text()

            logger.debug("File: ", readme_file)
            foldername = str(component_path).rsplit("/", 1)[-1]
            key = f"{foldername}/{self.data_file_name}"
            logger.debug("Path: ", foldername)
            logger.debug("Last update at: ", datesdict[key])            

            content_buffer = StringIO()
            content_buffer.write(f"# Source: {self.data_file_name}\n\n")
            content_buffer.write(
                f"*Path: {self.data_file_name}/{component_path.name}/{self.data_file_name}*\n\n"
            )
            content_buffer.write(readme_content)

            # Check for other .njk files in the directory
            for njk_file in component_path.glob("*.njk"):
                if njk_file == readme_file:
                    continue
                content_buffer.write("\n\n---\n\n")
                content_buffer.write(f"# Source: {njk_file.name}\n\n")
                content_buffer.write(
                    f"*Path: {readme_file}*\n\n"
                )
                content_buffer.write(njk_file.read_text())

            # 1. Parse the string into a datetime object
            # %a: Weekday, %b: Month, %d: Day, %H:%M:%S: Time, %Y: Year, %z: UTC offset
            dt_obj = datetime.strptime(datesdict[key], "%a %b %d %H:%M:%S %Y %z")

            # 2. Format the object into your desired string
            formatted_date = dt_obj.strftime("%Y-%m-%d %H:%M:%S")

            content_buffer.write(f"# Last Modification: {formatted_date}\n\n")

            full_content = content_buffer.getvalue()

            print(f"NEXT extract_from_content ==> {component_path.name}")
            logger.debug(f"NEXT extract_from_content ==> {component_path.name}")

            ingestion_result = self.extract_from_content(
              content=full_content,
              component_url=component_path.name,
            )
            print("------------------ RESPONSE --------------------")
            print(ingestion_result)

            yield LLMComponentEntry(
                component_path=component_path,
                llm_structured_output=ingestion_result,
                full_content=full_content,
            )



    def project_exists(self) -> bool:
        return self.project_root.exists()

    def project_root(self) -> Path:
        return self.project_root

    def component_count(self) -> int:
        if self.components_dir.exists() and self.components_dir.is_dir():
            return sum(1 for _ in self.components_dir.glob(f"*/{self.data_file_name}"))
        return 0

    def extract_components(self) -> Iterator[ComponentEntry]:
        for component in self.__walk_components():
            yield component.to_component_entry()


    def _build_extraction_prompt(
        self, file_content: str, component_name: str
    ) -> str:
        """
        Build the prompt for extracting component data from file content.

        Args:
            file_content: The content of the component file
            component_url: The URL where the component documentation is hosted
            parent: The parent design system name

        Returns:
            str: The formatted prompt
        """
        prompt = f"""You are an expert at extracting structured information from design system documentation.

<TASK>        
Analyze the following component documentation file and extract the key information into a structured JSON format.

Component URL: https://design-system.dwp.gov.uk/components/{component_name}
Parent Design System: DWP Design System
</TASK>

<CONSTRAINTS>
1. Output MUST be valid JSON only.
2. DO NOT include markdown code blocks (no ```json).
3. DO NOT include any introductory text, explanations, or post-amble.
4. If a field is unknown, use null.
5. Start your response with the character '{{' and end it with '}}'.
</CONSTRAINTS>

<DATA_MAPPING>
- title: From {component_name} replace dashes with spaces
- url: https://design-system.dwp.gov.uk/components/{component_name}
- description: Skip lines starting with Source: and Path: and use the first line of the content for the description
- parent: Extract from this prompt's Parent Design System value.
- accessibility: Value must be either 'Accessibility issues' or an empty string. Nothing else. Value is 'Accessibility issues' when there is a heading 'Accessibility issues', or the file content includes 'does not meet WCAG', 'known accessibility issues', 'users will find it difficult', 'assistive technology users', 'this fails' otherwise set to an empty string.
- created_at / updated_at: Convert value '# Last Modification:' to "YYYY-MM-DD HH:MM:SS". If time is missing, use "00:00:00".
- views: Set to 0.
- has_research: Value is True or False. True when there is a heading starting 'Research' or the file content has expressions 'research showed', 'users understood','we found','testing showed','we observed','usability tested','research has shown','found','has shown'. Else Fasle.
- needs_research: Value is True or False. True when there is a heading 'Needs more research' or the file content has expressions 'we need more research', 'research needed','we need more evidence','needs further testing','get in touch to share research','if you’ve done any user research'. Else Fasle.
- Return a valid JSON object with the following exact structure
</DATA_MAPPING>


<OUTPUT_SCHEMA>
{{
    "title": "Component Title",
    "url": "https://design-system.dwp.gov.uk/components/{component_name}",
    "description": "Component description",
    "parent": "DWP Design System",
    "accessibility": "",
    "created_at": "2026-03-04 10:55:00",
    "updated_at": "2026-03-04 10:55:00",
    "has_research": false,
    "needs_research": false,
    "views": 0,
    "status": "N/A"
}}
</OUTPUT_SCHEMA>


<COMPONENT_FILE_CONTENT>
{file_content}
</COMPONENT_FILE_CONTENT>

Final Instruction: Return ONLY the JSON object wrapped in."""


        print("------------------ PROMPT --------------------")    
        #print (prompt)
        print(f"Prompt for ingestion of: [{component_name}]")
        
        return prompt


