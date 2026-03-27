from abc import abstractmethod
from dataclasses import dataclass
import json
import logging
import hashlib
import re
from re import Pattern
from pathlib import Path
from typing import ClassVar, Iterator, Protocol

from milvus_lib import ComponentEntry
from llm_lib import LLMIngestionAssistantBase
from llm_lib import LLMComponentDataResponse

logger = logging.getLogger(__name__)

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


class ComponentsExtractor(Protocol):
    """Protocol for extractors that yield design system components from."""
    project_root: Path
    components_dir: Path
    llm_assistant: LLMIngestionAssistantBase

    # research_available_header_re: ClassVar[Pattern] = re.compile(
    #     r"#+\s*(?:Research|Research findings)", re.IGNORECASE
    # )
    # research_available_terms_re: ClassVar[Pattern] = re.compile(
    #     r"research showed|users understood|we found|testing showed|we observed|usability tested|research has shown|found|has shown",
    #     re.IGNORECASE,
    # )

    # research_needed_header_re: ClassVar[Pattern] = re.compile(
    #     r"#+\s*(?:Needs more research)", re.IGNORECASE
    # )
    # research_needed_terms_re: ClassVar[Pattern] = re.compile(
    #     r"we need more research|research needed|we need more evidence|needs further testing|get in touch to share research|we want to do more usability testing|if you\’ve done any user research",
    #     re.IGNORECASE,
    # )

    # accessibility_issues_header_re: ClassVar[Pattern] = re.compile(
    #     r"#+\s*(?:Accessibility issues)", re.IGNORECASE
    # )
    # accessibility_issues_terms_re: ClassVar[Pattern] = re.compile(
    #     r"does not meet WCAG|known accessibility issues|users will find it difficult|assistive technology users|this fails",
    #     re.IGNORECASE,

    @abstractmethod
    def _build_extraction_prompt(self, file_content: str, component_url: str) -> ComponentEntry:
        """Yield ComponentEntry instances from the project."""
        raise NotImplementedError
    
    @abstractmethod
    def extract_components(self) -> Iterator[ComponentEntry]:
        """Yield ComponentEntry instances from the project."""
        raise NotImplementedError    

    # @staticmethod
    # def _has_research(content: str) -> bool:
    #     header_match = ExtractComponents.research_available_header_re.search(content)
    #     body_match = ExtractComponents.research_available_terms_re.finditer(content)
    #     body_count = sum(1 for _ in body_match)
    #     logger.debug(f"Has research available: {header_match} and {body_count}")
    #     if header_match and body_count > 0:
    #         return True
    #     if body_count > 1:
    #         return True
    #     return False

    # @staticmethod
    # def _needs_research(content: str) -> bool:
    #     header_match = ComponentsExtractor.research_needed_header_re.search(content)
    #     body_match = ComponentsExtractor.research_needed_terms_re.finditer(content)
    #     body_count = sum(1 for _ in body_match)
    #     logger.debug(f"Has research needed: {header_match} and {body_count}")

    #     return bool(header_match) or body_count > 0

    # @staticmethod
    # def _has_accessibility_issues(content: str) -> bool:
    #     header_match = ExtractComponents.accessibility_issues_header_re.search(content)
    #     body_match = ExtractComponents.accessibility_issues_terms_re.finditer(content)
    #     body_count = sum(1 for _ in body_match)
    #     logger.debug(f"Has accessibility issues: {header_match} and {body_count}")
    #     return bool(header_match) or body_count > 0


    def extract_from_content(
        self,
        content: str,
        component_url: str,
    ) -> LLMComponentDataResponse:
        """
        Extract component data from content string using LLM.

        Args:
            content: The component file content as string
            component_url: URL where the component is hosted
            parent: Name of the parent design system

        Returns:
            LLMComponentDataResponse: Extracted component data

        Raises:
            ValueError: If LLM response cannot be parsed
        """
        # # Check for research using rule-based method
        # has_research = ComponentsExtractor._check_has_research(self, content=content)

        # Build prompt and query LLM
        prompt = self._build_extraction_prompt(content, component_url)

        try:
            response = self.llm_assistant.client.chat.completions.create(
                model=self.llm_assistant.inference_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that extracts structured data from documentation. Always respond with valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,  # Low temperature for consistent extraction
                max_tokens=3000,
            )

            logger.debug("---------------------- RAW LLM OUTPUT ------------------------")
            logger.debug(response)
            logger.debug("---------------------- RAW LLM OUTPUT END ------------------------")

            # Extract JSON from response
            llm_output = response.choices[0].message.content.strip()

            # Try to find JSON in the response (in case LLM adds extra text)
            json_match = re.search(r"\{.*\}", llm_output, re.DOTALL)
            if json_match:
                llm_output = json_match.group(0)

            # Parse JSON response
            data = json.loads(llm_output)

            print("---------------------- JSON LLM OUTPUT ------------------------")
            print(llm_output)
            logger.debug("---------------------- JSON LLM OUTPUT ------------------------")
            logger.debug(llm_output)            

            # # Override has_research with our rule-based check
            # if data.get("title") :
            #     data["has_research"] = has_research

            # Validate and return
            return LLMComponentDataResponse(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            logger.error(f"-----------------------------")
            logger.error(f"JSON SCHEMA :{json_match}")
            logger.error(f"-----------------------------")
            logger.error(f"LLM response was: {llm_output}")
            raise ValueError(f"LLM response was not valid JSON: {e}")

        except Exception as e:
            logger.error(f"Error querying LLM: {e}")
            raise


