"""
Example usage of the ComponentExtractor class.

This demonstrates how to extract component data from NHS Design System files.
"""
import os
from pathlib import Path

from llm_lib import ComponentExtractor


def main():
    # Initialize the extractor
    # You can provide custom base_url and api_key, or use environment variables
    extractor = ComponentExtractor(
        model="qwen3-14b-llm",
        # base_url="your-llm-endpoint-url",  # Optional, defaults to env var
        # api_key="your-api-key",  # Optional, defaults to env var
    )

    # Example 1: Extract from a file
    print("========================================================")
    print("EXAMPLE 1")
    print("========================================================")    
    file_path = Path(
        os.getenv("TEST_FILE_URL")  
        or 
        "/home/stkousso/Stelios/Projects/2026/0018-MoJ/customer-resources/data/nhsuk-service-manual/app/views/design-system/components/footer/index.njk"
    )

    if file_path.exists():
        result = extractor.extract_from_file(
            file_path=file_path,
            component_url="https://service-manual.nhs.uk/design-system/components/footer",
            parent="NHS Design System",
        )

        print("Extraction Result:")
        print(f"Message: {result.message}")
        print(f"Number of components: {len(result.components)}")

        for component in result.components:
            print(f"\nComponent: {component.title}")
            print(f"  URL: {component.url}")
            print(f"  Description: {component.description}")
            print(f"  Parent: {component.parent}")
            print(f"  Accessibility: {component.accessibility}")
            print(f"  Has Research: {component.has_research}")
            print(f"  Created: {component.created_at}")
            print(f"  Updated: {component.updated_at}")

        # Get as JSON
        print("\nJSON Output:")
        print(result.model_dump_json(indent=2))

    # # Example 2: Extract from content string
    # content = """
    # {% set pageTitle = "Action link" %}
    # {% set pageDescription = "Use action links to help users get to the next stage of a journey quickly." %}
    # {% set dateUpdated = "November 2025" %}

    # <h2 id="research">Research</h2>
    # <p>We tested the action links and users understood the purpose clearly.</p>
    # <p>Research showed that the green arrow was more visible than the blue one.</p>
    # """

    # result2 = extractor.extract_from_content(
    #     content=content,
    #     component_url="https://example.com/component",
    #     parent="Example Design System",
    # )

    # print("\n\nExtraction from content string:")
    # print(f"Has Research: {result2.components[0].has_research}")
    # print("(Should be True because it has Research heading and 'users understood' + 'research showed')")


if __name__ == "__main__":
    main()
