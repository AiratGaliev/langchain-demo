import requests
from bs4 import BeautifulSoup
from langchain_core.tools import BaseTool


class WebPageTool(BaseTool):
    name = "Get Webpage"
    description = "Useful for when you need to get the content from a specific webpage"

    def _run(self, webpage: str):
        response = requests.get(webpage)
        response_text = response.text

        def strip_html_tags(html_content):
            soup = BeautifulSoup(html_content, "html.parser")
            return soup.get_text()

        stripped_content = strip_html_tags(response_text)
        if len(stripped_content) > 4000:
            stripped_content = stripped_content[:4000]
        return stripped_content

    def _arun(self, webpage: str):
        raise NotImplementedError("This tool does not support async")
