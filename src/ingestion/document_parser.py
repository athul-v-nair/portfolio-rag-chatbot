import json
from typing import List
from langchain_core.documents import Document
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI

from src.utils.logger import logger
from src.utils.constants import GEMINI_API_KEY, GEMINI_TEXT_GENERATION_MODEL
from src.utils.prompts.parsing_prompt import RESUME_PARSER_PROMPT

class DocumentParser:
    def __init__(self, documents: List[Document]):
        self.documents = documents

        self.model = ChatGoogleGenerativeAI(
            model=GEMINI_TEXT_GENERATION_MODEL,
            api_key=GEMINI_API_KEY,
        )
        # Combining pages as the loader splits the document
        self.combined_docs = self.combine_pdf_pages()

        # Using an LLM to identify pdf sections
        self.parsed_documents = self.identify_sections()

    def combine_pdf_pages(self):
        combined_docs = {}

        for doc in self.documents:

            file_name = doc.metadata.get("file_name")
            source = doc.metadata.get("source")
            page_number = doc.metadata.get("page")

            combined_docs.setdefault(file_name, "")
            combined_docs[file_name] += "\n" + doc.page_content

        return combined_docs
    
    def identify_sections(self):

        combined_files = self.combined_docs

        parsed_documents = []

        for file_name, text in combined_files.items():
            logger.info("Calling LLM to section the document")
            response = self.model.invoke(RESUME_PARSER_PROMPT.format(resume_text=text))
            logger.info(f"Model response: {response}")

            content = response.content.strip()

            content = content.replace("```json", "").replace("```", "")

            parsed = json.loads(content)

            parsed_documents.extend(
                self.convert_to_documents(parsed, file_name)
            )

        return parsed_documents
    
    def convert_to_documents(self, parsed, file_name):
        documents = []

        # Summary
        if parsed.get("summary"):
            documents.append(
                Document(
                    page_content=parsed["summary"],
                    metadata={
                        "section": "summary",
                        "file_name": file_name
                    }
                )
            )

        # Skills
        if parsed.get("skills"):
            documents.append(
                Document(
                    page_content=", ".join(parsed["skills"]),
                    metadata={
                        "section": "skills",
                        "file_name": file_name
                    }
                )
            )

        # Projects
        for project in parsed.get("projects", []):
            projects_text = ""
            for project in parsed["projects"]:
                projects_text += f"Project: {project['title']}\n"
                projects_text += f"Description:\n{project['description']}\n"
                projects_text += f"Technologies: {', '.join(project['technologies'])}\n\n"

            documents.append(
                Document(
                    page_content=projects_text.strip(),
                    metadata={"section": "projects", "file_name": file_name}
                )
            )

        # Experience
        for exp in parsed.get("experience", []):
            exp_text = ""
            for exp in parsed["experience"]:
                exp_text += f"Company: {exp['company']}\n"
                exp_text += f"Role: {exp['role']}\n"
                exp_text += f"Description:\n{exp['description']}\n\n"

            documents.append(
                Document(
                    page_content=exp_text.strip(),
                    metadata={"section": "experience", "file_name": file_name}
                )
            )

        # education
        for edu in parsed.get("education", []):
            for edu in parsed["education"]:
                edu_text += f"College: {edu['college_name']}\n"
                edu_text += f"Course: {edu['course_name']}\n"
                edu_text += f"Related Courses: {edu.get('related_courses', '')}\n\n"

            documents.append(
                Document(
                    page_content=edu_text.strip(),
                    metadata={
                        "section": "education",
                        "college": edu.get("college_name"),
                        "file_name": file_name
                    }
                )
            )

        return documents