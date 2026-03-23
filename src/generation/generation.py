from typing import List, Dict
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from src.retrieval.vector_search import search_vector_db
from src.generation.memory import ChatMemory
from src.utils.logger import logger 
from src.utils.constants import GEMINI_API_KEY, GEMINI_TEXT_GENERATION_MODEL, GENERATION_TEMPERATURE, NUMBER_OF_CHATS
from src.utils.constants import MEMORY_STORE
from src.utils.prompts.generation_prompt import GENERATION_PROMPT

class Generator:
    """
    Generator class responsible for handling the Retrieval-Augmented Generation (RAG) pipeline.

    This includes:
        - Retrieving relevant documents from the vector database
        - Formatting retrieved context
        - Building chat history
        - Building prompts for the LLM
        - Streaming generated responses along with source attribution
    """
    def __init__(self):
        self.model = ChatGoogleGenerativeAI(
            api_key=GEMINI_API_KEY,
            model=GEMINI_TEXT_GENERATION_MODEL,
            temperature=GENERATION_TEMPERATURE
        )

    def format_context(self, documents: List[Dict]) -> str:
        """
        Convert retrieved documents into a structured context string.

        Args:
            documents (List[Dict]):
                A list of retrieved document dictionaries. Each dictionary
                should contain:
                    - content (str)
                    - metadata (dict)

        Returns:
            str:
                A formatted string containing all documents, ready to be
                injected into the prompt.
        """
        context = ""
        for i, doc in enumerate(documents):
            context += f"\n[Document {i+1}]\n"
            context += f"Content: {doc['content']}\n"
            context += f"Metadata: {doc['metadata']}\n"

        return context.strip()
    
    
    def get_memory(self, session_id: str) -> ChatMemory:
        """
        Retrieve the ChatMemory instance for a given session.

        If no memory exists for the provided session_id, a new ChatMemory
        object is created, stored in MEMORY_STORE, and returned.

        Args:
            session_id (str): Unique identifier for the session.

        Returns:
            ChatMemory: The memory object associated with the session.
        """
        if session_id not in MEMORY_STORE:
            MEMORY_STORE[session_id] = ChatMemory(NUMBER_OF_CHATS)
        return MEMORY_STORE[session_id]
    
    def build_prompt(self, query:str, context: str, memory: ChatMemory):
        """
        Construct the system and user messages for the LLM.
        The system message contains the full prompt template with injected
        context, query, and prior conversation history. The user message
        contains the current raw query.

        Args:
            query (str):
                The user's input question.

            context (str):
                The formatted context string generated from retrieved documents.

            memory (ChatMemory):
            The memory object used to fetch past conversation history.

        Returns:
            List:
                A list of message objects (SystemMessage, HumanMessage)
                formatted for the language model.
        """
        chat_history = memory.get_context()
        system_message = SystemMessage(content=GENERATION_PROMPT.format(
            context=context, 
            query=query, 
            chat_history=chat_history
        ))
        user_message = HumanMessage(content=query)

        return [system_message, user_message]
    
    def generate_answer(self, query: str, session_id: str, top_k: int = 3):
        """
        Execute the full Retrieval-Augmented Generation (RAG) pipeline.

        This method performs:
            1. Document retrieval from the vector database
            2. Context formatting
            3. Prompt construction
            4. Streaming response generation from the LLM
            5. Appending source metadata at the end of the response

        Args:
            query (str):
                The user's query or question.
            
            session_id (str):
                Unique session id for each user.

            top_k (int, optional):
                Number of top documents to retrieve from the vector database.
                Defaults to 3.

        Yields:
            str:
                Chunks of the generated response streamed from the model.
                After completion, yields a formatted list of sources.

        Raises:
            Exception:
                Propagates errors from retrieval, prompt construction,
                or model invocation.

        Notes:
            - Uses streaming (`model.stream`) for incremental response output.g
        """
        logger.info("Starting Generation pipeline")

        # Setting up session memory
        memory = self.get_memory(session_id)
        
        # 1. Retrieve
        retrieved_docs = search_vector_db(query, top_k=top_k)
        logger.info(f"Retrieved Documents: {retrieved_docs}")

        # 2. Format context
        context = self.format_context(retrieved_docs)

        # 3. Build prompt
        messages = self.build_prompt(query, context, memory)

        # 4. Generate response
        logger.info("Calling Model to Generate the result")
        response = self.model.stream(messages)

        full_answer = ""

        for chunk in response:
            if hasattr(chunk, "content") and chunk.content:
                full_answer += chunk.content 
                yield chunk.content

        # send metadata at end
        yield "\n\n[SOURCES]\n"

        sources_text = ""

        for i, doc in enumerate(retrieved_docs, 1):
            metadata = doc.get("metadata", {})
            file_name = metadata.get("file_name", "Unknown")
            section = metadata.get("section", "Unknown")

            sources = f"{i}. {file_name} — *{section}*\n"
            sources_text += sources
            yield sources

        # Store memory per session
        memory.add(query, full_answer)