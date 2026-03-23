from collections import deque

class ChatMemory:
    """
    A lightweight in-memory store for maintaining recent chat history.

    This class keeps track of the last `max_turns` interactions between
    the user and the bot. It provides methods to add new conversations
    and retrieve a formatted context string suitable for prompt injection.

    Attributes:
        history (deque):
            A fixed-size queue storing recent chat turns as dictionaries
            with 'user' and 'bot' keys.
    """
    def __init__(self, max_turns: int = 5):
        self.history = deque(maxlen=max_turns)

    def add(self, user_query: str, bot_response: str):
        """
        Add a new user-bot interaction to memory.

        Args:
            user_query (str):
                The user's input message.

            bot_response (str):
                The bot's generated response.
        """
        self.history.append({
            "user": user_query,
            "bot": bot_response
        })

    def get_context(self) -> str:
        """
        Generate a formatted string of the stored conversation history.

        Each conversation turn is labeled and structured for easy inclusion
        in LLM prompts.

        Returns:
            str:
                A formatted string containing recent chat history.
        """
        context = ""
        for i, chat in enumerate(self.history, 1):
            context += f"\n[Conversation {i}]\n"
            context += f"User: {chat['user']}\n"
            context += f"Bot: {chat['bot']}\n"
        return context.strip()