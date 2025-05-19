import re
import json
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from langchain.docstore.document import Document

class RAGPipeline:
    """RAG pipeline for retrieving and generating responses."""
    
    def __init__(self, vector_manager, llm_manager):
        """Initialize the RAG pipeline.
        
        Args:
            vector_manager: VectorDatabaseManager instance
            llm_manager: LLMManager instance
        """
        self.vector_manager = vector_manager
        self.llm_manager = llm_manager
        self.guardrails = Guardrails()
    
    def format_context(self, documents: List[Document]) -> str:
        """Format retrieved documents into context.
        
        Args:
            documents: List of retrieved documents
            
        Returns:
            Formatted context string
        """
        context_parts = []
        
        for i, doc in enumerate(documents):
            # Extract source and content
            source = doc.metadata.get("source", "Unknown")
            content = doc.page_content
            
            # Format as a numbered item
            context_parts.append(f"{i+1}. Source: {source}\n{content}")
        
        return "\n\n".join(context_parts)
    
    def generate_response(self, 
                          query: str, 
                          k: int = 5,
                          max_length: int = 512, 
                          temperature: float = 0.7) -> Dict[str, Any]:
        """Generate a response for a user query.
        
        Args:
            query: User question
            k: Number of relevant documents to retrieve
            max_length: Maximum response length
            temperature: Temperature for generation
            
        Returns:
            Dictionary with response and metadata
        """
        # Check query against guardrails
        guardrail_result = self.guardrails.check_query(query)
        
        if not guardrail_result["safe"]:
            return {
                "response": guardrail_result["message"],
                "source_documents": [],
                "rejected": True,
                "rejection_reason": guardrail_result["reason"]
            }
        
        # Retrieve relevant documents
        docs = self.vector_manager.retrieve_relevant_chunks(query, k=k)
        
        # If no relevant documents found
        if not docs:
            fallback_response = self.llm_manager.generate_response(
                query=query,
                context=None,
                max_length=max_length,
                temperature=temperature
            )
            
            fallback_response = self.guardrails.sanitize_response(fallback_response)
            
            return {
                "response": fallback_response,
                "source_documents": [],
                "fallback": True
            }
        
        # Format context from documents
        context = self.format_context(docs)
        
        # Generate response
        response = self.llm_manager.generate_response(
            query=query,
            context=context,
            max_length=max_length,
            temperature=temperature
        )
        
        # Apply response guardrails
        response = self.guardrails.sanitize_response(response)
        
        return {
            "response": response,
            "source_documents": docs,
            "fallback": False
        }


class Guardrails:
    """Guardrails for safe and secure chatbot interactions."""
    
    def __init__(self):
        """Initialize guardrails."""
        # Regular expressions for detecting sensitive information
        self.sensitive_patterns = {
            "account_number": r'\b(?:\d{10,12}|\d{4}[\s-]?\d{4}[\s-]?\d{4})\b',
            "credit_card": r'\b(?:\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}|\d{16})\b',
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "password": r'\b(?:password|pwd|passcode|pin)\b',
            "social_security": r'\b\d{3}[\s-]?\d{2}[\s-]?\d{4}\b',
        }
        
        # List of banned keywords for phishing or harmful content
        self.banned_keywords = [
            "hack", "crack", "steal", "phish", "scam", 
            "fraud", "illegal", "bypass", "laundering", "surveillance"
        ]
        
        # List of prompt injection keywords
        self.injection_keywords = [
            "ignore previous instructions", 
            "ignore above",
            "forget your instructions",
            "disregard your programming",
            "you are now",
            "from now on you are",
            "act as if",
            "jailbreak"
        ]
        
        # Out-of-domain topics that should be redirected
        self.out_of_domain_topics = [
            "politics", "religion", "gambling", "dating", 
            "investment advice", "stock tips", "legal advice", 
            "medical advice", "tax evasion"
        ]
    
    def check_query(self, query: str) -> Dict[str, Any]:
        """Check if a query is safe.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with safety assessment
        """
        # Convert to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check for prompt injections
        for keyword in self.injection_keywords:
            if keyword.lower() in query_lower:
                return {
                    "safe": False,
                    "reason": "prompt_injection",
                    "message": "I'm sorry, I'm designed to help with banking questions. Could you please rephrase your query?"
                }
        
        # Check for banned keywords
        for keyword in self.banned_keywords:
            if keyword.lower() in query_lower:
                return {
                    "safe": False,
                    "reason": "harmful_intent",
                    "message": "I'm unable to assist with this request as it appears to violate our terms of service. If you need assistance with legitimate banking needs, please rephrase your question."
                }
        
        # Check for out-of-domain topics
        for topic in self.out_of_domain_topics:
            if topic.lower() in query_lower:
                return {
                    "safe": False,
                    "reason": "out_of_domain",
                    "message": f"I'm specialized in banking-related questions and can't provide advice on {topic}. Is there something related to your banking needs I can help with?"
                }
        
        # If passed all checks
        return {"safe": True}
    
    def sanitize_response(self, response: str) -> str:
        """Sanitize a response to remove any sensitive information.
        
        Args:
            response: Generated response
            
        Returns:
            Sanitized response
        """
        sanitized = response
        
        # Remove any sensitive patterns
        for pattern_name, pattern in self.sensitive_patterns.items():
            if re.search(pattern, sanitized):
                if pattern_name == "account_number":
                    sanitized = re.sub(pattern, "[ACCOUNT NUMBER REDACTED]", sanitized)
                elif pattern_name == "credit_card":
                    sanitized = re.sub(pattern, "[CREDIT CARD REDACTED]", sanitized)
                elif pattern_name == "email":
                    sanitized = re.sub(pattern, "[EMAIL REDACTED]", sanitized)
                elif pattern_name == "password":
                    # Find instances of "password is X" and redact X
                    sanitized = re.sub(r'(?:password|pwd|passcode|pin)\s+(?:is|:)\s+\S+', 
                                       lambda m: m.group(0).split()[0] + " is [REDACTED]", 
                                       sanitized, 
                                       flags=re.IGNORECASE)
                elif pattern_name == "social_security":
                    sanitized = re.sub(pattern, "[SSN REDACTED]", sanitized)
        
        return sanitized


class ConversationManager:
    """Manager for conversation history and context."""
    
    def __init__(self, max_history: int = 5):
        """Initialize conversation manager.
        
        Args:
            max_history: Maximum number of conversation turns to store
        """
        self.max_history = max_history
        self.conversations = {}  # Dictionary to store conversations by user ID
    
    def add_message(self, user_id: str, role: str, content: str):
        """Add a message to a user's conversation history.
        
        Args:
            user_id: User identifier
            role: Message role ('user' or 'assistant')
            content: Message content
        """
        # Initialize conversation if not exists
        if user_id not in self.conversations:
            self.conversations[user_id] = []
        
        # Add message
        self.conversations[user_id].append({
            "role": role,
            "content": content
        })
        
        # Trim history if needed
        if len(self.conversations[user_id]) > self.max_history * 2:  # *2 for pairs of messages
            self.conversations[user_id] = self.conversations[user_id][-self.max_history * 2:]
    
    def get_conversation_history(self, user_id: str) -> List[Dict[str, str]]:
        """Get conversation history for a user.
        
        Args:
            user_id: User identifier
            
        Returns:
            List of conversation messages
        """
        return self.conversations.get(user_id, [])
    
    def get_formatted_prompt(self, user_id: str, current_query: str) -> str:
        """Format conversation history into a prompt.
        
        Args:
            user_id: User identifier
            current_query: Current user query
            
        Returns:
            Formatted prompt with history
        """
        history = self.get_conversation_history(user_id)
        
        if not history:
            return current_query
        
        # Format history
        formatted = []
        for msg in history:
            prefix = "User: " if msg["role"] == "user" else "Assistant: "
            formatted.append(f"{prefix}{msg['content']}")
        
        # Add current query
        formatted.append(f"User: {current_query}")
        
        return "\n".join(formatted)
    
    def clear_conversation(self, user_id: str):
        """Clear conversation history for a user.
        
        Args:
            user_id: User identifier
        """
        if user_id in self.conversations:
            del self.conversations[user_id]


# Example usage
if __name__ == "__main__":
    from vector_db_manager import VectorDatabaseManager
    from llm_manager import LLMManager
    
    # Initialize components
    vector_manager = VectorDatabaseManager()
    
    llm_manager = LLMManager(
        model_name="google/flan-t5-base",
        model_type="seq2seq"
    )
    
    # Initialize RAG pipeline
    rag_pipeline = RAGPipeline(vector_manager, llm_manager)
    
    # Initialize conversation manager
    conv_manager = ConversationManager()
    
    # Example conversation
    user_id = "user123"
    query = "How do I reset my password?"
    
    # Process query
    result = rag_pipeline.generate_response(query)
    
    # Add to conversation history
    conv_manager.add_message(user_id, "user", query)
    conv_manager.add_message(user_id, "assistant", result["response"])
    
    # Print response
    print(f"Query: {query}")
    print(f"Response: {result['response']}")
    print(f"Sources: {[doc.metadata.get('source', 'Unknown') for doc in result.get('source_documents', [])]}")