import google.generativeai as genai
import os
from typing import Optional, Dict, Any

class GeminiClient:
    def __init__(self, api_key: str):
        if not api_key:
            raise ValueError("Gemini API key is required")
        
        genai.configure(api_key=api_key)
        
        # Configure model settings
        self.generation_config = {
            "temperature": 0.8,  # Slightly creative but consistent
            "top_p": 0.9,
            "top_k": 40,
            "max_output_tokens": 1000,
        }
        
        # Safety settings
        self.safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH", 
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            }
        ]
        
        self.model = genai.GenerativeModel(
            model_name="gemini-pro",
            generation_config=self.generation_config,
            safety_settings=self.safety_settings
        )
        
        # Store conversation history for each twin
        self.conversations = {}
    
    def chat_with_twin(self, message: str, twin, rag_context: Optional[str] = None) -> str:
        """Generate a response for the specific twin"""
        try:
            # Get or create conversation for this twin
            twin_id = f"{twin.twin_type}_{twin.current_scenario}_{twin.rag_enabled}"
            
            if twin_id not in self.conversations:
                # Start new conversation with system prompt
                self.conversations[twin_id] = self.model.start_chat(history=[])
                
                # Send system prompt as first message
                system_prompt = twin.get_system_prompt()
                if rag_context:
                    system_prompt += f"\n\nADDITIONAL CONTEXT FROM YOUR MEMORIES:\n{rag_context}"
                
                # Send system prompt (this won't be visible to user)
                self.conversations[twin_id].send_message(
                    f"SYSTEM: {system_prompt}\n\nPlease acknowledge that you understand your role and are ready to chat."
                )
            
            # Send user message and get response
            chat = self.conversations[twin_id]
            
            # Add RAG context to current message if available
            if rag_context:
                enhanced_message = f"{message}\n\n[Additional context from your memories: {rag_context}]"
            else:
                enhanced_message = message
            
            response = chat.send_message(enhanced_message)
            
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini chat: {str(e)}")
            # Fallback response
            return self._get_fallback_response(twin.twin_type)
    
    def _get_fallback_response(self, twin_type: str) -> str:
        """Provide fallback responses when Gemini fails"""
        fallbacks = {
            'healthy': "I'm doing pretty well, thanks for asking! How about you?",
            'anxiety': "Sorry, I'm feeling a bit overwhelmed right now. Could you maybe ask me something else?",
            'depression': "I'm having a tough time right now... but I appreciate you talking with me."
        }
        return fallbacks.get(twin_type, "Sorry, I'm having trouble responding right now. Could you try again?")
    
    def reset_conversation(self, twin_type: str, scenario: str, rag_enabled: bool):
        """Reset conversation history for a twin"""
        twin_id = f"{twin_type}_{scenario}_{rag_enabled}"
        if twin_id in self.conversations:
            del self.conversations[twin_id]
    
    def generate_response(self, prompt: str, context: Optional[str] = None) -> str:
        """General purpose response generation"""
        try:
            full_prompt = prompt
            if context:
                full_prompt = f"{context}\n\n{prompt}"
            
            response = self.model.generate_content(full_prompt)
            return response.text
            
        except Exception as e:
            print(f"Error in Gemini generation: {str(e)}")
            return "I'm having trouble generating a response right now. Please try again."