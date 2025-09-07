import json
import os
from typing import List, Dict, Any, Optional
import re

class RAGSystem:
    def __init__(self):
        self.knowledge_bases = {}
        self.load_knowledge_bases()
    
    def load_knowledge_bases(self):
        """Load all knowledge base files"""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'knowledge_base')
        
        # Define the six knowledge base files
        kb_files = [
            'healthy_neutral.json',
            'healthy_toxic.json', 
            'anxiety_neutral.json',
            'anxiety_toxic.json',
            'depression_neutral.json',
            'depression_toxic.json'
        ]
        
        for filename in kb_files:
            file_path = os.path.join(data_dir, filename)
            
            # Extract twin_type and scenario from filename
            parts = filename.replace('.json', '').split('_')
            twin_type = parts[0]
            scenario = parts[1]
            
            key = f"{twin_type}_{scenario}"
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        self.knowledge_bases[key] = json.load(f)
                        print(f"Loaded knowledge base: {key}")
                except Exception as e:
                    print(f"Error loading {filename}: {e}")
                    self.knowledge_bases[key] = self._create_placeholder_kb(twin_type, scenario)
            else:
                # Create placeholder if file doesn't exist
                self.knowledge_bases[key] = self._create_placeholder_kb(twin_type, scenario)
                print(f"Created placeholder knowledge base: {key}")
    
    def _create_placeholder_kb(self, twin_type: str, scenario: str) -> Dict:
        """Create placeholder knowledge base structure"""
        return {
            "metadata": {
                "twin_type": twin_type,
                "scenario": scenario,
                "description": f"Knowledge base for {twin_type} twin in {scenario} environment"
            },
            "memories": [
                {
                    "id": f"{twin_type}_{scenario}_placeholder",
                    "content": f"This is a placeholder memory for {twin_type} in {scenario} context.",
                    "keywords": [twin_type, scenario, "placeholder"],
                    "emotional_context": "neutral",
                    "relevance_score": 0.5,
                    "source": "placeholder"
                }
            ]
        }
    
    def retrieve_relevant_info(self, query: str, twin_type: str, scenario: str, top_k: int = 3) -> Optional[str]:
        """Retrieve relevant information from knowledge base"""
        try:
            key = f"{twin_type}_{scenario}"
            
            if key not in self.knowledge_bases:
                print(f"Knowledge base not found: {key}")
                return None
            
            kb = self.knowledge_bases[key]
            memories = kb.get('memories', [])
            
            if not memories:
                return None
            
            # Simple keyword-based retrieval
            query_words = self._extract_keywords(query.lower())
            scored_memories = []
            
            for memory in memories:
                score = self._calculate_relevance_score(query_words, memory)
                if score > 0:
                    scored_memories.append((memory, score))
            
            # Sort by relevance score
            scored_memories.sort(key=lambda x: x[1], reverse=True)
            
            # Get top k memories
            top_memories = scored_memories[:top_k]