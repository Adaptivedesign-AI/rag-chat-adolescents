from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import sqlite3
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Any
import logging
from pathlib import Path
import csv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'your-secret-key-here')

# Configure Gemini AI
GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY environment variable is required")

genai.configure(api_key=GEMINI_API_KEY)

# Initialize models
chat_model = genai.GenerativeModel('gemini-2.0-flash-exp')
embedding_model = genai.GenerativeModel('text-embedding-004')

class DigitalTwinDatabase:
    def __init__(self, db_path='digital_twins.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize the database with required tables"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Chat history table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                twin_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                sender TEXT NOT NULL,
                message TEXT NOT NULL,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                scenario TEXT DEFAULT 'neutral',
                rag_enabled BOOLEAN DEFAULT FALSE
            )
        ''')
        
        # Twin sessions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS twin_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                twin_id TEXT NOT NULL,
                session_id TEXT NOT NULL,
                current_scenario TEXT DEFAULT 'neutral',
                rag_enabled BOOLEAN DEFAULT FALSE,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_active DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def save_message(self, twin_id: str, session_id: str, sender: str, message: str, scenario: str = 'neutral', rag_enabled: bool = False):
        """Save a chat message to the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (twin_id, session_id, sender, message, scenario, rag_enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (twin_id, session_id, sender, message, scenario, rag_enabled))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, twin_id: str, session_id: str, limit: int = 20) -> List[Dict]:
        """Get recent chat history for a twin session"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT sender, message, timestamp, scenario, rag_enabled
            FROM chat_history
            WHERE twin_id = ? AND session_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        ''', (twin_id, session_id, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        # Reverse to get chronological order
        return [
            {
                'sender': row[0],
                'message': row[1],
                'timestamp': row[2],
                'scenario': row[3],
                'rag_enabled': row[4]
            }
            for row in reversed(rows)
        ]
    
    def update_session(self, twin_id: str, session_id: str, scenario: str = None, rag_enabled: bool = None):
        """Update session configuration"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Check if session exists
        cursor.execute('SELECT id FROM twin_sessions WHERE twin_id = ? AND session_id = ?', (twin_id, session_id))
        if cursor.fetchone():
            # Update existing session
            updates = []
            params = []
            if scenario is not None:
                updates.append('current_scenario = ?')
                params.append(scenario)
            if rag_enabled is not None:
                updates.append('rag_enabled = ?')
                params.append(rag_enabled)
            
            if updates:
                updates.append('last_active = CURRENT_TIMESTAMP')
                params.extend([twin_id, session_id])
                
                cursor.execute(f'''
                    UPDATE twin_sessions 
                    SET {', '.join(updates)}
                    WHERE twin_id = ? AND session_id = ?
                ''', params)
        else:
            # Create new session
            cursor.execute('''
                INSERT INTO twin_sessions (twin_id, session_id, current_scenario, rag_enabled)
                VALUES (?, ?, ?, ?)
            ''', (twin_id, session_id, scenario or 'neutral', rag_enabled or False))
        
        conn.commit()
        conn.close()

class RAGSystem:
    def __init__(self):
        self.scenarios_path = Path('scenarios')
        self.embeddings_cache = {}
    
    def load_embeddings(self, bucket_name: str) -> pd.DataFrame:
        """Load embeddings for a specific scenario bucket"""
        if bucket_name in self.embeddings_cache:
            return self.embeddings_cache[bucket_name]
        
        embeddings_path = self.scenarios_path / bucket_name / 'embeddings.parquet'
        chunks_path = self.scenarios_path / bucket_name / 'chunks.jsonl'
        
        try:
            if embeddings_path.exists():
                df = pd.read_parquet(embeddings_path)
                self.embeddings_cache[bucket_name] = df
                return df
            elif chunks_path.exists():
                # Fallback to JSONL if parquet doesn't exist
                chunks = []
                with open(chunks_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        chunks.append(json.loads(line.strip()))
                
                # Create basic DataFrame without embeddings
                df = pd.DataFrame(chunks)
                self.embeddings_cache[bucket_name] = df
                return df
            else:
                logger.warning(f"No embeddings found for bucket: {bucket_name}")
                return pd.DataFrame()
        except Exception as e:
            logger.error(f"Error loading embeddings for {bucket_name}: {e}")
            return pd.DataFrame()
    
    def search_memories(self, query: str, twin_id: str, scenario: str, mental_health_type: str, k: int = 10) -> List[str]:
        """Search for relevant memories based on query"""
        bucket_name = f"{scenario}_{mental_health_type}"
        df = self.load_embeddings(bucket_name)
        
        if df.empty:
            logger.warning(f"No memories available for bucket: {bucket_name}")
            return []
        
        try:
            # Generate query embedding
            query_embedding = self._get_embedding(query)
            
            if 'embedding' in df.columns:
                # Calculate cosine similarities
                similarities = []
                for idx, row in df.iterrows():
                    chunk_embedding = np.array(row['embedding'])
                    similarity = np.dot(query_embedding, chunk_embedding) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(chunk_embedding)
                    )
                    similarities.append((similarity, idx))
                
                # Sort by similarity and get top k
                similarities.sort(reverse=True)
                top_indices = [idx for _, idx in similarities[:k]]
                
                # Return top chunks
                return [df.iloc[idx]['text'] for idx in top_indices if 'text' in df.columns]
            else:
                # Fallback: return random chunks if no embeddings
                n_chunks = min(k, len(df))
                if 'text' in df.columns:
                    return df['text'].sample(n=n_chunks).tolist()
                elif 'chunk' in df.columns:
                    return df['chunk'].sample(n=n_chunks).tolist()
                else:
                    return []
        
        except Exception as e:
            logger.error(f"Error searching memories: {e}")
            return []
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Generate embedding for text using Gemini"""
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=text,
                task_type="retrieval_query"
            )
            return np.array(result['embedding'])
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return np.random.random(768)  # Fallback random embedding
    
    def internalize_memories(self, memories: List[str], twin_config: Dict, twin_profile: str) -> List[str]:
        """Convert external memories to first-person perspective"""
        if not memories:
            return []
        
        try:
            memories_text = '\n- '.join(memories)
            
            prompt = f"""You are roleplaying as a {twin_config['age']}-year-old {twin_config['mental_health_type']} adolescent.
You will be given external scene descriptions (from video transcripts).
Rewrite them as if they were **your own personal memories**, in first-person voice.

Character Profile: {twin_profile}

Rules:
- Only keep the parts that could belong to "me".
- Rewrite in natural language, not clinical labels.
- If something does not fit "my perspective", skip it.
- Do not mention that these are videos or external descriptions — write them as if they are purely my memories.
- Keep each memory short (1–2 sentences).
- Output strictly as a JSON list of strings.

External descriptions:
- {memories_text}

Output example format: ["I remember being left out of a group chat by my classmates.", "I remember panicking before giving a class presentation."]"""

            response = chat_model.generate_content(prompt)
            
            # Parse JSON response
            json_text = response.text.strip()
            if json_text.startswith('```json'):
                json_text = json_text[7:-3]
            elif json_text.startswith('```'):
                json_text = json_text[3:-3]
            
            internalized = json.loads(json_text)
            return internalized if isinstance(internalized, list) else []
            
        except Exception as e:
            logger.error(f"Error internalizing memories: {e}")
            # Fallback: return original memories with simple first-person conversion
            return [f"I remember {memory.lower()}" for memory in memories[:5]]

class TwinManager:
    def __init__(self, db: DigitalTwinDatabase, rag_system: RAGSystem):
        self.db = db
        self.rag_system = rag_system
        self.twin_profiles = self.load_twin_profiles()
        self.shared_prompt = self.load_shared_prompt()
    
    def load_twin_profiles(self) -> Dict[str, str]:
        """Load individual twin profiles from CSV"""
        profiles = {}
        try:
            with open('youth12_prompt.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['name'].lower()
                    profiles[name] = row['prompt']
        except Exception as e:
            logger.error(f"Error loading twin profiles: {e}")
        
        return profiles
    
    def load_shared_prompt(self) -> str:
        """Load shared roleplay prompt"""
        try:
            with open('shared_prompt.txt', 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            logger.error(f"Error loading shared prompt: {e}")
            return ""
    
    def get_twin_config(self, twin_id: str) -> Dict:
        """Get configuration for a specific twin"""
        # This should match the JavaScript configuration
        configs = {
            'kaiya': {'name': 'Kaiya', 'age': 15, 'grade': '9th', 'mental_health_type': 'healthy'},
            'ethan': {'name': 'Ethan', 'age': 14, 'grade': '9th', 'mental_health_type': 'healthy'},
            'lucas': {'name': 'Lucas', 'age': 17, 'grade': '12th', 'mental_health_type': 'healthy'},
            'maya': {'name': 'Maya', 'age': 16, 'grade': '11th', 'mental_health_type': 'healthy'},
            'jaden': {'name': 'Jaden', 'age': 15, 'grade': '10th', 'mental_health_type': 'anxiety'},
            'nia': {'name': 'Nia', 'age': 14, 'grade': '9th', 'mental_health_type': 'anxiety'},
            'hana': {'name': 'Hana', 'age': 17, 'grade': '12th', 'mental_health_type': 'anxiety'},
            'mateo': {'name': 'Mateo', 'age': 16, 'grade': '11th', 'mental_health_type': 'anxiety'},
            'diego': {'name': 'Diego', 'age': 15, 'grade': '10th', 'mental_health_type': 'depression'},
            'emily': {'name': 'Emily', 'age': 14, 'grade': '9th', 'mental_health_type': 'depression'},
            'amara': {'name': 'Amara', 'age': 18, 'grade': '12th', 'mental_health_type': 'depression'},
            'tavian': {'name': 'Tavian', 'age': 17, 'grade': '12th', 'mental_health_type': 'depression'}
        }
        return configs.get(twin_id, {})
    
    def generate_response(self, twin_id: str, user_message: str, session_id: str, scenario: str = 'neutral', rag_enabled: bool = False) -> str:
        """Generate a response from the digital twin"""
        twin_config = self.get_twin_config(twin_id)
        if not twin_config:
            return "Sorry, I don't know who I am supposed to be."
        
        # Get twin's individual profile
        twin_profile = self.twin_profiles.get(twin_id, "")
        
        # Build conversation history
        chat_history = self.db.get_chat_history(twin_id, session_id, limit=10)
        
        # Build prompt
        prompt_parts = [self.shared_prompt]
        
        if twin_profile:
            prompt_parts.append(f"\nYour specific profile: {twin_profile}")
        
        # Add scenario context
        scenario_context = self.get_scenario_context(scenario, twin_config['mental_health_type'])
        if scenario_context:
            prompt_parts.append(f"\nCurrent environment context: {scenario_context}")
        
        # Add RAG memories if enabled
        if rag_enabled:
            memories = self.rag_system.search_memories(
                user_message, twin_id, scenario, twin_config['mental_health_type']
            )
            if memories:
                internalized_memories = self.rag_system.internalize_memories(
                    memories, twin_config, twin_profile
                )
                if internalized_memories:
                    memory_text = '\n- '.join(internalized_memories)
                    prompt_parts.append(f"\nI recall some experiences from my past:\n- {memory_text}\nThese memories still shape how I feel today.")
        
        # Add recent chat history
        if chat_history:
            history_text = "\nRecent conversation:\n"
            for msg in chat_history[-5:]:  # Last 5 messages
                if msg['sender'] == 'user':
                    history_text += f"Human: {msg['message']}\n"
                elif msg['sender'] == 'assistant':
                    history_text += f"Me: {msg['message']}\n"
            prompt_parts.append(history_text)
        
        # Add current user message
        prompt_parts.append(f"\nHuman: {user_message}\nMe:")
        
        full_prompt = '\n'.join(prompt_parts)
        
        try:
            response = chat_model.generate_content(full_prompt)
            return response.text.strip()
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I'm having trouble thinking right now. Can you try again?"
    
    def get_scenario_context(self, scenario: str, mental_health_type: str) -> str:
        """Get scenario-specific context"""
        if scenario == 'neutral':
            return "You are in a balanced, supportive social environment with typical peer interactions."
        elif scenario == 'toxic':
            return "You are in a challenging, stressful social environment with negative peer pressure and conflict."
        return ""

# Initialize system components
db = DigitalTwinDatabase()
rag_system = RAGSystem()
twin_manager = TwinManager(db, rag_system)

@app.route('/')
def index():
    """Main page with twin selection"""
    return render_template('index.html')

@app.route('/chat/<twin_id>')
def chat(twin_id):
    """Chat page for specific twin"""
    twin_config = twin_manager.get_twin_config(twin_id)
    if not twin_config:
        return "Twin not found", 404
    
    # Create or get session ID
    if 'session_id' not in session:
        session['session_id'] = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + os.urandom(4).hex()
    
    return render_template('chat.html')

@app.route('/api/send_message', methods=['POST'])
def send_message():
    """Handle chat message"""
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        twin_id = data.get('twin_id')
        scenario = data.get('scenario', 'neutral')
        rag_enabled = data.get('rag_enabled', False)
        
        if not message or not twin_id:
            return jsonify({'error': 'Message and twin_id required'}), 400
        
        # Get or create session ID
        session_id = session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        session['session_id'] = session_id
        
        # Save user message
        db.save_message(twin_id, session_id, 'user', message, scenario, rag_enabled)
        
        # Generate response
        response = twin_manager.generate_response(twin_id, message, session_id, scenario, rag_enabled)
        
        # Save assistant response
        db.save_message(twin_id, session_id, 'assistant', response, scenario, rag_enabled)
        
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error in send_message: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/switch_scenario', methods=['POST'])
def switch_scenario():
    """Handle scenario switching"""
    try:
        data = request.get_json()
        twin_id = data.get('twin_id')
        scenario = data.get('scenario')
        
        if not twin_id or not scenario:
            return jsonify({'error': 'twin_id and scenario required'}), 400
        
        session_id = session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Update session
        db.update_session(twin_id, session_id, scenario=scenario)
        
        # Save scenario change message
        db.save_message(twin_id, session_id, 'system', f'Environment changed to {scenario}', scenario)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error in switch_scenario: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/toggle_rag', methods=['POST'])
def toggle_rag():
    """Handle RAG toggle"""
    try:
        data = request.get_json()
        twin_id = data.get('twin_id')
        rag_enabled = data.get('rag_enabled', False)
        
        if not twin_id:
            return jsonify({'error': 'twin_id required'}), 400
        
        session_id = session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        # Update session
        db.update_session(twin_id, session_id, rag_enabled=rag_enabled)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error in toggle_rag: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/chat_history/<twin_id>')
def get_chat_history(twin_id):
    """Get chat history for a twin"""
    try:
        session_id = session.get('session_id')
        if not session_id:
            return jsonify({'messages': []})
        
        history = db.get_chat_history(twin_id, session_id)
        return jsonify({'messages': history})
    
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
