from flask import Flask, render_template, request, jsonify, session
import google.generativeai as genai
import sqlite3
import json
import numpy as np
from datetime import datetime
import os
from typing import List, Dict, Any
import logging
from pathlib import Path
import csv
import re
import pickle
import requests
from sklearn.metrics.pairwise import cosine_similarity
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type

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

# Use more stable model
chat_model = genai.GenerativeModel('gemini-2.0-flash')

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

class ImprovedRAGSystem:
    """
    Improved RAG system that properly handles embeddings alignment and directory naming
    """
    def __init__(self, scenarios_root: str = "scenarios"):
        self.scenarios_path = Path(scenarios_root)
        self.embeddings_cache: Dict[str, Dict[str, Any]] = {}
        
        # Directory alias mapping: external name -> actual directory name
        self.mh_dir_alias = {
            "anxiety": "anxious", 
            "anxious": "anxious",
            "healthy": "healthy",
            "depression": "depression"
        }
        
        # REST embedding configuration
        self._embed_model = "text-embedding-004"
        self._api_key = os.environ.get("GEMINI_API_KEY")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        
        # Clear potential proxy settings that could interfere
        for k in ["HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"]:
            os.environ.pop(k, None)
        
        self._embed_url = f"https://generativelanguage.googleapis.com/v1beta/models/{self._embed_model}:embedContent?key={self._api_key}"
        self._embed_timeout = 45

    def _resolve_bucket_dir(self, scenario: str, mh: str) -> Path:
        """
        Resolve bucket directory with alias support
        Try both original name and alias (e.g., anxiety -> anxious)
        """
        bn = f"{scenario}_{mh}"
        d1 = self.scenarios_path / bn
        if d1.exists():
            return d1
        
        # Try alias
        alt_mh = self.mh_dir_alias.get(mh, mh)
        d2 = self.scenarios_path / f"{scenario}_{alt_mh}"
        return d2

    def _read_chunks(self, chunks_path: Path) -> List[Dict[str, Any]]:
        """Read chunks from JSONL file"""
        rows = []
        if not chunks_path.exists():
            return rows
            
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Failed to parse line in {chunks_path}: {e}")
                    continue
        return rows

    def _load_bucket(self, scenario: str, mh: str) -> Dict[str, Any]:
        """
        Load and align a bucket, returning {'ids', 'texts', 'embeddings'}
        Properly aligns text chunks with their corresponding embeddings by ID
        """
        bucket_dir = self._resolve_bucket_dir(scenario, mh)
        chunks_path = bucket_dir / "chunks.jsonl"
        pkl_path = bucket_dir / "embeddings.pkl"

        if not chunks_path.exists():
            logger.warning(f"[RAG] No chunks.jsonl in {bucket_dir}")
            return {"ids": [], "texts": [], "embeddings": None}

        chunks = self._read_chunks(chunks_path)
        
        # Build id -> text mapping
        id2text = {}
        for ch in chunks:
            cid = ch.get("id")
            text = ch.get("text") or ch.get("summary") or ch.get("title") or ""
            if cid and text:
                id2text[cid] = text

        ids_aligned, texts_aligned, vecs_aligned = [], [], []

        if pkl_path.exists():
            try:
                with open(pkl_path, "rb") as f:
                    obj = pickle.load(f)  # 可能是 list[dict] / dict[id->vec] / list[(id, vec)]
        
                # 统一成可迭代的 (cid, emb)
                items = []
                if isinstance(obj, dict):
                    items = list(obj.items())
                elif isinstance(obj, list):
                    for rec in obj:
                        if isinstance(rec, dict):
                            cid = next((rec[k] for k in ("id", "chunk_id", "uid") if k in rec and rec[k] is not None), None)
                            emb = next((rec[k] for k in ("embedding", "values", "vector") if k in rec), None)
                            items.append((cid, emb))
                        elif isinstance(rec, (list, tuple)) and len(rec) == 2:
                            items.append((rec[0], rec[1]))
                else:
                    logger.warning(f"[RAG] Unsupported embeddings.pkl type: {type(obj)}")
        
                for cid, emb in items:
                    if not cid or cid not in id2text:
                        continue
                    # 兼容 ndarray / tuple / list
                    if isinstance(emb, np.ndarray):
                        emb = emb.astype(np.float32).tolist()
                    elif isinstance(emb, tuple):
                        emb = list(emb)
        
                    if isinstance(emb, list) and len(emb) > 0:
                        ids_aligned.append(cid)
                        texts_aligned.append(id2text[cid])
                        vecs_aligned.append(emb)
        
                if vecs_aligned:
                    embeddings = np.array(vecs_aligned, dtype=np.float32)
                    logger.info(f"[RAG] Loaded {len(texts_aligned)} aligned text-embedding pairs from {bucket_dir}")
                    return {"ids": ids_aligned, "texts": texts_aligned, "embeddings": embeddings}
                else:
                    logger.warning(f"[RAG] No usable embeddings vectors after normalization in {pkl_path}")
        
            except Exception as e:
                logger.error(f"[RAG] Failed to read {pkl_path}: {e}")

        # Fallback: return texts without pre-computed embeddings
        texts = list(id2text.values())
        ids = list(id2text.keys())
        logger.info(f"[RAG] Loaded {len(texts)} texts without pre-computed embeddings from {bucket_dir}")
        return {"ids": ids, "texts": texts, "embeddings": None}

    def load_embeddings_data(self, bucket_name: str) -> Dict[str, Any]:
        """
        Load embeddings data for a bucket name (backward compatibility)
        """
        if bucket_name in self.embeddings_cache:
            return self.embeddings_cache[bucket_name]

        if "_" in bucket_name:
            scenario, mh = bucket_name.split("_", 1)
        else:
            # Fallback: assume neutral scenario
            scenario, mh = "neutral", bucket_name

        data = self._load_bucket(scenario, mh)
        self.embeddings_cache[bucket_name] = data
        return data

    def _soft_sanitize(self, text: str) -> str:
        """
        温和化：仅在被拦截时二次尝试使用；将高敏词替换为中性占位，尽量不改变语义方向。
        """
        repl = [
            (r"\bsuicide\b", "self-harm mention"),
            (r"\bkill myself\b", "self-harm mention"),
            (r"\bcutting\b", "self-harm mention"),
            (r"\bgun\b", "weapon mention"),
            (r"\bknife\b", "weapon mention"),
            (r"\b(kill|die)\b", "harm mention"),
            (r"\b(chink|nigger|faggot|retard)\b", "a nasty slur"),
        ]
        s = text
        for pat, sub in repl:
            s = re.sub(pat, sub, s, flags=re.IGNORECASE)
        return s
    
    def _soften_list(self, items: List[str], limit: int = 8) -> List[str]:
        # 注意：只有在第一次尝试失败/被拦截后才会走到这里
        return [self._soft_sanitize(x) for x in items[:limit]]

    @retry(
        wait=wait_exponential(multiplier=1, min=1, max=20),
        stop=stop_after_attempt(6),
        retry=retry_if_exception_type((requests.Timeout, requests.ConnectionError))
    )
    def _embed_one(self, text: str) -> np.ndarray:
        """Generate embedding using REST API directly"""
        payload = {"content": {"parts": [{"text": text}]}}
        
        try:
            response = requests.post(
                self._embed_url, 
                json=payload, 
                timeout=self._embed_timeout
            )
            
            if response.status_code != 200:
                try:
                    err = response.json()
                except:
                    err = response.text
                raise RuntimeError(f"Embed HTTP {response.status_code}: {err}")
            
            data = response.json()
            embedding = data.get("embedding", {})
            vec = embedding.get("values")  # Note: might be "values" not "value"
            
            if isinstance(vec, list):
                return np.array(vec, dtype=np.float32)
            
            # Try alternative field names
            for field in ["value", "embedding"]:
                alt_vec = embedding.get(field)
                if isinstance(alt_vec, list):
                    return np.array(alt_vec, dtype=np.float32)
            
            raise RuntimeError(f"Unexpected embed response structure: {str(data)[:500]}")
            
        except Exception as e:
            logger.error(f"[RAG] REST embedding failed: {e}")
            raise

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with fallback to SDK if REST fails"""
        try:
            return self._embed_one(text)
        except Exception as e:
            logger.warning(f"[RAG] REST embedding failed, trying SDK fallback: {e}")
            try:
                # Fallback to SDK
                result = genai.embed_content(
                    model="models/text-embedding-004",
                    content=text,
                    task_type="retrieval_query"
                )
                return np.array(result['embedding'], dtype=np.float32)
            except Exception as e2:
                logger.error(f"[RAG] Both REST and SDK embedding failed, using random fallback: {e2}")
                return np.random.random(768).astype(np.float32)

    def search_memories(self, query: str, twin_id: str, scenario: str, mental_health_type: str, k: int = 10) -> List[str]:
        """
        Search for relevant memories with proper text-embedding alignment
        """
        # Try both original and alias directory names
        bucket_candidates = [
            f"{scenario}_{mental_health_type}",
            f"{scenario}_{self.mh_dir_alias.get(mental_health_type, mental_health_type)}"
        ]
        
        data = None
        for bucket_name in bucket_candidates:
            data = self.load_embeddings_data(bucket_name)
            if data["texts"]:
                logger.info(f"[RAG] Using bucket: {bucket_name}")
                break

        if not data or not data["texts"]:
            logger.warning(f"[RAG] No memories found for buckets: {bucket_candidates}")
            return []

        texts = data["texts"]
        embeddings_matrix = data["embeddings"]

        # Generate query embedding
        try:
            query_embedding = self.get_embedding(query).reshape(1, -1)
        except Exception as e:
            logger.error(f"[RAG] Failed to generate query embedding: {e}")
            return []

        if embeddings_matrix is not None and len(texts) == embeddings_matrix.shape[0]:
            # Use pre-computed embeddings
            similarities = cosine_similarity(query_embedding, embeddings_matrix)[0]
            top_indices = np.argsort(similarities)[::-1][:k]
            results = [texts[i] for i in top_indices if i < len(texts)]
            logger.info(f"[RAG] Found {len(results)} memories using pre-computed embeddings")
            return results
        else:
            # On-demand embedding computation (slower)
            logger.info(f"[RAG] Computing embeddings on-demand for {len(texts)} texts (this may be slow)")
            
            similarities = []
            query_vec = query_embedding.reshape(-1)
            query_norm = np.linalg.norm(query_vec) + 1e-8
            
            for i, text in enumerate(texts):
                try:
                    text_vec = self.get_embedding(text).reshape(-1)
                    text_norm = np.linalg.norm(text_vec) + 1e-8
                    similarity = float(np.dot(query_vec, text_vec) / (query_norm * text_norm))
                    similarities.append((similarity, i))
                except Exception as e:
                    logger.warning(f"[RAG] Failed to compute similarity for text {i}: {e}")
                    similarities.append((0.0, i))
            
            # Sort by similarity and return top k
            similarities.sort(reverse=True)
            results = [texts[i] for _, i in similarities[:k]]
            logger.info(f"[RAG] Found {len(results)} memories using on-demand embeddings")
            return results

    def internalize_memories(self, memories: List[str], twin_config: Dict, twin_profile: str) -> List[str]:
        """Convert external memories to first-person perspective with improved safety and retry logic"""
        if not memories:
            return []
    
        def _extract_json_list(txt: str):
            s = txt.strip()
            if s.startswith("```json"):
                s = s[7:-3].strip()
            elif s.startswith("```"):
                s = s[3:-3].strip()
            return json.loads(s) if s else []
    
        # 1) 先用原文尝试（不提前温和化）
        base_prompt = f"""You are roleplaying as a {twin_config['age']}-year-old {twin_config['mental_health_type']} adolescent.
    Rewrite the external scene descriptions as **my own memories** in first-person.
    Rules:
    - Keep only what plausibly belongs to "me"; 1–2 sentences each.
    - Do not mention videos/prompts.
    - Output **JSON list of strings** only.
    
    Character Profile: {twin_profile}
    
    External descriptions:
    - """ + "\n- ".join(memories)
    
        try:
            r1 = chat_model.generate_content(
                base_prompt,
                generation_config={"temperature": 0.3},
            )
            t1 = r1.text if hasattr(r1, "text") else r1.candidates[0].content.parts[0].text
            p1 = _extract_json_list(t1)
            if isinstance(p1, list) and p1:
                return p1
        except Exception as e:
            logger.warning(f"[internalize] first attempt failed: {e}")
    
        # 2) 若失败/被拦截 → 温和化后重试
        softened = self._soften_list(memories, limit=8)
        safe_prompt = f"""You are roleplaying as a {twin_config['age']}-year-old {twin_config['mental_health_type']} adolescent.
    Rewrite the external scene descriptions as **my own memories** in first-person.
    Rules:
    - Keep only what plausibly belongs to "me"; 1–2 sentences each.
    - Do not mention videos/prompts.
    - Output **JSON list of strings** only.
    
    Character Profile: {twin_profile}
    
    External descriptions:
    - """ + "\n- ".join(softened) + """
    
    Safety constraints:
    - Paraphrase any harmful or explicit content gently (e.g., "a hurtful nickname").
    - Do not include plans/instructions for self-harm, violence, or retaliation.
    """
        try:
            r2 = chat_model.generate_content(
                safe_prompt,
                generation_config={"temperature": 0.2},
            )
            t2 = r2.text if hasattr(r2, "text") else r2.candidates[0].content.parts[0].text
            p2 = _extract_json_list(t2)
            if isinstance(p2, list) and p2:
                return p2
        except Exception as e:
            logger.warning(f"[internalize] second attempt failed: {e}")
    
        # 3) Fallback
        return [f"I remember {m.lower()}" for m in memories[:5]]

class TwinManager:
    def __init__(self, db: DigitalTwinDatabase, rag_system: ImprovedRAGSystem):
        self.db = db
        self.rag_system = rag_system
        self.twin_profiles = self.load_twin_profiles()
        self.shared_prompt = self.load_shared_prompt()

    def _enforce_emotion_breaks(self, text: str) -> str:
        """
        在最后一次出现的 'Emotion tag: ...' 或 'Emotional tag: ...' 之前插入**两个换行**。
        仅对最后一个匹配生效，避免中间段落被改动。
        """
        if not text:
            return text
    
        # 兼容 Emotion tag / Emotional tag（大小写不敏感）
        pattern = re.compile(r"(?:^|\n)[ \t]*(Emotional?\s*tag\s*:\s*[^\n]+)\s*$", re.IGNORECASE)
    
        last = None
        for m in pattern.finditer(text):
            last = m
        if not last:
            return text
    
        start = last.start(1)
        before, tagline = text[:start], text[start:]
    
        # 去掉标签前多余的结尾换行，然后补上恰好两个
        before = before.rstrip("\n")
        return before + "\n\n" + tagline
    
    def load_twin_profiles(self) -> Dict[str, str]:
        """Load individual twin profiles from CSV"""
        profiles = {}
        try:
            with open('youth12_prompt.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    name = row['name'].lower()
                    profiles[name] = row['prompt']
            logger.info(f"Loaded {len(profiles)} twin profiles")
        except Exception as e:
            logger.error(f"Error loading twin profiles: {e}")
        
        return profiles
    
    def load_shared_prompt(self) -> str:
        """Load shared roleplay prompt"""
        try:
            with open('shared_prompt.txt', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info("Loaded shared prompt")
                return content
        except Exception as e:
            logger.error(f"Error loading shared prompt: {e}")
            return ""
    
    def get_twin_config(self, twin_id: str) -> Dict:
        """Get configuration for a specific twin"""
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
            try:
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
                        logger.info(f"[RAG] Added {len(internalized_memories)} internalized memories for {twin_id}")
            except Exception as e:
                logger.error(f"[RAG] Error retrieving memories for {twin_id}: {e}")
        
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
            txt = response.text.strip()
            txt = self._enforce_emotion_breaks(txt)  # ← 新增这一行
            return txt
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I'm having trouble thinking right now. Can you try again?"
    
    def get_scenario_context(self, scenario: str, mental_health_type: str) -> str:
        """Get scenario-specific context with fixed quote escaping"""
        if scenario == 'neutral':
            return (
                "ENVIRONMENT (neutral)\n"
                "- School climate is generally supportive; peers are friendly in class and during breaks.\n"
                "- Group chats are mostly about homework, memes, weekend plans; jokes stay light and inclusive.\n"
                "- Teachers respond fairly; conflicts are rare and usually resolved quickly.\n"
                "- You feel comfortable using your phone in public spaces and sitting with your usual group at lunch.\n"
                "\nROLEPLAY HINTS\n"
                "- Keep tone balanced and curious. Small worries are normal but don't dominate.\n"
                "- Be receptive to the user's questions; offer everyday details (class periods, clubs, hobbies) when helpful.\n"
            )
    
        if scenario == 'toxic':
            return (
                "ENVIRONMENT (toxic — bullying focus)\n"
                "- Setting: A U.S. public high school during the semester. Between classes, in hallways near lockers, and at lunch.\n"
                "- Offline bullying: A small group repeatedly targets you. They whisper and laugh when you pass; one person mutters a hurtful nickname.\n"
                "  At your locker, a classmate 'accidentally' bumps your shoulder; another films you on their phone while smirking.\n"
                "  In class, they swap notes and glance back at you; it feels obvious but hard to prove.\n"
                "- Online bullying: In a group chat you were added to, screenshots and inside jokes circulate about you.\n"
                "  Your posts get sarcastic emoji reactions; a rumor thread spreads with edited screenshots.\n"
                "  You sometimes wake up to dozens of notifications and feel a drop in your stomach before you even look.\n"
                "- Social dynamics: A few bystanders notice but stay quiet. One friend is kind in private but avoids speaking up publicly.\n"
                "- Adults: A teacher has 'talked to the class' before, but enforcement feels inconsistent. You're unsure reporting will help.\n"
                "\nINTERNAL STATE CUES\n"
                f"- Baseline MH type: {mental_health_type}. Adjust tone accordingly:\n"
                "  • healthy → You try to minimize it, but it still stings; you problem-solve and look for allies.\n"
                "  • anxiety → Your mind loops on what-ifs; you scan halls/group chats, afraid of being singled out.\n"
                "  • depression → You feel drained and isolated; it's hard to believe things will change.\n"
                "\nROLEPLAY HINTS\n"
                "- Stay first-person. Describe concrete moments: hallway timing, locker sounds, lunch seating, phone buzzing, specific class transitions.\n"
                "- Keep it real but safe: do not describe self-harm, graphic violence, or retaliation plans.\n"
                "- If the user asks about coping or help, you can mention safe steps (mute/report, save evidence, talk to a trusted adult/counselor, sit with supportive peers).\n"
                "- Avoid explicit slurs: paraphrase as 'a hurtful nickname' or 'a nasty comment'.\n"
            )
    
        return (
            "ENVIRONMENT\n"
            "- You are in a challenging, stressful social environment with ongoing peer conflict.\n"
            "ROLEPLAY HINTS\n"
            "- Keep descriptions concrete and first-person. Stay within safety guidelines.\n"
        )

# Initialize system components
db = DigitalTwinDatabase()
rag_system = ImprovedRAGSystem()
twin_manager = TwinManager(db, rag_system)

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat/<twin_id>')
def chat(twin_id):
    twin_config = twin_manager.get_twin_config(twin_id)
    if not twin_config:
        return "Twin not found", 404
    
    if 'session_id' not in session:
        session['session_id'] = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + os.urandom(4).hex()
    
    return render_template('chat.html')

@app.route('/api/send_message', methods=['POST'])
def send_message():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        twin_id = data.get('twin_id')
        scenario = data.get('scenario', 'neutral')
        rag_enabled = data.get('rag_enabled', False)
        
        if not message or not twin_id:
            return jsonify({'error': 'Message and twin_id required'}), 400
        
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
    try:
        data = request.get_json()
        twin_id = data.get('twin_id')
        scenario = data.get('scenario')
        
        if not twin_id or not scenario:
            return jsonify({'error': 'twin_id and scenario required'}), 400
        
        session_id = session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        db.update_session(twin_id, session_id, scenario=scenario)
        db.save_message(twin_id, session_id, 'system', f'Environment changed to {scenario}', scenario)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error in switch_scenario: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/toggle_rag', methods=['POST'])
def toggle_rag():
    try:
        data = request.get_json()
        twin_id = data.get('twin_id')
        rag_enabled = data.get('rag_enabled', False)
        
        if not twin_id:
            return jsonify({'error': 'twin_id required'}), 400
        
        session_id = session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        
        db.update_session(twin_id, session_id, rag_enabled=rag_enabled)
        
        return jsonify({'success': True})
    
    except Exception as e:
        logger.error(f"Error in toggle_rag: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/api/debug_rag', methods=['POST'])
def debug_rag():
    """Debug endpoint to see RAG retrieval results"""
    try:
        data = request.get_json()
        query = data.get('query', '')
        twin_id = data.get('twin_id')
        scenario = data.get('scenario', 'neutral')
        
        if not query or not twin_id:
            return jsonify({'error': 'Query and twin_id required'}), 400
        
        twin_config = twin_manager.get_twin_config(twin_id)
        if not twin_config:
            return jsonify({'error': 'Invalid twin_id'}), 400
        
        # Get raw memories
        memories = rag_system.search_memories(
            query, twin_id, scenario, twin_config['mental_health_type'], k=5
        )
        
        # Get internalized memories
        twin_profile = twin_manager.twin_profiles.get(twin_id, "")
        internalized = rag_system.internalize_memories(memories, twin_config, twin_profile)
        
        return jsonify({
            'query': query,
            'raw_memories': memories,
            'internalized_memories': internalized,
            'bucket': f"{scenario}_{twin_config['mental_health_type']}"
        })
    
    except Exception as e:
        logger.error(f"Error in debug_rag: {e}")
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
