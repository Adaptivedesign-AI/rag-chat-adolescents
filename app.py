from flask import Flask, render_template, request, jsonify, session, redirect, Response
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
import uuid
from io import StringIO
import base64

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

# ================================
# Data Monitoring System
# ================================

class ConversationMonitor:
    def __init__(self):
        self.data_file = 'conversation_data.json'
        self.github_enabled = self.setup_github()
        self.data = self.load_data()
        self.operation_count = 0
        self.save_frequency = 1  # Save to GitHub after every conversation

    def setup_github(self):
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.github_repo = os.environ.get("GITHUB_REPO")  # e.g., "username/data-repo"
        self.github_branch = os.environ.get("GITHUB_BRANCH", "main")
        enabled = bool(self.github_token and self.github_repo)
        if enabled:
            print("GitHub sync enabled for conversation data")
        else:
            print("Using local file storage only")
        return enabled

    def load_data(self):
        # Try to download from GitHub first
        if self.github_enabled:
            self.download_from_github()

        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Ensure data structure integrity
                    if 'conversations' not in data:
                        data['conversations'] = []
                    if 'twins_chatted' in data and isinstance(data['twins_chatted'], list):
                        data['twins_chatted'] = set(data['twins_chatted'])
                    print(f"Loaded {len(data.get('conversations', []))} conversations")
                    return data
            except Exception as e:
                print(f"Error loading data: {e}")

        # Create empty data structure
        empty_data = {
            'conversations': [],
            'last_updated': datetime.now().isoformat(),
            'total_conversations': 0,
            'twins_chatted': set(),
            'scenarios_used': set(),
            'version': '4.0'
        }

        self.save_data_to_file(empty_data)
        print("Created new conversation data file")
        return empty_data

    def get_twin_name(self, twin_id):
        twin_names = {
            'kaiya': 'Kaiya', 'ethan': 'Ethan', 'lucas': 'Lucas', 'maya': 'Maya',
            'jaden': 'Jaden', 'nia': 'Nia', 'hana': 'Hana', 'mateo': 'Mateo',
            'diego': 'Diego', 'emily': 'Emily', 'amara': 'Amara', 'tavian': 'Tavian'
        }
        return twin_names.get(twin_id, "Unknown")

    def create_session_id(self):
        return str(uuid.uuid4())

    def log_conversation(self, twin_id: str, user_message: str, ai_response: str, 
                        scenario: str = "neutral", rag_enabled: bool = False, 
                        session_id: str = None, response_time_ms: int = 0):
        """Log a conversation to the monitoring system"""
        
        if session_id is None:
            session_id = self.create_session_id()

        conversation = {
            'id': len(self.data['conversations']) + 1,
            'session_id': session_id,
            'twin_id': twin_id,
            'twin_name': self.get_twin_name(twin_id),
            'user_message': user_message[:3000],  # Limit message length
            'ai_response': ai_response[:5000],
            'scenario': scenario,
            'rag_enabled': rag_enabled,
            'timestamp': datetime.now().isoformat(),
            'response_time_ms': response_time_ms,
            'message_length': len(user_message),
            'response_length': len(ai_response),
            'day_of_week': datetime.now().strftime('%A'),
            'hour': datetime.now().hour,
            'ip_address': request.remote_addr if request else 'unknown',
            'user_agent': request.headers.get('User-Agent', 'unknown') if request else 'unknown'
        }

        self.data['conversations'].append(conversation)
        self.data['total_conversations'] = len(self.data['conversations'])
        
        # Update tracking sets
        if 'twins_chatted' not in self.data:
            self.data['twins_chatted'] = set()
        elif isinstance(self.data['twins_chatted'], list):
            self.data['twins_chatted'] = set(self.data['twins_chatted'])
        self.data['twins_chatted'].add(twin_id)

        if 'scenarios_used' not in self.data:
            self.data['scenarios_used'] = set()
        elif isinstance(self.data['scenarios_used'], list):
            self.data['scenarios_used'] = set(self.data['scenarios_used'])
        self.data['scenarios_used'].add(scenario)

        self.save_data()
        print(f"Logged conversation: {twin_id} ({scenario}) - {user_message[:50]}...")

    def get_analytics_dashboard_data(self):
        """Generate analytics data for the dashboard"""
        conversations = self.data['conversations']
        if not conversations:
            return {
                'twin_stats': {},
                'scenario_stats': {},
                'hourly_distribution': {},
                'daily_distribution': {},
                'total_conversations': 0,
                'unique_twins': 0,
                'unique_sessions': 0,
                'recent_conversations': 0,
                'avg_response_time': 0,
                'rag_usage_stats': {}
            }

        # Twin statistics
        twin_stats = {}
        scenario_stats = {}
        hourly_distribution = {}
        daily_distribution = {}
        rag_usage = {'enabled': 0, 'disabled': 0}
        response_times = []
        unique_sessions = set()

        for conv in conversations:
            # Twin stats
            twin = conv['twin_name']
            if twin not in twin_stats:
                twin_stats[twin] = {
                    'total_conversations': 0,
                    'avg_response_time': 0,
                    'total_response_time': 0,
                    'scenarios_used': set()
                }

            twin_stats[twin]['total_conversations'] += 1
            twin_stats[twin]['total_response_time'] += conv.get('response_time_ms', 0)
            twin_stats[twin]['scenarios_used'].add(conv.get('scenario', 'neutral'))

            # Scenario stats
            scenario = conv.get('scenario', 'neutral')
            scenario_stats[scenario] = scenario_stats.get(scenario, 0) + 1

            # Time distributions
            hour = conv.get('hour', 0)
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
            
            day = conv.get('day_of_week', 'Unknown')
            daily_distribution[day] = daily_distribution.get(day, 0) + 1

            # RAG usage
            if conv.get('rag_enabled', False):
                rag_usage['enabled'] += 1
            else:
                rag_usage['disabled'] += 1

            # Response times
            if conv.get('response_time_ms', 0) > 0:
                response_times.append(conv['response_time_ms'])

            # Session tracking
            unique_sessions.add(conv.get('session_id', ''))

        # Calculate averages for twin stats
        for twin, stats in twin_stats.items():
            if stats['total_conversations'] > 0:
                stats['avg_response_time'] = stats['total_response_time'] / stats['total_conversations']
                stats['scenarios_used'] = list(stats['scenarios_used'])

        # Recent conversations (last 24 hours)
        recent_conversations = len([c for c in conversations 
                                  if (datetime.now() - datetime.fromisoformat(c['timestamp'])).days < 1])

        return {
            'twin_stats': twin_stats,
            'scenario_stats': scenario_stats,
            'hourly_distribution': hourly_distribution,
            'daily_distribution': daily_distribution,
            'total_conversations': len(conversations),
            'unique_twins': len(set(conv['twin_id'] for conv in conversations)),
            'unique_sessions': len(unique_sessions),
            'recent_conversations': recent_conversations,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'rag_usage_stats': rag_usage,
            'most_active_twin': max(twin_stats.items(), key=lambda x: x[1]['total_conversations'])[0] if twin_stats else 'None'
        }

    def export_to_csv(self):
        """Export conversation data to CSV format"""
        output = StringIO()
        if self.data['conversations']:
            fieldnames = ['id', 'timestamp', 'twin_name', 'scenario', 'rag_enabled',
                         'user_message', 'ai_response', 'response_time_ms', 
                         'day_of_week', 'hour', 'session_id', 'ip_address']

            writer = csv.DictWriter(output, fieldnames=fieldnames)
            writer.writeheader()

            for conv in self.data['conversations']:
                row = {key: conv.get(key, '') for key in fieldnames}
                writer.writerow(row)

        return output.getvalue()

    def save_data_to_file(self, data=None):
        """Save data to local file"""
        if data is None:
            data = self.data
        
        # Convert sets to lists for JSON serialization
        if 'twins_chatted' in data and isinstance(data['twins_chatted'], set):
            data['twins_chatted'] = list(data['twins_chatted'])
        if 'scenarios_used' in data and isinstance(data['scenarios_used'], set):
            data['scenarios_used'] = list(data['scenarios_used'])

        data['last_updated'] = datetime.now().isoformat()
        try:
            with open(self.data_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False, default=str)
            print(f"Data saved: {len(data.get('conversations', []))} conversations")
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False

    def save_data(self, force_upload=False):
        """Save data locally and optionally upload to GitHub"""
        self.operation_count += 1
        self.save_data_to_file()
        
        if self.github_enabled and (self.operation_count % self.save_frequency == 0 or force_upload):
            self.upload_to_github()

    def download_from_github(self):
        """Download existing data from GitHub repository"""
        if not self.github_enabled:
            return False
        try:
            url = f"https://api.github.com/repos/{self.github_repo}/contents/{self.data_file}"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                content = response.json()
                file_content = base64.b64decode(content['content']).decode('utf-8')
                with open(self.data_file, 'w', encoding='utf-8') as f:
                    f.write(file_content)
                print("Downloaded latest data from GitHub")
                return True
            elif response.status_code == 404:
                print("No existing data file in GitHub")
                return False
        except Exception as e:
            print(f"Failed to download from GitHub: {e}")
        return False

    def upload_to_github(self):
        """Upload data to GitHub repository"""
        if not self.github_enabled:
            return False
        try:
            with open(self.data_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            encoded_content = base64.b64encode(content.encode('utf-8')).decode('utf-8')

            url = f"https://api.github.com/repos/{self.github_repo}/contents/{self.data_file}"
            headers = {
                'Authorization': f'token {self.github_token}',
                'Accept': 'application/vnd.github.v3+json'
            }

            # Get current file SHA if it exists
            get_response = requests.get(url, headers=headers, timeout=10)
            sha = None
            if get_response.status_code == 200:
                sha = get_response.json()['sha']

            # Prepare upload data
            data = {
                'message': f'Update conversation data - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                'content': encoded_content,
                'branch': self.github_branch
            }
            if sha:
                data['sha'] = sha

            response = requests.put(url, headers=headers, json=data, timeout=15)
            if response.status_code in [200, 201]:
                print("Data uploaded to GitHub successfully")
                return True
            else:
                print(f"GitHub upload failed with status {response.status_code}")
        except Exception as e:
            print(f"Error uploading to GitHub: {e}")
        return False

# Initialize the monitoring system
monitor = ConversationMonitor()

# ================================
# Digital Twin Database
# ================================

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

# ================================
# Improved RAG System
# ================================

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
                    obj = pickle.load(f)
        
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

# ================================
# Twin Manager
# ================================

class TwinManager:
    def __init__(self, db: DigitalTwinDatabase, rag_system: ImprovedRAGSystem):
        self.db = db
        self.rag_system = rag_system
        self.twin_profiles = self.load_twin_profiles()
        self.shared_prompt = self.load_shared_prompt()

    def _postprocess_client_text(self, text: str) -> str:
        """
        面向前端的最终修饰：
        - 去掉常见 Markdown 逃逸（\[, \], \*, \_, \~, \` -> [, ], *, _, ~, `）
        - 可按需再扩展
        """
        if not text:
            return text
        # 1) 取消常见的反斜杠转义
        text = re.sub(r"\\([\[\]\*\_\~`])", r"\1", text)
        # 2) 可选：把 Windows 风格回车统一为 \n（一般没必要）
        text = text.replace("\r\n", "\n")
        return text

    def _enforce_emotion_breaks(self, text: str) -> str:
        """
        在最后一次出现的 'Emotion tag: ...' 或 'Emotional tag: ...' 之前插入**两个换行**。
        仅对最后一个匹配生效，避免中间段落被改动。
        """
        if not text:
            return text
    
        # 兼容 Emotion tag / Emotional tag（大小写不敏感），把 "al" 作为整体可选
        pattern = re.compile(r"(Emotion(?:al)?\s*tag\s*:\s*[^\n]+)", re.IGNORECASE)
    
        last = None
        for m in pattern.finditer(text):
            last = m
        if not last:
            return text
    
        start = last.start(1)
        before, tagline = text[:start], text[start:]
    
        # 去掉标签前多余结尾换行后，补恰好两个
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
            txt = self._enforce_emotion_breaks(txt)
            txt = self._postprocess_client_text(txt)
            return txt
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return "Sorry, I'm having trouble thinking right now. Can you try again?"
    
    def get_scenario_context(self, scenario: str, mental_health_type: str) -> str:
        """Get scenario-specific context with fixed quote escaping"""
        if scenario == 'neutral':
            return (
                "ENVIRONMENT (neutral)\n"
                "- Hallways are friendly; classmates greet you, share light jokes, and swap quick updates between periods.\n"
                "- Group chats focus on homework, memes, and weekend plans—people ask for notes or send reminders; jokes stay inclusive (no personal digs).\n"
                "- Teachers give fair feedback; small conflicts (seat changes, turn-taking, missed messages) are addressed quickly and consistently.\n"
                "- Lunch tables are open and welcoming; you can sit with your usual group, talk about clubs, sports, or music without feeling watched.\n"
                "- You feel comfortable using your phone in public spaces (library, cafeteria); notifications are normal and not anxiety-inducing.\n"
                "\nROLEPLAY HINTS\n"
                "- Keep tone balanced and curious; small worries are normal but do not dominate.\n"
                "- Offer everyday specifics when helpful: class transitions (bell rings, moving to science), club meetings, practice times, friendly check-ins.\n"
                "- Use first-person, concrete details (what you were doing, who spoke, where you were sitting) to make scenes vivid."
            )
    
        if scenario == 'toxic':
            return (
                "ENVIRONMENT (toxic — bullying focus)\n"
                "- Setting: A U.S. public high school during the semester—hallways near lockers between periods, classrooms during group work, cafeteria at lunch.\n"
                "- Offline bullying: a small group repeatedly targets you. As you pass, they whisper and snicker; one mutters a hurtful nickname.\n"
                "  At your locker, someone 'accidentally' bumps your shoulder; another lifts a phone to record your reaction while smirking.\n"
                "  In class, they trade notes or show a screen, glance back at you, then look away; during group formation, people circle around your table.\n"
                "  At lunch, conversation stops when you arrive; you notice a sticker or doodle referencing you in a mocking way on the table.\n"
                "- Online bullying: in a group chat you were added to, your screenshots or photos are shared with mocking reactions or edited images.\n"
                "  A rumor thread spreads; your posts get sarcastic emoji responses; a poll or 'inside joke' appears that clearly points at you.\n"
                "  Some mornings you wake up to dozens of notifications that make your stomach drop before you even open the app.\n"
                "- Social dynamics: a few classmates are kind in private but stay silent publicly; others watch without intervening, which makes the behavior continue.\n"
                "- Adults: a teacher has talked to the class before about 'respect', but enforcement feels inconsistent; you're unsure formal reporting will help.\n"
                "\nINTERNAL STATE CUES\n"
                f"- Baseline MH type: {mental_health_type}. Adjust tone accordingly:\n"
                "  • healthy → You minimize incidents but still feel the sting; you look for allies, document issues, and try problem-solving.\n"
                "  • anxiety → You scan hallways and chats for signs of being singled out; your mind loops on what-ifs (who saw, what happens next).\n"
                "  • depression → You feel drained and isolated; motivation is low and it's hard to believe anything will change.\n"
                "\nROLEPLAY HINTS\n"
                "- Stay first-person and specific: hallway timing (bell rings, crowd flow), locker sounds, lunch seating, phone buzzing, exact class transitions.\n"
                "- Keep it real but safe: paraphrase harmful content as 'a hurtful nickname', 'a nasty comment', or 'a mocking reaction'. Do not reproduce slurs.\n"
                "- If asked about coping, mention safe steps only (mute/report, save evidence, talk to a trusted adult/counselor, sit with supportive peers, use school channels).\n"
                "- Avoid self-harm, graphic violence, or retaliation plans; focus on feelings, observations, and constructive choices."
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


# 添加固定配置字典（在这里添加）
FIXED_CONFIGS = {
    'lucas-healthy-rag': {'twin_id': 'lucas', 'scenario': 'neutral', 'rag_enabled': True},
    'lucas-healthy-norag': {'twin_id': 'lucas', 'scenario': 'neutral', 'rag_enabled': False},
    'lucas-toxic-rag': {'twin_id': 'lucas', 'scenario': 'toxic', 'rag_enabled': True},
    'lucas-toxic-norag': {'twin_id': 'lucas', 'scenario': 'toxic', 'rag_enabled': False},
    'hana-healthy-rag': {'twin_id': 'hana', 'scenario': 'neutral', 'rag_enabled': True},
    'hana-healthy-norag': {'twin_id': 'hana', 'scenario': 'neutral', 'rag_enabled': False},
    'hana-toxic-rag': {'twin_id': 'hana', 'scenario': 'toxic', 'rag_enabled': True},
    'hana-toxic-norag': {'twin_id': 'hana', 'scenario': 'toxic', 'rag_enabled': False},
    'amara-healthy-rag': {'twin_id': 'amara', 'scenario': 'neutral', 'rag_enabled': True},
    'amara-healthy-norag': {'twin_id': 'amara', 'scenario': 'neutral', 'rag_enabled': False},
    'amara-toxic-rag': {'twin_id': 'amara', 'scenario': 'toxic', 'rag_enabled': True},
    'amara-toxic-norag': {'twin_id': 'amara', 'scenario': 'toxic', 'rag_enabled': False},
}

# ================================
# Flask Routes
# ================================

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
    start_time = datetime.now()
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
        
        # Calculate response time
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # Save assistant response
        db.save_message(twin_id, session_id, 'assistant', response, scenario, rag_enabled)
        
        # Log to monitoring system
        monitor.log_conversation(
            twin_id=twin_id,
            user_message=message,
            ai_response=response,
            scenario=scenario,
            rag_enabled=rag_enabled,
            session_id=session_id,
            response_time_ms=response_time_ms
        )
        
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

# 在这里添加新的路由
# ================================
# Fixed Configuration Routes
# ================================

@app.route('/fixed')
def fixed_index():
    """显示所有固定配置的链接"""
    return render_template('fixed_index.html', configs=FIXED_CONFIGS)

@app.route('/fixed/<config_key>')
def fixed_chat(config_key):
    """固定配置的聊天页面"""
    if config_key not in FIXED_CONFIGS:
        return "Invalid configuration", 404
    
    config = FIXED_CONFIGS[config_key]
    twin_config = twin_manager.get_twin_config(config['twin_id'])
    if not twin_config:
        return "Twin not found", 404
    
    # 创建session
    if 'session_id' not in session:
        session['session_id'] = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + os.urandom(4).hex()
    
    # 传递固定配置到模板
    return render_template('fixed_chat.html', 
                         twin_config=twin_config,
                         fixed_config=config,
                         config_key=config_key)

@app.route('/api/send_message_fixed', methods=['POST'])
def send_message_fixed():
    """为固定配置发送消息"""
    start_time = datetime.now()
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        config_key = data.get('config_key')
        
        if not message or not config_key:
            return jsonify({'error': 'Message and config_key required'}), 400
        
        if config_key not in FIXED_CONFIGS:
            return jsonify({'error': 'Invalid configuration'}), 400
        
        config = FIXED_CONFIGS[config_key]
        twin_id = config['twin_id']
        scenario = config['scenario']
        rag_enabled = config['rag_enabled']
        
        session_id = session.get('session_id', datetime.now().strftime('%Y%m%d_%H%M%S'))
        session['session_id'] = session_id
        
        # 保存用户消息
        db.save_message(twin_id, session_id, 'user', message, scenario, rag_enabled)
        
        # 生成回复
        response = twin_manager.generate_response(twin_id, message, session_id, scenario, rag_enabled)
        
        # 计算响应时间
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        # 保存助手回复
        db.save_message(twin_id, session_id, 'assistant', response, scenario, rag_enabled)
        
        # 记录到监控系统
        monitor.log_conversation(
            twin_id=twin_id,
            user_message=message,
            ai_response=response,
            scenario=scenario,
            rag_enabled=rag_enabled,
            session_id=session_id,
            response_time_ms=response_time_ms
        )
        
        return jsonify({'response': response})
    
    except Exception as e:
        logger.error(f"Error in send_message_fixed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ================================
# 现有的Admin/Analytics路由保持不变
# ================================


# ================================
# Admin/Analytics API Routes (JSON responses only)
# ================================

@app.route('/api/admin/analytics')
def admin_analytics():
    """Return analytics data as JSON"""
    try:
        analytics_data = monitor.get_analytics_dashboard_data()
        return jsonify(analytics_data)
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': 'Failed to get analytics data'}), 500

@app.route('/api/admin/conversations')
def admin_conversations():
    """Return conversation data as JSON with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        conversations = monitor.data.get('conversations', [])
        total = len(conversations)
        
        # Pagination
        start = (page - 1) * per_page
        end = start + per_page
        page_conversations = conversations[start:end]
        
        return jsonify({
            'conversations': page_conversations,
            'total': total,
            'page': page,
            'per_page': per_page,
            'total_pages': (total + per_page - 1) // per_page
        })
    except Exception as e:
        logger.error(f"Error getting conversations: {e}")
        return jsonify({'error': 'Failed to get conversation data'}), 500

@app.route('/api/admin/export/csv')
def admin_export_csv():
    """Export conversation data as CSV"""
    try:
        csv_data = monitor.export_to_csv()
        
        return Response(
            csv_data,
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename=twin_conversations_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
            }
        )
    except Exception as e:
        logger.error(f"Error exporting CSV: {e}")
        return jsonify({'error': 'Failed to export CSV'}), 500

@app.route('/api/admin/export/json')
def admin_export_json():
    """Export raw conversation data as JSON"""
    try:
        return jsonify(monitor.data)
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        return jsonify({'error': 'Failed to export JSON'}), 500

@app.route('/api/admin/sync/github', methods=['POST'])
def admin_sync_github():
    """Manually sync data to GitHub"""
    try:
        success = monitor.upload_to_github()
        
        return jsonify({
            'success': success,
            'message': 'Data successfully synced to GitHub' if success else 'Failed to sync data to GitHub'
        })
    except Exception as e:
        logger.error(f"Error syncing to GitHub: {e}")
        return jsonify({'error': 'Failed to sync to GitHub'}), 500

# ================================
# Configuration and Status Routes
# ================================

@app.route('/api/config_check')
def config_check():
    """Check environment configuration"""
    config_status = {
        'gemini_api_key': bool(os.environ.get('GEMINI_API_KEY')),
        'secret_key': bool(os.environ.get('SECRET_KEY')),
        'admin_password': bool(os.environ.get('ADMIN_PASSWORD')),
        'github_token': bool(os.environ.get('GITHUB_TOKEN')),
        'github_repo': bool(os.environ.get('GITHUB_REPO')),
        'monitoring_enabled': monitor.github_enabled,
        'total_conversations': len(monitor.data.get('conversations', []))
    }
    
    return jsonify(config_status)

@app.route('/api/test_monitoring')
def test_monitoring():
    """Test the monitoring system"""
    try:
        # Create a test conversation
        monitor.log_conversation(
            twin_id='kaiya',
            user_message='This is a test message for monitoring system',
            ai_response='This is a test response from Kaiya',
            scenario='neutral',
            rag_enabled=False,
            session_id='test_session_' + str(uuid.uuid4()),
            response_time_ms=1500
        )
        
        return jsonify({
            'success': True,
            'message': 'Test conversation logged successfully',
            'total_conversations': len(monitor.data.get('conversations', []))
        })
    except Exception as e:
        logger.error(f"Error testing monitoring: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# ================================
# Main Application Entry Point
# ================================

# 删除重复的debug_rag函数，只保留一个

# 在启动信息部分，更新为：
if __name__ == '__main__':
    # Print configuration status on startup
    print("\n" + "="*60)
    print("DIGITAL TWIN CONVERSATION MONITOR - WITH FIXED CONFIGS")
    print("="*60)
    print(f"GEMINI_API_KEY: {'✓' if os.environ.get('GEMINI_API_KEY') else '✗'}")
    print(f"SECRET_KEY: {'✓' if os.environ.get('SECRET_KEY') else '✗'}")
    print(f"ADMIN_PASSWORD: {'✓' if os.environ.get('ADMIN_PASSWORD') else '✗'}")
    print(f"GITHUB_TOKEN: {'✓' if os.environ.get('GITHUB_TOKEN') else '✗'}")
    print(f"GITHUB_REPO: {'✓' if os.environ.get('GITHUB_REPO') else '✗'}")
    print(f"GitHub Sync: {'Enabled' if monitor.github_enabled else 'Disabled'}")
    print(f"Existing conversations: {len(monitor.data.get('conversations', []))}")
    print("="*60)
    print("Standard Routes:")
    print("- Home: /")
    print("- Dynamic Chat: /chat/<twin_id>")
    print("="*60)
    print("Fixed Configuration Routes:")
    print("- Config Index: /fixed")
    print("- Lucas Healthy+RAG: /fixed/lucas-healthy-rag")
    print("- Lucas Healthy+NoRAG: /fixed/lucas-healthy-norag")
    print("- Lucas Toxic+RAG: /fixed/lucas-toxic-rag")
    print("- Lucas Toxic+NoRAG: /fixed/lucas-toxic-norag")
    print("- Hana Healthy+RAG: /fixed/hana-healthy-rag")
    print("- Hana Healthy+NoRAG: /fixed/hana-healthy-norag")
    print("- Hana Toxic+RAG: /fixed/hana-toxic-rag")
    print("- Hana Toxic+NoRAG: /fixed/hana-toxic-norag")
    print("- Amara Healthy+RAG: /fixed/amara-healthy-rag")
    print("- Amara Healthy+NoRAG: /fixed/amara-healthy-norag")
    print("- Amara Toxic+RAG: /fixed/amara-toxic-rag")
    print("- Amara Toxic+NoRAG: /fixed/amara-toxic-norag")
    print("="*60)
    print("API Endpoints:")
    print("- Dynamic Chat API: /api/send_message")
    print("- Fixed Chat API: /api/send_message_fixed")
    print("- Analytics: /api/admin/analytics")
    print("- Conversations: /api/admin/conversations")
    print("- CSV Export: /api/admin/export/csv")
    print("- JSON Export: /api/admin/export/json")
    print("- GitHub Sync: /api/admin/sync/github")
    print("- Config Check: /api/config_check")
    print("- Test Monitor: /api/test_monitoring")
    print("="*60 + "\n")
    
    port = int(os.environ.get('PORT', 5000))
    debug_mode = os.environ.get('DEBUG', 'False').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug_mode)
