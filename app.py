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
import uuid
from io import StringIO
import base64
import requests

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

# Use Gemini 2.5 Flash model
chat_model = genai.GenerativeModel('gemini-2.5-flash')

# ================================
# Data Monitoring System
# ================================

class ConversationMonitor:
    def __init__(self):
        self.data_file = 'conversation_data.json'
        self.github_enabled = self.setup_github()
        self.data = self.load_data()
        self.operation_count = 0
        self.save_frequency = 1

    def setup_github(self):
        self.github_token = os.environ.get("GITHUB_TOKEN")
        self.github_repo = os.environ.get("GITHUB_REPO")
        self.github_branch = os.environ.get("GITHUB_BRANCH", "main")
        enabled = bool(self.github_token and self.github_repo)
        if enabled:
            print("GitHub sync enabled for conversation data")
        else:
            print("Using local file storage only")
        return enabled

    def load_data(self):
        if self.github_enabled:
            self.download_from_github()

        if os.path.exists(self.data_file):
            try:
                with open(self.data_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if 'conversations' not in data:
                        data['conversations'] = []
                    if 'twins_chatted' in data and isinstance(data['twins_chatted'], list):
                        data['twins_chatted'] = set(data['twins_chatted'])
                    print(f"Loaded {len(data.get('conversations', []))} conversations")
                    return data
            except Exception as e:
                print(f"Error loading data: {e}")

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
        if session_id is None:
            session_id = self.create_session_id()

        conversation = {
            'id': len(self.data['conversations']) + 1,
            'session_id': session_id,
            'twin_id': twin_id,
            'twin_name': self.get_twin_name(twin_id),
            'user_message': user_message[:3000],
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
        conversations = self.data['conversations']
        if not conversations:
            return {
                'twin_stats': {}, 'scenario_stats': {}, 'hourly_distribution': {},
                'daily_distribution': {}, 'total_conversations': 0, 'unique_twins': 0,
                'unique_sessions': 0, 'recent_conversations': 0, 'avg_response_time': 0,
                'rag_usage_stats': {}
            }

        twin_stats = {}
        scenario_stats = {}
        hourly_distribution = {}
        daily_distribution = {}
        rag_usage = {'enabled': 0, 'disabled': 0}
        response_times = []
        unique_sessions = set()

        for conv in conversations:
            twin = conv['twin_name']
            if twin not in twin_stats:
                twin_stats[twin] = {
                    'total_conversations': 0, 'avg_response_time': 0,
                    'total_response_time': 0, 'scenarios_used': set()
                }

            twin_stats[twin]['total_conversations'] += 1
            twin_stats[twin]['total_response_time'] += conv.get('response_time_ms', 0)
            twin_stats[twin]['scenarios_used'].add(conv.get('scenario', 'neutral'))

            scenario = conv.get('scenario', 'neutral')
            scenario_stats[scenario] = scenario_stats.get(scenario, 0) + 1

            hour = conv.get('hour', 0)
            hourly_distribution[hour] = hourly_distribution.get(hour, 0) + 1
            
            day = conv.get('day_of_week', 'Unknown')
            daily_distribution[day] = daily_distribution.get(day, 0) + 1

            if conv.get('rag_enabled', False):
                rag_usage['enabled'] += 1
            else:
                rag_usage['disabled'] += 1

            if conv.get('response_time_ms', 0) > 0:
                response_times.append(conv['response_time_ms'])

            unique_sessions.add(conv.get('session_id', ''))

        for twin, stats in twin_stats.items():
            if stats['total_conversations'] > 0:
                stats['avg_response_time'] = stats['total_response_time'] / stats['total_conversations']
                stats['scenarios_used'] = list(stats['scenarios_used'])

        recent_conversations = len([c for c in conversations 
                                  if (datetime.now() - datetime.fromisoformat(c['timestamp'])).days < 1])

        return {
            'twin_stats': twin_stats, 'scenario_stats': scenario_stats,
            'hourly_distribution': hourly_distribution, 'daily_distribution': daily_distribution,
            'total_conversations': len(conversations),
            'unique_twins': len(set(conv['twin_id'] for conv in conversations)),
            'unique_sessions': len(unique_sessions), 'recent_conversations': recent_conversations,
            'avg_response_time': sum(response_times) / len(response_times) if response_times else 0,
            'rag_usage_stats': rag_usage,
            'most_active_twin': max(twin_stats.items(), key=lambda x: x[1]['total_conversations'])[0] if twin_stats else 'None'
        }

    def export_to_csv(self):
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
        if data is None:
            data = self.data
        
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
        self.operation_count += 1
        self.save_data_to_file()
        
        if self.github_enabled and (self.operation_count % self.save_frequency == 0 or force_upload):
            self.upload_to_github()

    def download_from_github(self):
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

            get_response = requests.get(url, headers=headers, timeout=10)
            sha = None
            if get_response.status_code == 200:
                sha = get_response.json()['sha']

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

monitor = ConversationMonitor()

# ================================
# Digital Twin Database
# ================================

class DigitalTwinDatabase:
    def __init__(self, db_path='digital_twins.db'):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
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
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO chat_history (twin_id, session_id, sender, message, scenario, rag_enabled)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (twin_id, session_id, sender, message, scenario, rag_enabled))
        
        conn.commit()
        conn.close()
    
    def get_chat_history(self, twin_id: str, session_id: str, limit: int = 20) -> List[Dict]:
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
        
        return [
            {
                'sender': row[0], 'message': row[1], 'timestamp': row[2],
                'scenario': row[3], 'rag_enabled': row[4]
            }
            for row in reversed(rows)
        ]
    
    def update_session(self, twin_id: str, session_id: str, scenario: str = None, rag_enabled: bool = None):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM twin_sessions WHERE twin_id = ? AND session_id = ?', (twin_id, session_id))
        if cursor.fetchone():
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
            cursor.execute('''
                INSERT INTO twin_sessions (twin_id, session_id, current_scenario, rag_enabled)
                VALUES (?, ?, ?, ?)
            ''', (twin_id, session_id, scenario or 'neutral', rag_enabled or False))
        
        conn.commit()
        conn.close()

# ================================
# Enriched Memory System
# ================================

class EnrichedMemorySystem:
    def __init__(self, memory_root: str = "enrich memory"):
        self.memory_path = Path(memory_root)
        self.neutral_data = self.load_enriched_data('neutral_enriched_pro.json')
        self.toxic_data = self.load_enriched_data('toxic_enriched_pro.json')
        
        # Mapping from test_ids to twin names
        self.test_id_to_twin = {
            'test_1': 'kaiya', 'test_2': 'ethan', 'test_3': 'lucas', 'test_4': 'maya',
            'test_5': 'jaden', 'test_6': 'nia', 'test_7': 'hana', 'test_8': 'mateo',
            'test_9': 'diego', 'test_10': 'emily', 'test_11': 'amara', 'test_12': 'tavian'
        }
        
        logger.info("Enriched memory system initialized")
    
    def load_enriched_data(self, filename: str) -> Dict:
        filepath = self.memory_path / filename
        if not filepath.exists():
            logger.warning(f"Enriched data file not found: {filepath}")
            return {}
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                logger.info(f"Loaded enriched data from {filename}")
                return data
        except Exception as e:
            logger.error(f"Error loading enriched data from {filename}: {e}")
            return {}
    
    def get_enriched_profile(self, twin_id: str, scenario: str) -> str:
        # Get the appropriate data based on scenario
        data_source = self.neutral_data if scenario == 'neutral' else self.toxic_data
        
        if not data_source or 'enrichments' not in data_source:
            return ""
        
        # Find the twin's data by matching test_id
        twin_test_id = None
        for test_id, name in self.test_id_to_twin.items():
            if name == twin_id.lower():
                twin_test_id = test_id
                break
        
        if not twin_test_id:
            logger.warning(f"Could not find test_id for twin: {twin_id}")
            return ""
        
        # Find the enrichment data for this twin
        enrichment = None
        for enrich in data_source['enrichments']:
            if enrich.get('student_id') == twin_test_id:
                enrichment = enrich
                break
        
        if not enrichment:
            logger.warning(f"No enrichment data found for {twin_id} ({twin_test_id}) in {scenario} scenario")
            return ""
        
        # Build the enriched profile text
        profile_parts = []
        
        # Add domain narratives
        if 'enriched_domains' in enrichment:
            profile_parts.append("Based on your experiences and background, here's how you might respond in different situations:\n")
            
            for domain in enrichment['enriched_domains']:
                domain_name = domain.get('domain', 'Unknown')
                narrative = domain.get('overall_domain_narrative', '')
                
                if narrative and 'error' not in narrative.lower():
                    profile_parts.append(f"In terms of {domain_name}: {narrative}\n")
        
        # Add conversation examples
        if 'daily_conversations' in enrichment and enrichment['daily_conversations']:
            env_type = 'neutral' if scenario == 'neutral' else 'toxic'
            profile_parts.append(f"\nIn a {env_type} environment, you might have conversations like these:\n")
            
            for conv in enrichment['daily_conversations']:
                setting = conv.get('setting', 'Unknown setting')
                profile_parts.append(f"\nSetting: {setting}")
                
                if 'dialogue' in conv:
                    for exchange in conv['dialogue']:
                        speaker = exchange.get('speaker', 'Unknown')
                        text = exchange.get('text', '')
                        profile_parts.append(f"{speaker}: {text}")
                
                profile_parts.append("")  # Empty line between conversations
        
        return '\n'.join(profile_parts)

# ================================
# Twin Manager
# ================================

class TwinManager:
    def __init__(self, db: DigitalTwinDatabase, memory_system: EnrichedMemorySystem):
        self.db = db
        self.memory_system = memory_system
        self.twin_profiles = self.load_twin_profiles()
        self.shared_prompt = self.load_shared_prompt()

    def _postprocess_client_text(self, text: str) -> str:
        if not text:
            return text
        text = re.sub(r"\\([\[\]\*\_\~`])", r"\1", text)
        text = text.replace("\r\n", "\n")
        return text

    def _enforce_emotion_breaks(self, text: str) -> str:
        if not text:
            return text
    
        pattern = re.compile(r"(Emotion(?:al)?\s*tag\s*:\s*[^\n]+)", re.IGNORECASE)
    
        last = None
        for m in pattern.finditer(text):
            last = m
        if not last:
            return text
    
        start = last.start(1)
        before, tagline = text[:start], text[start:]
        before = before.rstrip("\n")
        return before + "\n\n" + tagline
    
    def load_twin_profiles(self) -> Dict[str, str]:
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
        try:
            with open('shared_prompt.txt', 'r', encoding='utf-8') as f:
                content = f.read().strip()
                logger.info("Loaded shared prompt")
                return content
        except Exception as e:
            logger.error(f"Error loading shared prompt: {e}")
            return ""
    
    def get_twin_config(self, twin_id: str) -> Dict:
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
        
        # Add enriched profile if RAG is enabled
        if rag_enabled:
            enriched_profile = self.memory_system.get_enriched_profile(twin_id, scenario)
            if enriched_profile:
                prompt_parts.append(f"\n{enriched_profile}")
                logger.info(f"Added enriched profile for {twin_id} in {scenario} scenario")
        
        # Add scenario context
        scenario_context = self.get_scenario_context(scenario, twin_config['mental_health_type'])
        if scenario_context:
            prompt_parts.append(f"\nCurrent environment context: {scenario_context}")
        
        # Add recent chat history
        if chat_history:
            history_text = "\nRecent conversation:\n"
            for msg in chat_history[-5:]:
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
memory_system = EnrichedMemorySystem()
twin_manager = TwinManager(db, memory_system)

# Fixed configurations
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
        
        db.save_message(twin_id, session_id, 'user', message, scenario, rag_enabled)
        
        response = twin_manager.generate_response(twin_id, message, session_id, scenario, rag_enabled)
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        db.save_message(twin_id, session_id, 'assistant', response, scenario, rag_enabled)
        
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

# ================================
# Fixed Configuration Routes
# ================================

@app.route('/fixed')
def fixed_index():
    return render_template('fixed_index.html', configs=FIXED_CONFIGS)

@app.route('/fixed/<config_key>')
def fixed_chat(config_key):
    if config_key not in FIXED_CONFIGS:
        return "Invalid configuration", 404
    
    config = FIXED_CONFIGS[config_key]
    twin_config = twin_manager.get_twin_config(config['twin_id'])
    if not twin_config:
        return "Twin not found", 404
    
    # Initialize session with start time for fixed chats
    if 'session_id' not in session or session.get('config_key') != config_key:
        session['session_id'] = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + os.urandom(4).hex()
        session['fixed_chat_start_time'] = datetime.now().isoformat()
        session['config_key'] = config_key
    
    return render_template('fixed_chat.html', 
                         twin_config=twin_config,
                         fixed_config=config,
                         config_key=config_key)

@app.route('/api/send_message_fixed', methods=['POST'])
def send_message_fixed():
    start_time = datetime.now()
    try:
        # Check 10-minute time limit for fixed configurations
        if 'fixed_chat_start_time' in session:
            start_time_str = session['fixed_chat_start_time']
            start_time_dt = datetime.fromisoformat(start_time_str)
            elapsed_seconds = (datetime.now() - start_time_dt).total_seconds()
            
            if elapsed_seconds > 600:  # 10 minutes = 600 seconds
                return jsonify({
                    'error': 'session_expired',
                    'message': 'Your 10-minute chat session has expired. Please refresh the page to start a new session.'
                }), 403
        
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
        
        db.save_message(twin_id, session_id, 'user', message, scenario, rag_enabled)
        
        response = twin_manager.generate_response(twin_id, message, session_id, scenario, rag_enabled)
        
        response_time_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        
        db.save_message(twin_id, session_id, 'assistant', response, scenario, rag_enabled)
        
        monitor.log_conversation(
            twin_id=twin_id,
            user_message=message,
            ai_response=response,
            scenario=scenario,
            rag_enabled=rag_enabled,
            session_id=session_id,
            response_time_ms=response_time_ms
        )
        
        # Calculate remaining time
        remaining_seconds = 600 - elapsed_seconds if 'fixed_chat_start_time' in session else 600
        
        return jsonify({
            'response': response,
            'remaining_seconds': max(0, int(remaining_seconds))
        })
    
    except Exception as e:
        logger.error(f"Error in send_message_fixed: {e}")
        return jsonify({'error': 'Internal server error'}), 500

# ================================
# Admin/Analytics API Routes
# ================================

@app.route('/api/admin/analytics')
def admin_analytics():
    try:
        analytics_data = monitor.get_analytics_dashboard_data()
        return jsonify(analytics_data)
    except Exception as e:
        logger.error(f"Error getting analytics: {e}")
        return jsonify({'error': 'Failed to get analytics data'}), 500

@app.route('/api/admin/conversations')
def admin_conversations():
    try:
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        
        conversations = monitor.data.get('conversations', [])
        total = len(conversations)
        
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
    try:
        return jsonify(monitor.data)
    except Exception as e:
        logger.error(f"Error exporting JSON: {e}")
        return jsonify({'error': 'Failed to export JSON'}), 500

@app.route('/api/admin/sync/github', methods=['POST'])
def admin_sync_github():
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
    try:
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

if __name__ == '__main__':
    print("\n" + "="*60)
    print("DIGITAL TWIN CONVERSATION SYSTEM - JSON-BASED RAG")
    print("="*60)
    print(f"GEMINI_API_KEY: {'✓' if os.environ.get('GEMINI_API_KEY') else '✗'}")
    print(f"SECRET_KEY: {'✓' if os.environ.get('SECRET_KEY') else '✗'}")
    print(f"ADMIN_PASSWORD: {'✓' if os.environ.get('ADMIN_PASSWORD') else '✗'}")
    print(f"GITHUB_TOKEN: {'✓' if os.environ.get('GITHUB_TOKEN') else '✗'}")
    print(f"GITHUB_REPO: {'✓' if os.environ.get('GITHUB_REPO') else '✗'}")
    print(f"GitHub Sync: {'Enabled' if monitor.github_enabled else 'Disabled'}")
    print(f"Existing conversations: {len(monitor.data.get('conversations', []))}")
    print("="*60)
    print("Model: gemini-2.5-flash")
    print("RAG Method: JSON-based enriched profiles")
    print("="*60)
    print("Standard Routes:")
    print("- Home: /")
    print("- Dynamic Chat: /chat/<twin_id>")
    print("="*60)
    print("Fixed Configuration Routes (10-minute time limit):")
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
