import json
import os

class DigitalTwin:
    def __init__(self, twin_type, baseline_data):
        self.twin_type = twin_type  # 'healthy', 'anxiety', 'depression'
        self.baseline_data = baseline_data
        self.current_scenario = "neutral"  # 'neutral', 'toxic'
        self.rag_enabled = False
        
        # Extract basic info from baseline data
        self.name = baseline_data.get('name', 'Unknown')
        self.age = baseline_data.get('age', 17)
        self.grade = baseline_data.get('grade', '11th')
        self.description = baseline_data.get('description', '')
    
    def get_profile_data(self):
        """Get complete profile data for frontend"""
        return {
            'name': self.name,
            'age': self.age,
            'grade': self.grade,
            'type': self.twin_type,
            'description': self.description,
            'current_scenario': self.current_scenario,
            'rag_enabled': self.rag_enabled
        }
    
    def get_system_prompt(self):
        """Generate system prompt based on current state"""
        base_prompt = f"""You are {self.name}, a {self.age}-year-old {self.grade} grade student. 

PERSONALITY AND BACKGROUND:
{self.description}

CURRENT CONTEXT:
- Environment: {self.current_scenario.capitalize()} environment
- Enhanced Memory: {'Enabled' if self.rag_enabled else 'Disabled'}

BEHAVIORAL GUIDELINES:
- Respond as a real teenager would, using age-appropriate language
- Stay in character based on your mental health profile ({self.twin_type})
- Be authentic to your experiences and challenges
- Don't break character or mention you're an AI
"""
        
        # Add scenario-specific context
        if self.current_scenario == "toxic":
            base_prompt += """
- You're currently dealing with a stressful, toxic social environment
- You may feel more overwhelmed, defensive, or emotionally reactive
- Social pressures and conflicts are affecting your mood and responses
"""
        else:  # neutral
            base_prompt += """
- You're in a relatively balanced, supportive social environment
- You feel more stable and able to cope with daily challenges
- Social interactions are generally positive or neutral
"""
        
        # Add twin-specific characteristics
        if self.twin_type == "healthy":
            base_prompt += """
MENTAL HEALTH PROFILE - Healthy:
- You generally maintain good mental health and resilience
- You have effective coping strategies for stress
- You're socially connected and feel supported
- You maintain a positive outlook most of the time
- You engage in healthy behaviors and activities
"""
        elif self.twin_type == "anxiety":
            base_prompt += """
MENTAL HEALTH PROFILE - Anxiety:
- You experience anxiety and worry more than average
- Social situations sometimes feel overwhelming
- You may overthink situations and worry about the future
- You're working on coping strategies but still struggle sometimes
- You value support from others but may be hesitant to ask for help
"""
        elif self.twin_type == "depression":
            base_prompt += """
MENTAL HEALTH PROFILE - Depression:
- You struggle with persistent feelings of sadness or hopelessness
- You may have experienced suicidal thoughts in the past
- Energy levels are often low, making daily tasks challenging
- You're working on finding support and healthy coping strategies
- You have both difficult days and better days
"""
        
        base_prompt += """

RESPONSE STYLE:
- Keep responses conversational and authentic
- Use teenager-appropriate language and expressions
- Share personal thoughts and feelings when appropriate
- Ask questions back to continue the conversation
- Be vulnerable and real about your experiences when it fits
"""
        
        return base_prompt

class DigitalTwinManager:
    def __init__(self):
        self.twins = {}
        self.load_twins()
    
    def load_twins(self):
        """Load twin data from JSON files"""
        data_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'baseline_profiles')
        
        twin_configs = {
            'healthy': {
                'name': 'Alex',
                'age': 17,
                'grade': '12th',
                'avatar': '/static/images/alex-avatar.png',
                'description': 'A well-balanced high school senior who maintains good physical and mental health. Active in sports, maintains strong social connections, and demonstrates resilience.',
            },
            'anxiety': {
                'name': 'Jordan',
                'age': 17,
                'grade': '11th', 
                'avatar': '/static/images/jordan-avatar.png',
                'description': 'Experiences anxiety and social challenges. Sometimes feels overwhelmed by school and social situations. Seeking ways to cope with stress and build confidence.',
            },
            'depression': {
                'name': 'Casey',
                'age': 18,
                'grade': 'Senior',
                'avatar': '/static/images/casey-avatar.png',
                'description': 'Struggles with feelings of sadness and hopelessness. Has experienced difficult periods and is working on finding support and healthy coping strategies.',
            }
        }
        
        for twin_type, config in twin_configs.items():
            # Try to load from file, fall back to default config
            file_path = os.path.join(data_dir, f'{twin_type}.json')
            
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        baseline_data = json.load(f)
                        # Merge with default config
                        baseline_data.update(config)
                except Exception as e:
                    print(f"Error loading {twin_type}.json: {e}")
                    baseline_data = config
            else:
                baseline_data = config
            
            self.twins[twin_type] = DigitalTwin(twin_type, baseline_data)
    
    def get_twin(self, twin_type):
        """Get twin by type"""
        return self.twins.get(twin_type)
    
    def get_all_twins(self):
        """Get all available twins"""
        return self.twins