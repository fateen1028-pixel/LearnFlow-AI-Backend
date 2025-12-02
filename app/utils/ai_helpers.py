import os
import json
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser

# LLM Setup
gemini_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0,
    markdown=False
)

search = DuckDuckGoSearchRun()
json_parser = JsonOutputParser()


def detect_language_from_topic(topic: str) -> str:
    """
    Lightweight heuristic to choose the most appropriate programming language
    based on the user's topic.
    """
    if not topic:
        return "auto"
    t = topic.lower()
    
    # explicit keywords -> languages
    if any(k in t for k in ["react", "javascript", "node", "vue", "angular", "js", "jsx"]):
        return "javascript"
    if any(k in t for k in ["typescript", "ts"]):
        return "typescript"
    if any(k in t for k in ["python", "django", "flask", "fastapi", "pytorch", "tensorflow", "ml", "machine learning", "data science"]):
        return "python"
    if any(k in t for k in ["html", "css", "tailwind", "bootstrap"]):
        return "html"
    if any(k in t for k in ["java", "spring", "android"]):
        return "java"
    if any(k in t for k in ["c++", "cpp"]):
        return "cpp"
    if any(k in t for k in ["c#", "c sharp"]):
        return "csharp"
    if any(k in t for k in ["go ", "golang"]):
        return "go"
    if any(k in t for k in ["rust"]):
        return "rust"
    if any(k in t for k in ["sql", "database", "postgres", "mysql"]):
        return "sql"
    if any(k in t for k in ["bash", "shell"]):
        return "bash"
    
    return "auto"


# -------------------------
# Prompt templates (updated)
# -------------------------

chat_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional AI tutor. You MUST respond with ONLY valid JSON, no other text.

CRITICAL: Your entire response must be ONLY a JSON object. Do not write any text before or after the JSON.

Required JSON structure:
{{
  "bullets": ["Main point 1", "Main point 2"],
  "steps": ["Step 1", "Step 2"],
  "bold": ["Key concept 1", "Key concept 2"],
  "markdown": "# Full Response\\n\\nYour complete answer in Markdown format with **bold**, lists, and code blocks.",
  "code_blocks": [
    {{
      "language": "javascript",
      "code": "console.log('example');"
    }}
  ]
}}

Instructions:
- Use {language} for code examples (if 'auto', choose based on {topic})
- Make markdown field complete and readable
- Include code_blocks array (can be empty if no code needed)
- CRITICAL: Return ONLY the JSON object, nothing else

Topic: {topic}
Tasks: {tasks_context}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])


roadmap_prompt = PromptTemplate.from_template("""
You are an expert study planner.

Return ONLY valid JSON. Do not add any explanation.

The JSON you MUST return must follow this structure:

{{
  "topic": "{topic}",
  "days": {days},
  "hours": {hours},
  "roadmap": [
    {{
      "day": 1,
      "tasks": [
        {{
          "parent_task": "High-level task title",
          "original_duration_minutes": 120,
          "sub_tasks": [
            {{
              "task": "Micro task",
              "duration_minutes": 30,
              "description": "One sentence explanation."
            }}
          ]
        }}
      ]
    }}
  ]
}}

Rules:
- Return ONLY valid JSON.
- Use the provided topic to generate appropriate content.
- If the topic is not programming-related, DO NOT insert programming tasks.
- Sub-task durations MUST sum to original_duration_minutes.
""")




# task_qa_prompt updated to reference topic + language
task_qa_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a professional AI tutor. You MUST respond with ONLY valid JSON, no other text.

CRITICAL: Your entire response must be ONLY a JSON object. Do not write any text before or after the JSON.

Required JSON structure:
{{
  "answer": "# Complete Answer\\n\\nFull explanation in Markdown with **bold**, code blocks, etc.",
  "key_points": ["Point 1", "Point 2"],
  "steps": ["Step 1", "Step 2"],
  "examples": ["Example explanation"],
  "code_blocks": [
    {{
      "language": "{language}",
      "code": "example code"
    }}
  ]
}}

Instructions:
- Use {language} for code (if 'auto', choose based on {topic})
- Make answer field complete with Markdown formatting
- CRITICAL: Return ONLY the JSON object, nothing else

Topic: {topic}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "Tasks: {tasks_context}\\n\\nQuestion: {question}")
])


refinement_prompt_template = PromptTemplate.from_template("""
You must refine the given roadmap.

Return ONLY the updated JSON. No text outside JSON.

Current Roadmap:
{roadmap}

Instruction:
{instruction}

Rules:
- Maintain identical structure.
- Keep sub-tasks nested.
- Sub-task durations MUST sum to original_duration_minutes.
""")


flashcards_prompt = PromptTemplate.from_template("""You MUST respond with ONLY valid JSON, no other text.

CRITICAL: Your entire response must be ONLY a JSON object.

Required JSON structure:
{{
  "flashcards": [
    {{
      "question": "Question text?",
      "answer": "Answer text",
      "category": "Category name",
      "difficulty": "easy"
    }}
  ]
}}

Generate 8-10 flashcards for: {topic}

Focus on areas where understanding is low:
{understanding}

Return ONLY the JSON object, nothing else.""")


study_guide_prompt = PromptTemplate.from_template("""You MUST respond with ONLY valid JSON, no other text.

CRITICAL: Your entire response must be ONLY a JSON object.

Required JSON structure:
{{
  "learning_objectives": ["Objective 1", "Objective 2"],
  "key_concepts": ["Concept 1", "Concept 2"],
  "practice_exercises": [
    {{
      "title": "Exercise name",
      "description": "What to do",
      "difficulty": "beginner"
    }}
  ],
  "study_schedule": [
    {{
      "week": 1,
      "topics": ["Topic A", "Topic B"],
      "exercises": ["Exercise 1"]
    }}
  ],
  "resources": [
    {{
      "type": "documentation",
      "title": "Resource title",
      "url": "https://example.com"
    }}
  ]
}}

Topic: {topic}
User understanding: {understanding}

Return ONLY the JSON object, nothing else.""")

materials_prompt = PromptTemplate.from_template("""You MUST respond with ONLY valid JSON, no other text.

CRITICAL: Your entire response must be ONLY a JSON object.

Required JSON structure:
{{
  "videos": [
    {{
      "title": "Video title",
      "url": "https://youtube.com/...",
      "channel": "Channel name",
      "duration": "10 min",
      "type": "video"
    }}
  ],
  "articles": [
    {{
      "title": "Article title",
      "url": "https://example.com",
      "source": "Website name",
      "reading_time": "5 min",
      "type": "article"
    }}
  ],
  "practice": [
    {{
      "title": "Practice resource",
      "url": "https://example.com",
      "difficulty": "Beginner",
      "type": "practice"
    }}
  ],
  "tools": [
    {{
      "name": "Tool name",
      "url": "https://example.com",
      "description": "Brief description",
      "type": "tool"
    }}
  ]
}}

Topic: {topic}

Return ONLY the JSON object, nothing else.""")

search_enhanced_prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an expert tutor with access to search results.

Return ONLY valid JSON with:

{
  "answer": "Full explanation",
  "key_points": ["Point 1", "Point 2"],
  "updated_understanding": {"concept": 60},
  "resources": [
    {
      "title": "Resource",
      "url": "https://example.com",
      "type": "video/article/tool"
    }
  ]
}

Rules:
- Use search results only inside the JSON.
- No markdown.
- No text outside JSON.

Search Results:
{search_results}

User understanding:
{understanding}
"""),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])




def extract_json_from_text(text: str) -> dict:
    """
    Robust JSON extraction from text that might contain markdown or extra content.
    """
    if not text:
        return None
    
    text = text.strip()
    
    # Remove markdown JSON code blocks
    json_block_pattern = r'```(?:json)?\s*([\s\S]*?)\s*```'
    json_block_match = re.search(json_block_pattern, text, re.IGNORECASE)
    if json_block_match:
        text = json_block_match.group(1).strip()
    
    # Remove leading/trailing backticks
    text = text.strip('`').strip()
    if text.lower().startswith('json'):
        text = text[4:].strip()
    
    # Find JSON object boundaries
    first_brace = text.find('{')
    last_brace = text.rfind('}')
    
    if first_brace == -1 or last_brace == -1:
        return None
    
    json_str = text[first_brace:last_brace + 1]
    
    # Try to parse
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        # Try to fix common issues
        # Remove trailing commas
        json_str = re.sub(r',\s*}', '}', json_str)
        json_str = re.sub(r',\s*]', ']', json_str)
        
        # Try again
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            print(f"JSON parse error: {e}")
            print(f"Attempted to parse: {json_str[:500]}...")
            return None


# -------------------------
# run_chain (inject language)
# -------------------------
def run_chain(prompt, data):
    """
    IMPROVED: Better error handling and JSON extraction
    """
    try:
        # Ensure data is a dict
        if data is None:
            data = {}
        if not isinstance(data, dict):
            try:
                data = dict(data)
            except Exception:
                data = {}

        # Inject language if missing
        if "language" not in data:
            topic = data.get("topic") or data.get("Topic") or ""
            inferred = detect_language_from_topic(topic)
            data["language"] = inferred

        # Ensure tasks_context exists
        if "tasks_context" not in data:
            data["tasks_context"] = "No specific tasks provided"

        print(f"DEBUG: Invoking chain with keys: {list(data.keys())}")
        
        # Build and invoke chain
        chain = prompt | llm
        response = chain.invoke(data)
        
        # Extract content
        if isinstance(response, dict):
            return response
        
        if hasattr(response, 'content'):
            content = str(response.content).strip()
        else:
            content = str(response).strip()
        
        print(f"DEBUG: Raw response (first 500 chars): {content[:500]}")
        
        # Extract JSON
        parsed = extract_json_from_text(content)
        
        if parsed:
            print("DEBUG: Successfully extracted and parsed JSON")
            return parsed
        else:
            print("DEBUG: Failed to extract valid JSON from response")
            print(f"DEBUG: Full response: {content}")
            return None

    except Exception as e:
        print(f"ERROR in run_chain: {e}")
        import traceback
        traceback.print_exc()
        return None
    




def create_fallback_response(topic: str, question: str) -> dict:
    """
    Create a fallback response when AI fails to return valid JSON
    """
    return {
        "markdown": f"# Response to: {question}\n\nI apologize, but I encountered an issue generating a proper response. Let me provide a basic answer:\n\nRegarding **{topic}**, this is a complex topic that requires careful study. Here are some key points to consider:\n\n- Start with the fundamentals\n- Practice regularly\n- Use official documentation\n- Build small projects to reinforce learning\n\nWould you like me to elaborate on any specific aspect?",
        "bullets": [
            "Start with fundamentals",
            "Practice regularly",
            "Use documentation",
            "Build projects"
        ],
        "steps": [
            "Learn basic concepts",
            "Practice with examples",
            "Build small projects",
            "Review and refine"
        ],
        "bold": ["fundamentals", "practice", "documentation"],
        "code_blocks": []
    }


# (the rest of your helper functions remain unchanged — I preserved them)
def create_fallback_flashcards(topic: str) -> dict:
    """
    Create fallback flashcards when AI fails
    """
    return {
        "flashcards": [
            {
                "question": f"What is {topic}?",
                "answer": f"{topic} is an important concept in programming that requires practice and understanding.",
                "category": "Fundamentals",
                "difficulty": "easy"
            },
            {
                "question": f"Why is {topic} important?",
                "answer": f"{topic} is important because it forms the foundation for more advanced concepts.",
                "category": "Concepts",
                "difficulty": "medium"
            },
            {
                "question": f"How do you practice {topic}?",
                "answer": f"Practice {topic} by working on small projects and exercises regularly.",
                "category": "Practice",
                "difficulty": "medium"
            }
        ]
    }




def create_fallback_study_guide(topic: str) -> dict:
    """
    Create fallback study guide when AI fails
    """
    return {
        "learning_objectives": [
            f"Understand the fundamentals of {topic}",
            f"Apply {topic} concepts in practice",
            f"Build projects using {topic}"
        ],
        "key_concepts": [
            "Basic principles",
            "Core functionality",
            "Best practices",
            "Common patterns"
        ],
        "practice_exercises": [
            {
                "title": f"Introduction to {topic}",
                "description": "Learn the basic concepts and syntax",
                "difficulty": "beginner"
            },
            {
                "title": f"Practical {topic}",
                "description": "Build a simple project using core concepts",
                "difficulty": "intermediate"
            }
        ],
        "study_schedule": [
            {
                "week": 1,
                "topics": ["Fundamentals", "Basic syntax"],
                "exercises": [f"Introduction to {topic}"]
            },
            {
                "week": 2,
                "topics": ["Advanced concepts", "Best practices"],
                "exercises": [f"Practical {topic}"]
            }
        ],
        "resources": [
            {
                "type": "documentation",
                "title": f"Official {topic} Documentation",
                "url": f"https://www.google.com/search?q={topic.replace(' ', '+')}+documentation"
            }
        ]
    }

def process_ai_response(raw_text):
    """
    Minimal processing to preserve Markdown formatting.
    """
    if not raw_text:
        return {"text": "", "code_blocks": []}

    code_blocks = []
    code_block_regex = r"```(\w+)?\n([\s\S]*?)```"

    if not re.search(code_block_regex, raw_text):
        return {"text": raw_text, "code_blocks": []}

    def extract_code(match):
        language = match.group(1) or "text"
        code = match.group(2).strip()
        block_id = f"CODE_BLOCK_{len(code_blocks)}"
        code_blocks.append({
            "id": block_id,
            "language": language,
            "code": code
        })
        return f"\n\n{block_id}\n\n"

    text_with_placeholders = re.sub(code_block_regex, extract_code, raw_text)

    processed_text = text_with_placeholders
    for block in code_blocks:
        processed_text = processed_text.replace(
            block['id'],
            f"```{block['language']}\n{block['code']}\n```"
        )

    return {
        "text": processed_text.strip(),
        "code_blocks": code_blocks
    }


def should_use_search(message: str, topic: str) -> bool:
    """Determine if web search should be used for this query"""
    search_triggers = [
        'current', 'recent', 'latest', 'new', 'update',
        'search', 'find', 'look up', "what's new",
        'tutorial', 'guide', 'how to', 'examples',
        'resources', 'tools', 'libraries', 'frameworks',
        'trend', 'best practice', 'popular'
    ]

    message_lower = message.lower()

    if any(trigger in message_lower for trigger in ['current', 'recent', 'latest', '2024', '2025']):
        return True

    if any(trigger in message_lower for trigger in ['tutorial', 'how to', 'guide', 'learn']):
        return True

    if any(trigger in message_lower for trigger in ['tools', 'libraries', 'frameworks', 'resources']):
        return True

    return False


def extract_resources_from_search(search_results: str, topic: str) -> list:
    """Extract structured resources from search results"""
    resources = []
    lines = search_results.split('\n')

    for line in lines:
        if 'http' in line:
            urls = re.findall(r'https?://[^\s]+', line)
            for url in urls:
                resource_type = classify_resource_type(url, line)
                title = extract_title_from_line(line)
                if title and url:
                    resources.append({
                        "url": url,
                        "type": resource_type,
                        "title": title,
                        "description": line.strip()[:150] + "..." if len(line) > 150 else line.strip()
                    })

    seen_urls = set()
    unique_resources = []

    for resource in resources:
        if resource['url'] not in seen_urls and len(unique_resources) < 8:
            seen_urls.add(resource['url'])
            unique_resources.append(resource)

    return unique_resources


def classify_resource_type(url: str, context: str) -> str:
    """Classify the type of resource based on URL and context"""
    url_lower = url.lower()
    context_lower = context.lower()

    if 'youtube.com' in url_lower or 'youtu.be' in url_lower:
        return 'video'
    elif 'github.com' in url_lower:
        return 'tool'
    elif 'docs' in url_lower or 'documentation' in context_lower:
        return 'documentation'
    elif 'course' in context_lower or 'tutorial' in context_lower:
        return 'article'
    else:
        return 'article'


def extract_title_from_line(line: str) -> str:
    """Extract a title from search result line"""
    clean_line = re.sub(r'https?://[^\s]+', '', line)
    clean_line = clean_line.strip(' -•')
    return clean_line[:60] + ('...' if len(clean_line) > 60 else '')


def extract_domain_from_url(url: str) -> str:
    """Extract domain name from URL"""
    from urllib.parse import urlparse
    domain = urlparse(url).netloc
    return domain.replace('www.', '')


def update_understanding_level(question: str, response: str, current_understanding: dict, topic: str) -> dict:
    """Update user's understanding based on the conversation"""
    understanding_update = current_understanding.copy()
    concepts_to_update = extract_concepts_from_text(question + " " + response, topic)

    for concept in concepts_to_update:
        if concept in understanding_update:
            understanding_update[concept] = min(100, understanding_update[concept] + 5)
        else:
            understanding_update[concept] = 30

    return understanding_update


def extract_concepts_from_text(text: str, topic: str) -> list:
    """Extract key concepts from text (simplified implementation)"""
    words = text.lower().split()
    potential_concepts = []

    for i, word in enumerate(text.split()):
        if (word.istitle() and len(word) > 3) or word in topic.lower():
            potential_concepts.append(word)

    return list(set(potential_concepts))[:5]


def enhanced_process_ai_response(raw_text):
    """
    Enhanced AI response processing that preserves Markdown formatting.
    Only extracts code blocks for special handling.
    """
    if not raw_text:
        return {"text": "", "code_blocks": []}

    code_blocks = []
    code_block_regex = r"```(\w+)?\n([\s\S]*?)```"

    def extract_code(match):
        language = match.group(1) or "text"
        code = match.group(2).strip()
        block_id = f"CODE_BLOCK_{len(code_blocks)}"
        code_blocks.append({
            "id": block_id,
            "language": language,
            "code": code
        })
        return f"\n\n{block_id}\n\n"

    text_with_placeholders = re.sub(code_block_regex, extract_code, raw_text)

    final_text = text_with_placeholders
    for block in code_blocks:
        final_text = final_text.replace(
            block['id'],
            f"```{block['language']}\n{block['code']}\n```"
        )

    return {
        "text": final_text.strip(),
        "code_blocks": code_blocks
    }


def enhanced_update_understanding_level(question, response, current_understanding, topic):
    """Enhanced understanding level tracking with conversation analysis"""
    understanding_update = current_understanding.copy()
    conversation_depth = analyze_conversation_depth(question, response)
    concepts = extract_concepts_with_context(question + " " + response, topic)

    for concept_data in concepts:
        concept = concept_data['concept']
        confidence = concept_data['confidence']
        complexity = concept_data['complexity']

        current_level = understanding_update.get(concept, 0)

        improvement = 0
        improvement += conversation_depth * 1.5
        improvement += confidence * 2
        improvement += complexity * 1.2

        improvement = min(improvement, 12)

        new_level = min(current_level + improvement, 100)
        understanding_update[concept] = round(new_level)

    return understanding_update


def analyze_conversation_depth(question, response):
    """Analyze the depth and quality of the conversation"""
    depth_score = 0

    question_lower = question.lower()
    response_lower = response.lower()

    if len(question.split()) > 15:
        depth_score += 2
    if len(response.split()) > 100:
        depth_score += 3

    complexity_indicators = [
        'how', 'why', 'explain', 'compare', 'difference',
        'implement', 'optimize', 'architecture', 'best practice'
    ]

    for indicator in complexity_indicators:
        if indicator in question_lower:
            depth_score += 2

    follow_up_indicators = ['following up', 'previous', 'earlier', 'based on']
    for indicator in follow_up_indicators:
        if indicator in question_lower:
            depth_score += 3

    if 'code' in question_lower or '```' in response:
        depth_score += 3

    return min(depth_score, 10)


def extract_concepts_with_context(text, main_topic):
    """Extract concepts with context and confidence scoring"""
    concepts = []

    concepts.append({
        'concept': main_topic.lower(),
        'confidence': 0.8,
        'complexity': 2
    })

    technical_terms = {
        'basic': ['variable', 'function', 'loop', 'if', 'else', 'print'],
        'intermediate': ['class', 'object', 'method', 'array', 'string', 'number'],
        'advanced': ['algorithm', 'framework', 'api', 'database', 'async', 'promise']
    }

    text_lower = text.lower()

    for term in technical_terms['basic']:
        if term in text_lower:
            concepts.append({
                'concept': term,
                'confidence': 0.6,
                'complexity': 1
            })

    for term in technical_terms['intermediate']:
        if term in text_lower:
            concepts.append({
                'concept': term,
                'confidence': 0.7,
                'complexity': 2
            })

    for term in technical_terms['advanced']:
        if term in text_lower:
            concepts.append({
                'concept': term,
                'confidence': 0.8,
                'complexity': 3
            })

    seen = set()
    unique_concepts = []
    for concept in concepts:
        if concept['concept'] not in seen:
            seen.add(concept['concept'])
            unique_concepts.append(concept)

    return unique_concepts[:6]
