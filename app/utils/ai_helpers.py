# ai_helpers.py - COMPLETE FIXED VERSION

import os
import json
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_community.tools import DuckDuckGoSearchResults
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import JsonOutputParser
from datetime import datetime
from typing import List, Dict, Any, Optional

# LLM Setup
gemini_api_key = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=gemini_api_key,
    temperature=0,
    # markdown=False
)

search_tool = DuckDuckGoSearchResults()
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

# FIX: Removed ("system", ...) and merged instructions into the ("human", ...) message
# as Gemini does not support the "system" role.
chat_qa_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """You are a professional AI tutor. You MUST respond with ONLY valid JSON, no other text.

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

---
Question: {question}""")
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


# FIX: Removed ("system", ...) and merged instructions into the ("human", ...) message
# as Gemini does not support the "system" role.
# FIX: Enhanced prompt for better code generation
task_qa_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """You are a professional AI tutor and CODE EXPERT. You MUST respond with ONLY valid JSON, no other text.

CRITICAL: Your entire response must be ONLY a JSON object. Do not write any text before or after the JSON.

CODE GENERATION PRIORITY:
- If user asks for code, you MUST provide COMPLETE, WORKING code
- Use proper markdown code blocks with language specification
- Include explanations BEFORE and AFTER code blocks
- Make code PRACTICAL, RUNNABLE, and WELL-COMMENTED
- Focus on IMPLEMENTATION, not just theory

PAST CONVERSATIONS CONTEXT:
{memory_context}

IMPORTANT: When using past conversations:
1. Reference them naturally if relevant
2. Build upon previous explanations
3. Don't repeat the same answer word-for-word
4. Update or correct if new information is available

REQUIRED JSON structure for CODE REQUESTS:
{{
  "answer": "# Complete Answer with CODE\\n\\nExplanation of what the code does...\\n\\n```{{language}}\\n# COMPLETE CODE HERE\\ndef example():\\n    return 'Working code'\\n```\\n\\nExplanation of how it works...",
  "key_points": ["Point 1", "Point 2"],
  "steps": ["Step 1: Set up", "Step 2: Write function", "Step 3: Test"],
  "examples": ["Code explanation"],
  "code_blocks": [
    {{
      "language": "{{language}}",
      "code": "# Complete working code\\ndef main():\\n    print('Hello World')\\n\\nif __name__ == '__main__':\\n    main()"
    }}
  ]
}}

REQUIRED JSON structure for NON-CODE requests:
{{
  "answer": "# Complete Answer\\n\\nDetailed explanation...",
  "key_points": ["Point 1", "Point 2"],
  "steps": ["Step 1", "Step 2"],
  "examples": ["Example 1", "Example 2"],
  "code_blocks": []  # Empty array for non-code responses
}}

CRITICAL RULES:
1. For code requests: code_blocks MUST contain at least ONE complete code snippet
2. Use {{language}} for ALL code examples (if 'auto', choose appropriate language)
3. Make answer field COMPLETE with Markdown formatting
4. Code must be WELL-FORMATTED, INDENTED, and COMPLETE
5. Include COMMENTS in code for clarity
6. Provide REAL-WORLD examples when possible
7. Return ONLY the JSON object, NOTHING ELSE

Topic: {topic}
Task Context: {tasks_context}
Is Code Request: {is_code_request}

---
Question: {question}""")
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

# FIX: Removed ("system", ...) and merged instructions into the ("human", ...) message
# as Gemini does not support the "system" role.
search_enhanced_prompt = ChatPromptTemplate.from_messages([
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", """
You are an expert tutor with access to search results and past conversations.

PAST CONVERSATIONS CONTEXT:
{memory_context}

Return ONLY valid JSON with:

{{
  "answer": "Full explanation",
  "key_points": ["Point 1", "Point 2"],
  "updated_understanding": {{"concept": 60}},
  "resources": [
    {{
      "title": "Resource",
      "url": "https://example.com",
      "type": "video/article/tool"
    }}
  ]
}}

Rules:
- Use search results only inside the JSON.
- Reference past conversations if relevant.
- No markdown.
- No text outside JSON.

Search Results:
{search_results}

User understanding:
{understanding}

---
Question: {question}
""")
])


def extract_json_from_text(text: str) -> dict:
    """
    Safely extract and parse JSON from AI output.
    """
    if not text or not isinstance(text, str):
        return None

    text = text.strip()

    # Remove outer code fences
    text = re.sub(r'^```[\w\-]*\n?', '', text)
    text = re.sub(r'\n?```$', '', text)
    text = text.strip()

    if text.lower().startswith("json"):
        text = text[4:].strip()

    # Find JSON object boundaries
    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1:
        return None

    json_str = text[start:end + 1].strip()
    
    # QUICK FIX: Escape apostrophes and newlines
    # First, try to parse as-is
    try:
        return json.loads(json_str)
    except json.JSONDecodeError:
        pass
    
    # If that fails, try to fix common issues
    # Escape apostrophes in string values
    def escape_apostrophes(match):
        content = match.group(1)
        # Escape apostrophes
        content = content.replace("'", "\\'")
        return f'"{content}"'
    
    # Pattern to match string values (simplified)
    json_str = re.sub(r'"([^"]*)"', escape_apostrophes, json_str)
    
    # Also escape newlines
    json_str = json_str.replace('\n', '\\n').replace('\r', '\\r')
    
    # Remove trailing commas
    json_str = re.sub(r',\s*}', '}', json_str)
    json_str = re.sub(r',\s*]', ']', json_str)
    
    try:
        return json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        # Last resort: return a minimal response
        return {
            "answer": text[start:end+1][:500] + "...",
            "key_points": [],
            "steps": [],
            "examples": [],
            "code_blocks": []
        }




# -------------------------
# run_chain (inject language)
# -------------------------
def run_chain(prompt, data):
    """
    Ultra-stable chain executor for Gemini Flash.
    Ensures:
    - Always returns parsed JSON dict OR None
    - Never crashes the backend
    - Handles code fences and mixed text
    """

    try:
        # --------------------------------------
        # 0. SANITIZE / PREPARE INPUT DATA
        # --------------------------------------
        if data is None:
            data = {}

        if not isinstance(data, dict):
            try:
                data = dict(data)
            except Exception:
                data = {}

        # Inject language automatically
        if "language" not in data:
            topic = data.get("topic") or data.get("Topic") or ""
            data["language"] = detect_language_from_topic(topic)

        # Ensure tasks_context exists
        if "tasks_context" not in data:
            data["tasks_context"] = "No specific tasks provided"

        print(f"\n==============================")
        print(f"RUN CHAIN → Prompt: {prompt}")
        print(f"DATA KEYS → {list(data.keys())}")
        print("==============================\n")

        # --------------------------------------
        # 1. EXECUTE THE MODEL
        # --------------------------------------
        chain = prompt | llm
        response = chain.invoke(data)

        # Case A: LLM returned a dict already (rare)
        if isinstance(response, dict):
            print("AI returned already-parsed dict ✔")
            return response

        # Case B: LLM returned a LangChain object with `.content`
        if hasattr(response, "content"):
            raw_text = str(response.content).strip()
        else:
            raw_text = str(response).strip()

        print("RAW RESPONSE (first 400 chars):")
        print(raw_text[:400])
        print("\n----------------------------------\n")

        # --------------------------------------
        # 2. TRY EXTRACTING JSON
        # --------------------------------------
        parsed = extract_json_from_text(raw_text)

        if parsed is not None:
            print("JSON successfully extracted ✔")
            return parsed

        # --------------------------------------
        # 3. FINAL FALLBACK
        # --------------------------------------
        print("❌ JSON extraction failed. Returning None.")
        return None

    except Exception as e:
        print(f"\n❌ run_chain ERROR: {e}")
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



# Create a wrapper function for backward compatibility
def search(query: str) -> str:
    """Wrapper function for DuckDuckGo search with new API."""
    try:
        results = search_tool.invoke({"query": query})
        if isinstance(results, list):
            # Format results as string
            formatted_results = []
            for result in results:
                if isinstance(result, dict):
                    formatted_results.append(f"{result.get('title', 'No title')}: {result.get('snippet', 'No description')}")
                else:
                    formatted_results.append(str(result))
            return "\n\n".join(formatted_results)
        elif isinstance(results, str):
            return results
        else:
            return str(results)
    except Exception as e:
        print(f"Search error: {e}")
        return f"Search failed: {str(e)}"


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
            urls = re.findall(r'httpsK?://[^\s]+', line)
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
    clean_line = re.sub(r'httpsK?://[^\s]+', '', line)
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



def format_retrieved_context(similar_chats: List[Dict]) -> str:
    """
    Format retrieved similar chats into context string.
    """
    if not similar_chats:
        return ""
    
    context_lines = ["\n\n## 📚 Relevant Past Conversations:"]
    
    for i, chat in enumerate(similar_chats, 1):
        user_msg = chat.get("user_message", "").strip()
        ai_resp = chat.get("ai_response", "").strip()
        score = chat.get("score", 0)
        chat_topic = chat.get("topic", "general")
        
        if user_msg and ai_resp:
            # Truncate if too long
            if len(user_msg) > 150:
                user_msg = user_msg[:150] + "..."
            if len(ai_resp) > 200:
                ai_resp = ai_resp[:200] + "..."
            
            context_lines.append(f"\n{i}. **{chat_topic.upper()}** (Relevance: {score:.1%})")
            context_lines.append(f"   **User**: {user_msg}")
            context_lines.append(f"   **AI**: {ai_resp}")
    
    context_lines.append("\n---")
    return "\n".join(context_lines)

def create_memory_context(
    query: str, 
    user_id: str, 
    topic: str, 
    use_memory: bool = True
) -> Dict[str, Any]:
    """
    Create context from memory if available.
    Returns: dict with context_text and similar_chats
    
    Enhanced with:
    1. Personal query detection (lower threshold, no topic filter)
    2. Fallback search strategies
    3. Better debugging
    4. Special handling for short/general queries
    """
    if not use_memory:
        return {"context_text": "", "similar_chats": []}
    
    try:
        from app.utils.pinecone_service import get_pinecone_service
        
        pinecone_service = get_pinecone_service()
        if not pinecone_service:
            print("⚠️ Pinecone service not available for memory context")
            return {"context_text": "", "similar_chats": []}
        
        if not pinecone_service.available:
            print("⚠️ Pinecone service not available (available=False)")
            return {"context_text": "", "similar_chats": []}
        
        print(f"🔍 CREATE MEMORY CONTEXT - User: {user_id}, Query: '{query[:100]}...'")
        print(f"   Topic: {topic}")
        
        # -------------------------
        # 1. DETECT QUERY TYPE (ENHANCED)
        # -------------------------
        query_lower = query.lower().strip()
        
        # Personal/context queries
        personal_keywords = [
            "my name", "who am i", "remember me", "i am ", "call me",
            "do you know", "can you recall", "have we talked", "previous conversation",
            "before", "earlier", "last time"
        ]
        
        # Code/technical queries  
        code_keywords = [
            "code", "function", "class", "method", "import", "def ",
            "javascript", "python", "java", "c++", "html", "css",
            "example", "syntax", "error", "debug", "fix", "how to"
        ]
        
        # Conceptual/theory queries
        concept_keywords = [
            "what is", "explain", "define", "meaning of", "understand",
            "concept", "theory", "principle", "basics", "fundamentals"
        ]
        
        # Short/general queries (like greetings)
        short_queries = [
            "hi", "hello", "hey", "good morning", "good afternoon",
            "good evening", "what's up", "how are you"
        ]
        
        is_personal = any(keyword in query_lower for keyword in personal_keywords)
        is_code = any(keyword in query_lower for keyword in code_keywords)
        is_concept = any(keyword in query_lower for keyword in concept_keywords)
        is_short = any(keyword == query_lower or f"{keyword} " in query_lower for keyword in short_queries)
        is_very_short = len(query_lower.split()) <= 2  # 1-2 word queries
        
        print(f"   Query type - Personal: {is_personal}, Code: {is_code}, Concept: {is_concept}, Short: {is_short}")
        
        # -------------------------
        # 2. SET SEARCH PARAMETERS (ENHANCED)
        # -------------------------
        search_topic = topic
        search_threshold = 0.75
        search_limit = 5  # Increased limit for better context
        
        if is_personal:
            # Personal queries: lower threshold, no topic filter
            search_topic = None
            search_threshold = 0.4  # Very low for personal context
            search_limit = 5
            print(f"   Using PERSONAL search: topic=None, threshold={search_threshold}")
            
        elif is_short or is_very_short:
            # Short/greeting queries: very low threshold, no topic filter
            search_topic = None
            search_threshold = 0.3  # Very low for greetings
            search_limit = 3
            print(f"   Using SHORT QUERY search: topic=None, threshold={search_threshold}")
            
        elif is_code:
            # Code queries: higher threshold, topic filter
            search_threshold = 0.8  # Higher for code similarity
            print(f"   Using CODE search: threshold={search_threshold}")
            
        elif is_concept:
            # Concept queries: moderate threshold
            search_threshold = 0.7
            print(f"   Using CONCEPT search: threshold={search_threshold}")
        
        # -------------------------
        # 3. PERFORM SEARCH WITH ENHANCED STRATEGY
        # -------------------------
        similar_chats = []
        
        # Strategy 1: Try with topic filter first (unless it's personal/short query)
        if search_topic and not (is_personal or is_short):
            print(f"   Strategy 1: Searching WITH topic filter '{search_topic}'")
            similar_chats = pinecone_service.search_similar_chats(
                user_id=user_id,
                query=query,
                topic=search_topic,
                limit=search_limit,
                threshold=search_threshold
            )
            
            if similar_chats:
                print(f"   ✅ Found {len(similar_chats)} chats with topic filter")
        
        # Strategy 2: If no results or personal/short query, try without topic filter
        if not similar_chats:
            fallback_threshold = max(0.2, search_threshold - 0.2)
            print(f"   Strategy 2: Searching WITHOUT topic filter, threshold={fallback_threshold}")
            
            similar_chats = pinecone_service.search_similar_chats(
                user_id=user_id,
                query=query,
                topic=None,  # No topic filter
                limit=search_limit,
                threshold=fallback_threshold
            )
            
            if similar_chats:
                print(f"   ✅ Found {len(similar_chats)} chats without topic filter")
        
        # Strategy 3: Last resort - search for ANY context from this user
        if not similar_chats:
            print(f"   Strategy 3: Search for ANY user context (threshold=0.1)")
            
            # Get general chat history without similarity search
            history = pinecone_service.get_user_chat_history(
                user_id=user_id,
                limit=search_limit,
                topic=None
            )
            
            if history:
                # Convert history to similar_chats format
                similar_chats = []
                for i, chat in enumerate(history):
                    similar_chats.append({
                        "id": chat.get("id", f"history_{i}"),
                        "score": 0.5,  # Default score
                        "user_message": chat.get("user_message", ""),
                        "ai_response": chat.get("ai_response", ""),
                        "topic": chat.get("topic", "general"),
                        "timestamp": chat.get("timestamp", ""),
                        "metadata": chat.get("metadata", {})
                    })
                
                if similar_chats:
                    print(f"   ⚠️ Found {len(similar_chats)} from general history (no similarity)")
        
        # -------------------------
        # 4. ENHANCED DEBUGGING
        # -------------------------
        if similar_chats:
            print(f"   📊 Search results summary:")
            for i, chat in enumerate(similar_chats):
                score = chat.get("score", 0)
                chat_topic = chat.get("topic", "unknown")
                user_msg = chat.get("user_message", "")[:50]
                print(f"     {i+1}. Score: {score:.4f}, Topic: {chat_topic}")
                print(f"        User: {user_msg}...")
        else:
            print(f"   ❌ No similar chats found with any strategy")
            
            # Let's see what's in the index for this user
            try:
                history = pinecone_service.get_user_chat_history(
                    user_id=user_id,
                    limit=5,
                    topic=None
                )
                if history:
                    print(f"   ℹ️ User has {len(history)} total stored chats:")
                    for i, chat in enumerate(history):
                        chat_topic = chat.get("topic", "unknown")
                        user_msg = chat.get("user_message", "")[:50]
                        print(f"     {i+1}. Topic: {chat_topic}, User: {user_msg}...")
                else:
                    print(f"   ℹ️ User has no stored chats")
            except Exception as debug_e:
                print(f"   ℹ️ Could not get user history: {debug_e}")
        
        # -------------------------
        # 5. FORMAT CONTEXT
        # -------------------------
        context_text = format_retrieved_context(similar_chats)
        
        if context_text:
            print(f"   ✅ Memory context created ({len(similar_chats)} chats)")
        else:
            print(f"   ℹ️ No memory context available")
        
        return {
            "context_text": context_text,
            "similar_chats": similar_chats,
            "search_debug": {
                "is_personal": is_personal,
                "is_code": is_code,
                "is_concept": is_concept,
                "is_short": is_short,
                "original_threshold": 0.75,
                "used_threshold": search_threshold,
                "used_topic_filter": search_topic is not None,
                "found_chats": len(similar_chats)
            }
        }
        
    except ImportError as e:
        print(f"❌ Import error in create_memory_context: {e}")
        return {"context_text": "", "similar_chats": []}
    except Exception as e:
        print(f"❌ Error in create_memory_context: {e}")
        import traceback
        traceback.print_exc()
        return {"context_text": "", "similar_chats": []}
    


def store_conversation_memory(
    user_id: str,
    user_message: str,
    ai_response: str,
    topic: str,
    session_id: str,
    metadata: Optional[Dict] = None
) -> bool:
    """
    Store conversation in Pinecone memory.
    """
    try:
        from app.utils.pinecone_service import get_pinecone_service
        
        pinecone_service = get_pinecone_service()
        if not pinecone_service:
            return False
        
        # Prepare additional metadata
        enhanced_metadata = {
            "response_length": len(ai_response),
            "has_code": "```" in ai_response,
            "stored_at": datetime.now().isoformat()
        }
        
        if metadata:
            enhanced_metadata.update(metadata)
        
        # Store in Pinecone
        vector_id = pinecone_service.store_chat_pair(
            user_id=user_id,
            user_message=user_message,
            ai_response=ai_response,
            topic=topic,
            session_id=session_id,
            metadata=enhanced_metadata
        )
        
        return vector_id is not None
        
    except Exception as e:
        print(f"Error storing conversation memory: {e}")
        return False