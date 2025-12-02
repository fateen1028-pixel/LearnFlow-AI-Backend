# ai_service.py - COMPLETE FIXED VERSION

import json
from datetime import datetime
from langchain_core.messages import HumanMessage, AIMessage
import re
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.utils.ai_helpers import (
    run_chain, 
    process_ai_response, 
    chat_qa_prompt,
    task_qa_prompt,
    materials_prompt,
    flashcards_prompt,
    study_guide_prompt,
    search_enhanced_prompt,
    should_use_search,
    extract_resources_from_search,
    update_understanding_level,
    search,
    extract_title_from_line,
    extract_domain_from_url,
    enhanced_process_ai_response,
    detect_language_from_topic
)
from app.utils.helpers import get_db

class AIService:
    @staticmethod
    def ask_about_task(user_id, data):
        question = data.get("question")
        topic = data.get("topic", "")  # FIX: Get topic from data
        tasks_context = data.get("context")
        chat_history_raw = data.get("chat_history", [])

        chat_history = []
        for msg in chat_history_raw:
            if msg.get("role") == "user":
                chat_history.append(HumanMessage(content=msg.get("text")))
            elif msg.get("role") == "ai":
                chat_history.append(AIMessage(content=msg.get("text")))

        if not question:
            return {"status": "error", "message": "Missing question"}, 400

        try:
            tasks_str = ""
            if tasks_context:
                tasks_str = "User's current tasks:\n"
                for task in tasks_context:
                    status = "✅ Completed" if task.get('completed') else "⏳ Pending"
                    tasks_str += f"- [Group: {task.get('parent_task_title')}] {task.get('task')} ({task.get('duration_minutes', 0)}min) - {status}. Description: {task.get('description')}\n"
                tasks_str += "\n"
            else:
                tasks_str = "No tasks are currently defined."

            # FIX: Add topic and language to prompt_data
            prompt_data = {
                "tasks_context": tasks_str,
                "question": question,
                "topic": topic,
                "language": detect_language_from_topic(topic),
                "chat_history": chat_history
            }

            response = run_chain(chat_qa_prompt, prompt_data)

            if response is None:
                raise Exception("AI did not return JSON")

            # Handle different possible response structures
            if "markdown" in response:
                answer_text = response.get("markdown", "")
            elif "text" in response:
                answer_text = response.get("text", "")
            else:
                answer_text = str(response)

            # Process the answer text to extract code blocks
            processed = enhanced_process_ai_response(answer_text)

            return {
                "status": "success",
                "answer": processed["text"],
                "bullets": response.get("bullets", []),
                "steps": response.get("steps", []),
                "bold": response.get("bold", []),
                "code_blocks": processed.get("code_blocks", [])
            }

        except Exception as e:
            print(f"Error in ask-about-task: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": "Failed to get AI response"}, 500


    @staticmethod
    def fetch_current_materials_with_search(topic):
        """Fetch materials using web search"""
        try:
            # Search for different types of materials
            video_results = search.run(f"{topic} tutorial video YouTube 2024")
            article_results = search.run(f"{topic} guide article documentation 2024")
            practice_results = search.run(f"{topic} practice exercises examples code")
            tool_results = search.run(f"{topic} tools libraries frameworks")
            
            return {
                "videos": AIService.extract_videos_from_search(video_results),
                "articles": AIService.extract_articles_from_search(article_results),
                "practice": AIService.extract_practice_from_search(practice_results),
                "tools": AIService.extract_tools_from_search(tool_results)
            }
        except Exception as e:
            print(f"Error in search-based materials: {e}")
            return {
                "videos": [],
                "articles": [],
                "practice": [],
                "tools": []
            }
    

    @staticmethod
    def extract_videos_from_search(results):
        """Extract video resources from search results"""
        videos = []
        lines = results.split('\n')
        
        for line in lines:
            if 'youtube.com' in line.lower() or 'youtu.be' in line.lower():
                urls = re.findall(r'https?://[^\s]+', line)
                for url in urls:
                    if 'youtube.com' in url or 'youtu.be' in url:
                        videos.append({
                            "title": extract_title_from_line(line),
                            "url": url,
                            "channel": "YouTube",
                            "duration": "Unknown duration",
                            "type": "video"
                        })
        
        return videos[:5]

    @staticmethod
    def extract_articles_from_search(results):
        """Extract article resources from search results"""
        articles = []
        lines = results.split('\n')
        
        for line in lines:
            urls = re.findall(r'https?://[^\s]+', line)
            for url in urls:
                if not any(site in url for site in ['youtube.com', 'youtu.be', 'twitter.com', 'facebook.com']):
                    articles.append({
                        "title": extract_title_from_line(line),
                        "url": url,
                        "source": extract_domain_from_url(url),
                        "reading_time": "Unknown reading time",
                        "type": "article"
                    })
        
        return articles[:5]

    @staticmethod
    def extract_practice_from_search(results):
        """Extract practice resources from search results"""
        practice = []
        lines = results.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['exercise', 'practice', 'example', 'tutorial']):
                urls = re.findall(r'https?://[^\s]+', line)
                for url in urls:
                    practice.append({
                        "title": extract_title_from_line(line),
                        "url": url,
                        "difficulty": "Intermediate",
                        "type": "practice"
                    })
        
        return practice[:5]

    @staticmethod
    def extract_tools_from_search(results):
        """Extract tool resources from search results"""
        tools = []
        lines = results.split('\n')
        
        for line in lines:
            if any(keyword in line.lower() for keyword in ['library', 'framework', 'tool', 'package']):
                urls = re.findall(r'https?://[^\s]+', line)
                for url in urls:
                    tools.append({
                        "name": extract_title_from_line(line),
                        "url": url,
                        "description": line.strip()[:100] + "..." if len(line) > 100 else line.strip(),
                        "type": "tool"
                    })
        
        return tools[:5]

    @staticmethod
    def get_ai_generated_materials(topic):
        """Generate materials using AI with fallback to search"""
        try:
            print(f"Generating materials for topic: {topic}")
            
            # FIX: Add language and tasks_context to the prompt data
            prompt_data = {
                "topic": topic,
                "language": detect_language_from_topic(topic),
                "tasks_context": "No specific tasks provided"
            }
            
            # Try LLM first
            materials = run_chain(materials_prompt, prompt_data)

            # Case A: LLM returned nothing
            if not materials:
                print("LLM returned nothing, using search fallback")
                return AIService.fetch_current_materials_with_search(topic)

            # Case B: LLM returned empty lists
            if (
                len(materials.get("videos", [])) == 0 or
                len(materials.get("articles", [])) == 0 or
                len(materials.get("practice", [])) == 0 or
                len(materials.get("tools", [])) == 0
            ):
                print("LLM returned incomplete data, using hybrid approach")
                fallback = AIService.fetch_current_materials_with_search(topic)

                return {
                    "videos": materials.get("videos") or fallback["videos"],
                    "articles": materials.get("articles") or fallback["articles"],
                    "practice": materials.get("practice") or fallback["practice"],
                    "tools": materials.get("tools") or fallback["tools"],
                }

            # Case C: everything is fine
            print(f"LLM returned complete materials")
            return materials
            
        except Exception as e:
            print(f"Error in get_ai_generated_materials: {e}")
            import traceback
            traceback.print_exc()
            return AIService.fetch_current_materials_with_search(topic)


    @staticmethod
    def generate_flashcards(data):
        """Generate flashcards with proper error handling"""
        topic = data.get("topic")
        user_understanding = data.get("userUnderstanding", {})
        
        try:
            print(f"Generating flashcards for topic: {topic}")
            
            # FIX: Add language to prompt data
            result = run_chain(flashcards_prompt, {
                "topic": topic,
                "understanding": json.dumps(user_understanding),
                "language": detect_language_from_topic(topic)
            })
            
            print(f"Raw flashcard result: {result}")
            
            # Validate the structure
            if result and isinstance(result, dict) and "flashcards" in result:
                flashcards = result["flashcards"]
                validated_flashcards = []
                for card in flashcards:
                    if isinstance(card, dict):
                        validated_card = {
                            "question": card.get("question", "No question provided"),
                            "answer": card.get("answer", "No answer provided"),
                            "category": card.get("category", "General"),
                            "difficulty": card.get("difficulty", "medium")
                        }
                        validated_flashcards.append(validated_card)
                
                print(f"Validated flashcards: {len(validated_flashcards)}")
                
                return {
                    "status": "success", 
                    "flashcards": validated_flashcards
                }
            else:
                print("Using fallback flashcards")
                fallback_flashcards = AIService.create_fallback_flashcards(topic)
                return {
                    "status": "success",
                    "flashcards": fallback_flashcards
                }
                
        except Exception as e:
            print(f"Error generating flashcards: {e}")
            import traceback
            traceback.print_exc()
            fallback_flashcards = AIService.create_fallback_flashcards(topic)
            return {
                "status": "success",
                "flashcards": fallback_flashcards
            }


    @staticmethod
    def create_fallback_flashcards(topic):
        """Create fallback flashcards when AI generation fails"""
        topic_lower = str(topic).lower()
        
        if "python" in topic_lower:
            return [
                {
                    "question": "What is the difference between a list and a tuple in Python?",
                    "answer": "Lists are mutable (can be modified) while tuples are immutable (cannot be modified). Lists use square brackets [] and tuples use parentheses ().",
                    "category": "Data Structures",
                    "difficulty": "easy"
                },
                {
                    "question": "How do you define a function in Python?",
                    "answer": "Using the 'def' keyword followed by the function name and parentheses containing parameters. Example: def my_function(param1, param2):",
                    "category": "Functions",
                    "difficulty": "easy"
                },
                {
                    "question": "What are Python decorators and how are they used?",
                    "answer": "Decorators are functions that modify the behavior of other functions. They are denoted by the @ symbol and are placed above function definitions.",
                    "category": "Advanced Concepts",
                    "difficulty": "medium"
                }
            ]
        else:
            return [
                {
                    "question": f"What are the key concepts of {topic}?",
                    "answer": f"This flashcard covers fundamental concepts in {topic}. Study the main principles and applications.",
                    "category": "Fundamentals",
                    "difficulty": "easy"
                },
                {
                    "question": f"How is {topic} applied in real-world scenarios?",
                    "answer": f"{topic} has various practical applications across different industries and use cases.",
                    "category": "Applications", 
                    "difficulty": "medium"
                }
            ]

    @staticmethod
    def generate_study_guide(data):
        """Generate study guide with proper error handling"""
        topic = data.get("topic")
        user_understanding = data.get("userUnderstanding", {})
        
        try:
            print(f"Generating study guide for topic: {topic}")
            
            # FIX: Add language to prompt data
            result = run_chain(study_guide_prompt, {
                "topic": topic,
                "understanding": json.dumps(user_understanding),
                "language": detect_language_from_topic(topic)
            })
            
            print(f"Raw study guide result: {result}")
            
            if result and isinstance(result, dict):
                return {
                    "status": "success", 
                    "study_guide": result
                }
            else:
                print("Using fallback study guide")
                fallback_guide = AIService.create_fallback_study_guide(topic)
                return {
                    "status": "success",
                    "study_guide": fallback_guide
                }
                
        except Exception as e:
            print(f"Error generating study guide: {e}")
            import traceback
            traceback.print_exc()
            fallback_guide = AIService.create_fallback_study_guide(topic)
            return {
                "status": "success",
                "study_guide": fallback_guide
            }
    

    @staticmethod
    def create_fallback_study_guide(topic):
        """Create fallback study guide when AI generation fails"""
        return {
            "learning_objectives": [
                f"Master fundamental concepts of {topic}",
                f"Develop practical skills in {topic} application",
                f"Understand advanced {topic} concepts and patterns",
                f"Build real-world projects using {topic}"
            ],
            "key_concepts": [
                "Basic syntax and structure",
                "Data types and variables",
                "Control flow and functions",
                "Object-oriented programming",
                "Error handling and debugging"
            ],
            "practice_exercises": [
                {
                    "title": "Basic Syntax Practice",
                    "description": "Write a simple program to demonstrate basic syntax",
                    "difficulty": "beginner"
                },
                {
                    "title": "Data Structures Implementation", 
                    "description": "Create and manipulate common data structures",
                    "difficulty": "intermediate"
                },
                {
                    "title": "Project Building",
                    "description": "Build a complete application using core concepts",
                    "difficulty": "advanced"
                }
            ],
            "resources": [
                {
                    "type": "documentation",
                    "title": f"Official {topic} Documentation",
                    "url": "#"
                },
                {
                    "type": "tutorial",
                    "title": f"Complete {topic} Tutorial",
                    "url": "#"
                }
            ],
            "study_schedule": [
                {
                    "week": 1,
                    "topics": ["Introduction", "Basic Syntax", "Variables"],
                    "exercises": ["Hello World", "Basic Calculator"]
                },
                {
                    "week": 2, 
                    "topics": ["Functions", "Control Flow", "Data Structures"],
                    "exercises": ["Function Practice", "Data Manipulation"]
                }
            ]
        }

    @staticmethod
    def handle_ai_chat(user_id, data):
        """Main handler for AI chat"""
        message = data.get("message")
        topic = data.get("topic")
        tasks = data.get("tasks", [])
        chat_history = data.get("chatHistory", [])
        user_understanding = data.get("userUnderstanding", {})
        
        print(f"DEBUG: handle_ai_chat called with topic: {topic}")
        
        # Check if web search is needed
        needs_search = should_use_search(message, topic)
        
        if needs_search:
            return AIService.handle_search_enhanced_chat(data, needs_search)
        else:
            return AIService.handle_regular_chat(data)

    @staticmethod
    def handle_search_enhanced_chat(data, search_type):
        """Handle chat with web search integration"""
        message = data.get("message")
        topic = data.get("topic")
        chat_history = data.get("chatHistory", [])
        user_understanding = data.get("userUnderstanding", {})
        
        try:
            # Perform web search based on query type
            if 'trend' in message.lower() or 'current' in message.lower():
                search_query = f"{topic} current trends developments 2024"
            elif 'tool' in message.lower():
                search_query = f"{topic} tools libraries frameworks 2024"
            else:
                search_query = f"{topic} {message} tutorial guide examples 2024"
            
            print(f"DEBUG: Performing search with query: {search_query}")
            search_results = search.run(search_query)
            
            # Convert chat history
            history_messages = []
            for msg in chat_history:
                if msg.get('sender') == 'user':
                    history_messages.append(HumanMessage(content=msg.get('text', '')))
                else:
                    history_messages.append(AIMessage(content=msg.get('text', '')))
            
            # FIX: Add language to prompt_data
            prompt_data = {
                "topic": topic,
                "search_results": search_results[:2000],
                "question": message,
                "understanding": json.dumps(user_understanding),
                "language": detect_language_from_topic(topic),
                "chat_history": history_messages
            }
            
            response = run_chain(search_enhanced_prompt, prompt_data)
            
            if response is None:
                raise Exception("AI did not return valid JSON")
            
            # Process the response
            response_text = response.get("answer", "")
            if not response_text:
                response_text = str(response)
            
            processed = enhanced_process_ai_response(response_text) or {
                "text": response_text,
                "code_blocks": []
            }
            
            # Extract resources from search results
            resources = extract_resources_from_search(search_results, topic) or []
            
            # Update understanding
            understanding_update = AIService.calculate_understanding_update(
                message, processed["text"], user_understanding, topic
            )
            
            return {
                "status": "success",
                "response": {
                    "text": processed["text"],
                    "type": "search_enhanced",
                    "resources": resources,
                    "understandingUpdate": understanding_update,
                    "search_used": True,
                    "code_blocks": processed.get("code_blocks", [])
                }
            }
            
        except Exception as e:
            print(f"Search-enhanced chat error: {e}")
            import traceback
            traceback.print_exc()
            return {"status": "error", "message": "Failed to process search-enhanced request"}

    @staticmethod
    def handle_regular_chat(data):
        """Enhanced regular chat implementation with code generation support"""
        message = data.get("message")
        topic = data.get("topic")  # FIX: Get topic from data
        tasks = data.get("tasks", [])
        chat_history = data.get("chatHistory", [])
        user_understanding = data.get("userUnderstanding", {})
        
        print(f"DEBUG: handle_regular_chat - Topic: {topic}, Message: {message[:50]}...")
        
        # Create context from tasks
        tasks_context = ""
        if tasks and len(tasks) > 0:
            tasks_context = "Current learning tasks:\n"
            for task in tasks:
                status = "✅" if task.get('completed') else "⏳"
                task_title = task.get('task') or task.get('parent_task') or 'Unknown task'
                tasks_context += f"- {status} {task_title}\n"
        
        # Convert chat history
        history_messages = []
        for msg in chat_history:
            if msg.get('sender') == 'user':
                history_messages.append(HumanMessage(content=msg.get('text', '')))
            else:
                history_messages.append(AIMessage(content=msg.get('text', '')))
        
        try:
            # FIX: Prepare the prompt data with ALL required fields including topic
            prompt_data = {
                "tasks_context": tasks_context,
                "question": message,
                "topic": topic,  # CRITICAL FIX: Add topic
                "language": detect_language_from_topic(topic),  # CRITICAL FIX: Add language
                "chat_history": history_messages
            }
            
            print(f"DEBUG: Invoking chain with keys: {list(prompt_data.keys())}")
            
            response = run_chain(task_qa_prompt, prompt_data)
            
            if response is None:
                print("DEBUG: AI returned None response")
                raise Exception("AI did not return valid JSON")
            
            print(f"DEBUG: AI response keys: {response.keys() if isinstance(response, dict) else 'not a dict'}")
            
            # Extract the answer from the structured response
            response_text = response.get("answer", "")
            if not response_text:
                print("DEBUG: No 'answer' field, trying other fields")
                response_text = response.get("markdown", "") or response.get("text", "") or str(response)
            
            # Get code blocks from response
            code_blocks = response.get("code_blocks", [])
            
            # If there are markdown code blocks in text, extract them
            processed = enhanced_process_ai_response(response_text)
            if processed and processed.get("code_blocks"):
                code_blocks = processed.get("code_blocks")
            
            # Calculate understanding update
            understanding_update = AIService.calculate_understanding_update(
                message, response_text, user_understanding, topic
            )
            
            print(f"DEBUG: Returning response with {len(code_blocks)} code blocks")
            
            return {
                "status": "success",
                "response": {
                    "text": response_text,
                    "type": "general",
                    "understandingUpdate": understanding_update,
                    "search_used": False,
                    "code_blocks": code_blocks
                }
            }
            
        except Exception as e:
            print(f"Regular chat error: {e}")
            import traceback
            traceback.print_exc()
            return {
                "status": "error", 
                "message": f"Failed to process request: {str(e)}"
            }

    @staticmethod
    def calculate_understanding_update(user_message, ai_response, current_understanding, topic):
        """Enhanced understanding level calculation"""
        understanding_update = current_understanding.copy()
        
        # Analyze conversation complexity
        complexity_score = AIService.analyze_conversation_complexity(user_message, ai_response)
        
        # Extract key concepts
        concepts = AIService.extract_key_concepts(user_message + " " + ai_response, topic)
        
        # Update understanding for each concept
        for concept in concepts:
            current_level = understanding_update.get(concept, 0)
            
            # Calculate improvement based on complexity and engagement
            improvement = complexity_score * 2
            improvement = min(improvement, 10)
            
            new_level = min(current_level + improvement, 100)
            understanding_update[concept] = new_level
        
        # Ensure main topic is always included
        if topic not in understanding_update:
            understanding_update[topic] = 10
        
        return understanding_update

    @staticmethod
    def analyze_conversation_complexity(user_message, ai_response):
        """Analyze the complexity of the conversation"""
        score = 0
        
        user_words = len(user_message.split())
        ai_words = len(ai_response.split())
        
        if user_words > 20:
            score += 2
        if ai_words > 100:
            score += 3
        
        complex_indicators = [
            'how does', 'why does', 'explain', 'compare', 'difference between',
            'implement', 'optimize', 'architecture', 'best practice', 'advanced'
        ]
        
        for indicator in complex_indicators:
            if indicator in user_message.lower():
                score += 2
        
        if '```' in ai_response:
            score += 3
        
        return min(score, 5)

    @staticmethod
    def extract_key_concepts(text, main_topic):
        """Extract key concepts from text"""
        concepts = set()
        
        # Always include main topic
        concepts.add(str(main_topic).lower())
        
        # Technical terms
        technical_terms = [
            'function', 'variable', 'class', 'object', 'method',
            'algorithm', 'framework', 'library', 'api', 'database',
            'syntax', 'compiler', 'debug', 'deploy', 'optimize'
        ]
        
        words = text.lower().split()
        for word in words:
            if len(word) > 4 and word in technical_terms:
                concepts.add(word)
        
        return list(concepts)[:10]

    @staticmethod
    def validate_resources(resources):
        """Validate and clean resource objects"""
        validated = []
        for resource in resources:
            if isinstance(resource, dict) and resource.get('url'):
                clean_resource = {
                    'url': resource['url'],
                    'title': resource.get('title', resource.get('name', 'Unknown')),
                    'type': resource.get('type', 'unknown')
                }
                
                optional_fields = ['channel', 'source', 'duration', 'description', 'difficulty']
                for field in optional_fields:
                    if field in resource:
                        clean_resource[field] = resource[field]
                
                validated.append(clean_resource)
        
        return validated[:8]