from flask import Blueprint, request, jsonify
from app.middleware.auth import token_required
from app.services.ai_service import AIService

ai_bp = Blueprint('ai', __name__)

@ai_bp.route("/ask-about-task", methods=["POST"])
@token_required
def ask_about_task():
    data = request.json
    user_id = request.user_id
    
    result = AIService.ask_about_task(user_id, data)
    return jsonify(result)

@ai_bp.route("/ai-env/materials", methods=["POST"])
@token_required
def get_learning_materials():
    data = request.json
    topic = data.get("topic")
    tasks = data.get("tasks", [])  # Add this line to get tasks
    
    try:
        result = AIService.get_learning_materials(topic, tasks)  # Pass tasks too
        # Wrap the result in the expected format
        return jsonify({
            "status": "success",
            "materials": result
        })
    except Exception as e:
        print(f"Error in get_learning_materials: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to fetch learning materials"
        }), 500

@ai_bp.route("/ai-env/chat", methods=["POST"])
@token_required
def ai_env_chat():
    data = request.json
    user_id = request.user_id
    
    result = AIService.handle_ai_chat(user_id, data)
    return jsonify(result)

@ai_bp.route("/ai-env/flashcards", methods=["POST"])
@token_required
def generate_flashcards():
    data = request.json
    try:
        result = AIService.generate_flashcards(data)
        return jsonify({
            "status": "success",
            "flashcards": result
        })
    except Exception as e:
        print(f"Error in generate_flashcards: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to generate flashcards"
        }), 500

@ai_bp.route("/ai-env/study-guide", methods=["POST"])
@token_required
def generate_study_guide():
    data = request.json
    try:
        result = AIService.generate_study_guide(data)
        return jsonify({
            "status": "success",
            "study_guide": result
        })
    except Exception as e:
        print(f"Error in generate_study_guide: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to generate study guide"
        }), 500