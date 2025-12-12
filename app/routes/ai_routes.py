# ai_routes.py

from flask import Blueprint, request, jsonify
from app.middleware.auth import token_required
from app.services.ai_service import AIService
from flask_cors import CORS
from app.utils.pinecone_service import is_pinecone_available

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

    try:
        # FIX: Call the correct service function 'get_learning_materials'
        # which is search-first and includes validation.
        materials = AIService.get_ai_generated_materials(topic)

        return jsonify({
            "status": "success",
            "materials": materials
        })
    except Exception as e:
        print("Error in get_learning_materials:", e)
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
        
        # FIX: Return the result directly. The service already
        # formats it as {"status": "success", "flashcards": [...]}.
        return jsonify(result)
        
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
        
        # FIX: Return the result directly. The service already
        # formats it as {"status": "success", "study_guide": {...}}.
        return jsonify(result)
        
    except Exception as e:
        print(f"Error in generate_study_guide: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to generate study guide"
        }), 500
    



@ai_bp.route("/ai-env/memory/stats", methods=["GET"])
@token_required
def get_memory_stats():
    """Get user's memory statistics."""
    user_id = request.user_id
    
    try:
        stats = AIService.get_user_memory_stats(user_id)
        return jsonify({
            "status": "success",
            "stats": stats
        })
    except Exception as e:
        print(f"Error getting memory stats: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to get memory statistics"
        }), 500


@ai_bp.route("/ai-env/memory/clear", methods=["POST"])
@token_required
def clear_memory():
    """Clear user's memory."""
    user_id = request.user_id
    
    try:
        result = AIService.clear_user_memory(user_id)
        return jsonify({
            "status": "success" if result.get("success") else "error",
            "message": result.get("message", ""),
            "success": result.get("success", False)
        })
    except Exception as e:
        print(f"Error clearing memory: {e}")
        return jsonify({
            "status": "error",
            "message": "Failed to clear memory"
        }), 500


@ai_bp.route("/ai-env/memory/status", methods=["GET"])
@token_required
def get_memory_status():
    """Check if memory system is available."""
    return jsonify({
        "status": "success",
        "memory_available": is_pinecone_available(),
        "message": "Memory system is available" if is_pinecone_available() else "Memory system not configured"
    })