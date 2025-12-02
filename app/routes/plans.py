from flask import Blueprint, request, jsonify
from app.middleware.auth import token_required
from app.services.plan_service import PlanService

plans_bp = Blueprint('plans', __name__)

@plans_bp.route("/generate-roadmap", methods=["POST"])
@token_required
def generate_roadmap():
    data = request.json
    user_id = request.user_id
    
    result = PlanService.generate_roadmap(data)
    return jsonify(result)

@plans_bp.route("/generate-todo", methods=["POST"])
@token_required
def generate_todo():
    data = request.json
    user_id = request.user_id
    
    result = PlanService.generate_todo_list(user_id, data.get("roadmap"))
    return jsonify(result)

@plans_bp.route("/refine", methods=["POST"])
@token_required
def refine():
    data = request.json
    roadmap = data.get("roadmap")
    instruction = data.get("instruction")

    if not roadmap or not instruction:
        return jsonify({"status": "error", "message": "Missing roadmap or instruction"}), 400

    result = PlanService.refine_roadmap(roadmap, instruction)
    return jsonify(result)

@plans_bp.route("/plans/active", methods=["GET"])
@token_required
def get_active_plans():
    user_id = request.user_id
    result = PlanService.get_active_plans(user_id)
    return jsonify(result)

@plans_bp.route("/plans/all", methods=["GET"])
@token_required
def get_all_plans():
    user_id = request.user_id
    result = PlanService.get_all_plans(user_id)
    return jsonify(result)

@plans_bp.route("/plans/<plan_id>", methods=["DELETE"])
@token_required
def delete_plan(plan_id):
    user_id = request.user_id
    result = PlanService.delete_plan(user_id, plan_id)
    return jsonify(result)

@plans_bp.route("/check-initial-data", methods=["GET"])
@token_required
def check_initial_data():
    user_id = request.user_id
    result = PlanService.check_initial_data(user_id)
    return jsonify(result)

# ADD THIS NEW ROUTE
@plans_bp.route("/todos/plan/<plan_id>/next-day-task", methods=["POST"])  # Change to POST
@token_required
def get_next_day_task(plan_id):
    user_id = request.user_id
    result = PlanService.get_next_day_task(user_id, plan_id)
    return jsonify(result)