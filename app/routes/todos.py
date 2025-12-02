# app/routes/todos.py

from flask import Blueprint, request, jsonify
from app.middleware.auth import token_required
from app.services.todo_service import TodoService

todos_bp = Blueprint('todos', __name__)

@todos_bp.route("/todos/plan/<plan_id>", methods=["GET"])
@token_required
def get_todos_for_plan(plan_id):
    user_id = request.user_id
    result = TodoService.get_todos_for_plan(user_id, plan_id)
    return jsonify(result)

@todos_bp.route("/toggle-todo/<todo_id>", methods=["POST"])
@token_required
def toggle_todo(todo_id):
    user_id = request.user_id
    result = TodoService.toggle_todo(user_id, todo_id)
    return jsonify(result)

@todos_bp.route("/delete-todo/<todo_id>", methods=["DELETE"])
@token_required
def delete_todo(todo_id):
    user_id = request.user_id
    result = TodoService.delete_todo(user_id, todo_id)
    return jsonify(result)

@todos_bp.route("/move-todo/<todo_id>", methods=["POST"])
@token_required
def move_todo(todo_id):
    data = request.json
    new_day = data.get("newDay")
    user_id = request.user_id
    
    result = TodoService.move_todo(user_id, todo_id, new_day)
    return jsonify(result)

@todos_bp.route("/edit-todo/<todo_id>", methods=["PUT"])
@token_required
def edit_todo(todo_id):
    data = request.json
    user_id = request.user_id
    
    result = TodoService.edit_todo(user_id, todo_id, data)
    return jsonify(result)