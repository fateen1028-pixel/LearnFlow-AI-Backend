from flask import Blueprint, jsonify, request
from app.middleware.auth import token_required
from app.services.dashboard_service import DashboardService
from datetime import datetime

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "success",
        "message": "Welcome to the Learning Planner API!",
        "timestamp": datetime.now().isoformat()
    }), 200

@dashboard_bp.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "success",
        "message": "Server is running",
        "timestamp": datetime.now().isoformat()
    })

@dashboard_bp.route("/dashboard/data", methods=["GET"])
@token_required
def get_dashboard_data():
    user_id = request.user_id
    time_range = request.args.get('range', 'week')
    result = DashboardService.get_dashboard_data(user_id, time_range)
    return jsonify(result)

@dashboard_bp.route("/.well-known/appspecific/com.chrome.devtools.json")
def chrome_devtools():
    return jsonify({}), 200