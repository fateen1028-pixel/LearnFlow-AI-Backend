from flask import Blueprint, request, jsonify
from app.services.auth_service import AuthService
from app.utils.validators import validate_email_password
import re
from app.utils.helpers import get_db, get_jwt_secret

auth_bp = Blueprint('auth', __name__)

@auth_bp.route("/register", methods=["POST"])
def register():
    data = request.json
    email = data.get("email")
    password = data.get("password")
    name = data.get("name")

    # Validation
    error = validate_email_password(email, password)
    if error:
        return error
    
    if not name:
        return jsonify({"status": "error", "message": "Name is required"}), 400

    result = AuthService.register_user(email, password, name)
    return jsonify(result)

@auth_bp.route("/login", methods=["POST"])
def login():
    if not request.is_json:
        return jsonify({"status": "error", "message": "Content-Type must be application/json"}), 400

    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400

    email = data.get("email")
    password = data.get("password")

    # Validation
    error = validate_email_password(email, password)
    if error:
        return error

    result = AuthService.login_user(email, password)
    return jsonify(result)

@auth_bp.route("/auth/firebase", methods=["POST"])
def handle_firebase_auth():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400

    firebase_token = data.get('token')
    if not firebase_token:
        return jsonify({"status": "error", "message": "Firebase token is required"}), 400

    result = AuthService.authenticate_firebase_user(firebase_token)
    return jsonify(result)

@auth_bp.route("/debug/login", methods=["POST"])
def debug_login():
    print("🔍 DEBUG: /debug/login called")
    print(f"🔍 Headers: {dict(request.headers)}")
    print(f"🔍 Content-Type: {request.content_type}")
    print(f"🔍 JSON data: {request.get_json()}")
    
    return jsonify({
        "status": "debug",
        "message": "Debug endpoint working",
        "headers": dict(request.headers),
        "content_type": request.content_type,
        "data_received": request.get_json()
    })

@auth_bp.route("/debug/firebase", methods=["GET"])
def debug_firebase():
    from app.utils.helpers import get_firebase_status
    return jsonify(get_firebase_status())



@auth_bp.route("/auth/forgot-password", methods=["POST"])
def forgot_password():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400
    
    email = data.get('email')
    if not email:
        return jsonify({"status": "error", "message": "Email is required"}), 400
    
    import re
    if not re.match(r'^[^@]+@[^@]+\.[^@]+$', email):
        return jsonify({"status": "error", "message": "Invalid email format"}), 400
    
    db = get_db()
    user = db.users.find_one({"email": email})
    
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404

    from app.services.email_service import EmailService
    
    # Generate PIN
    pin = EmailService.generate_pin()

    # Store PIN (using ObjectId)
    EmailService.create_pin_entry(user["_id"], pin)

    # Send via email
    EmailService.send_password_reset_pin(email, pin)

    return jsonify({"status": "success", "message": "Password reset PIN sent"}), 200



@auth_bp.route("/auth/reset-password", methods=["POST"])
def reset_password():
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400
    
    email = data.get('email')
    pin = data.get('pin')
    new_password = data.get('password')
    
    if not email or not pin or not new_password:
        return jsonify({"status": "error", "message": "Email, PIN, and new password are required"}), 400
    
    if len(new_password) < 6:
        return jsonify({"status": "error", "message": "Password must be at least 6 characters"}), 400
    
    db = get_db()
    user = db.users.find_one({"email": email})
    if not user:
        return jsonify({"status": "error", "message": "User not found"}), 404
    
    from bson import ObjectId
    from app.services.email_service import EmailService

    # Verify PIN
    if not EmailService.verify_pin(ObjectId(user["_id"]), pin):
        return jsonify({"status": "error", "message": "Invalid or expired PIN"}), 400
    
    # Hash password with bcrypt (CONSISTENT WITH auth_service.py)
    import bcrypt
    hashed = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())

    # Update user
    db.users.update_one(
        {"_id": user["_id"]},
        {"$set": {"password": hashed}}
    )

    return jsonify({"status": "success", "message": "Password reset successfully"})



@auth_bp.route("/auth/verify-reset-token", methods=["POST"])
def verify_reset_token():
    """Verify if reset token is valid"""
    data = request.json
    if not data:
        return jsonify({"status": "error", "message": "No JSON data provided"}), 400
    
    token = data.get('token')
    if not token:
        return jsonify({"status": "error", "message": "Token is required"}), 400
    
    from app.services.email_service import EmailService
    payload = EmailService.verify_reset_token(token)
    
    if payload:
        return jsonify({
            "status": "success",
            "message": "Token is valid",
            "email": payload.get('email')
        })
    else:
        return jsonify({
            "status": "error",
            "message": "Invalid or expired token"
        }), 400