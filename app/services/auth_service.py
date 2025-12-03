import bcrypt
import jwt
from datetime import datetime, timedelta
from bson import ObjectId
from app.models.user import User
from app.utils.helpers import get_db, get_jwt_secret, initialize_firebase, verify_firebase_token
from app.services.email_service import EmailService

class AuthService:
    @staticmethod
    def register_user(email, password, name):
        users_col = get_db().users
        
        # Check if user already exists
        if users_col.find_one({"email": email}):
            return {"status": "error", "message": "User already exists"}, 400

        # Hash password
        hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

        # Create user
        user_doc = User.create_user_doc(email, hashed_password, name)
        result = users_col.insert_one(user_doc)
        user_id = str(result.inserted_id)

        # Generate JWT token
        token = jwt.encode({
            'user_id': user_id,
            'exp': datetime.utcnow() + timedelta(hours=24)
        }, get_jwt_secret(), algorithm="HS256")

        return {
            "status": "success",
            "message": "User registered successfully",
            "token": token,
            "user": User.get_public_user_data(user_doc)
        }

    @staticmethod
    def login_user(email, password):
        users_col = get_db().users
        
        user = users_col.find_one({"email": email})
        if not user:
            return {"status": "error", "message": "Invalid credentials"}, 401

        # Check if user has a password (Firebase users might not)
        if 'password' not in user:
            return {
                "status": "error", 
                "message": "This account uses Google authentication. Please sign in with Google."
            }, 401

        # Handle both bytes and string password storage
        stored_password = user['password']
        
        # If password is stored as string, convert to bytes
        if isinstance(stored_password, str):
            stored_password = stored_password.encode('utf-8')
        
        # Check password
        try:
            if bcrypt.checkpw(password.encode('utf-8'), stored_password):
                token = jwt.encode({
                    'user_id': str(user['_id']),
                    'exp': datetime.utcnow() + timedelta(hours=24)
                }, get_jwt_secret(), algorithm="HS256")

                return {
                    "status": "success",
                    "message": "Login successful",
                    "token": token,
                    "user": User.get_public_user_data(user)
                }
            else:
                return {"status": "error", "message": "Invalid credentials"}, 401
        except Exception as e:
            print(f"❌ Password check error: {str(e)}")
            return {"status": "error", "message": "Authentication error"}, 500

    @staticmethod
    def authenticate_firebase_user(firebase_token):
        try:
            # Check if Firebase is initialized
            if not initialize_firebase():
                return {
                    "status": "error", 
                    "message": "Firebase service not available. Please try again later."
                }, 503

            # Verify Firebase token
            decoded_token = verify_firebase_token(firebase_token)
            if not decoded_token:
                return {"status": "error", "message": "Invalid Firebase token"}, 401

            email = decoded_token.get('email')
            name = decoded_token.get('name', email.split('@')[0] if email else 'User')
            firebase_uid = decoded_token.get('uid')
            picture = decoded_token.get('picture')

            if not email:
                return {"status": "error", "message": "Email not found in token"}, 401

            users_col = get_db().users
            
            # Check if user exists
            user = users_col.find_one({"email": email})
            
            if not user:
                # Create new user
                user_doc = User.create_user_doc(email, None, name, firebase_uid, picture)
                result = users_col.insert_one(user_doc)
                user_id = str(result.inserted_id)
            else:
                user_id = str(user['_id'])

            # Generate JWT token
            jwt_token = jwt.encode({
                'user_id': user_id,
                'exp': datetime.utcnow() + timedelta(hours=24)
            }, get_jwt_secret(), algorithm="HS256")

            return {
                "status": "success",
                "message": "Authentication successful",
                "token": jwt_token,
                "user": {
                    "id": user_id,
                    "email": email,
                    "name": name,
                    "avatar": picture
                }
            }
            
        except Exception as e:
            print(f"❌ Firebase auth error: {str(e)}")
            return {
                "status": "error", 
                "message": f"Authentication failed: {str(e)}"
            }, 500
        
    @staticmethod
    def initiate_password_reset(email):
        """Initiate password reset process"""
        users_col = get_db().users
        
        # Find user by email
        user = users_col.find_one({"email": email})
        if not user:
            # Return success even if user doesn't exist (for security)
            print(f"ℹ️ Password reset requested for non-existent email: {email}")
            return {
                "status": "success",
                "message": "If an account exists with this email, you will receive a password reset link."
            }
        
        # Generate reset token
        reset_token = EmailService.generate_reset_token(user['_id'], email)
        if not reset_token:
            return {
                "status": "error",
                "message": "Failed to generate reset token. Please try again."
            }, 500
        
        # Send reset email
        email_sent = EmailService.send_password_reset_email(email, reset_token)
        
        if email_sent:
            # Store reset token in database (optional, for validation)
            users_col.update_one(
                {"_id": user['_id']},
                {"$set": {
                    "reset_token": reset_token,
                    "reset_token_expires": datetime.utcnow() + timedelta(hours=1),
                    "updated_at": datetime.utcnow()
                }}
            )
            
            return {
                "status": "success",
                "message": "Password reset email sent. Please check your inbox."
            }
        else:
            return {
                "status": "error",
                "message": "Failed to send password reset email. Please try again later."
            }, 500

    @staticmethod
    def reset_password(token, new_password):
        """Reset password using token"""
        # Verify token
        payload = EmailService.verify_reset_token(token)
        if not payload:
            return {
                "status": "error",
                "message": "Invalid or expired reset token."
            }, 400
        
        user_id = payload.get('user_id')
        email = payload.get('email')
        
        if not user_id or not email:
            return {
                "status": "error",
                "message": "Invalid reset token."
            }, 400
        
        users_col = get_db().users
        
        # Find user
        try:
            user = users_col.find_one({"_id": ObjectId(user_id), "email": email})
        except:
            user = None
        
        if not user:
            return {
                "status": "error",
                "message": "User not found."
            }, 404
        
        # Hash new password
        hashed_password = bcrypt.hashpw(new_password.encode('utf-8'), bcrypt.gensalt())
        
        # Update password and clear reset token
        update_result = users_col.update_one(
            {"_id": ObjectId(user_id)},
            {"$set": {
                "password": hashed_password,
                "updated_at": datetime.utcnow()
            },
            "$unset": {
                "reset_token": "",
                "reset_token_expires": ""
            }}
        )
        
        if update_result.modified_count > 0:
            return {
                "status": "success",
                "message": "Password has been reset successfully. You can now login with your new password."
            }
        else:
            return {
                "status": "error",
                "message": "Failed to reset password. Please try again."
            }, 500