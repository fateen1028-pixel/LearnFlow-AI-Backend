import os
import jwt
from datetime import datetime, timedelta
from flask import current_app
import resend
from bson import ObjectId
from app.utils.helpers import get_db, get_jwt_secret
import random

class EmailService:
    @staticmethod
    def generate_pin():
        """Generate a 6-digit numeric PIN"""
        return str(random.randint(100000, 999999))
    


    @staticmethod
    def create_pin_entry(user_id, pin):
        db = get_db()
        db.password_reset_pins.insert_one({
            "user_id": user_id,
            "pin": pin,
            "expires_at": datetime.utcnow() + timedelta(minutes=10)
    })

    @staticmethod
    def verify_pin(user_id, pin):
        db = get_db()
        record = db.password_reset_pins.find_one({
            "user_id": user_id,
            "pin": pin,
            "expires_at": {"$gt": datetime.utcnow()}
    })
        return record is not None
    
    @staticmethod
    def send_password_reset_pin(email, pin):
        """Send 6-digit password reset PIN using Resend.com"""
        try:
            resend_api_key = os.environ.get('RESEND_API_KEY')
            if not resend_api_key:
                print("❌ RESEND_API_KEY not configured")
                return False
            
            resend.api_key = resend_api_key
            app_name = os.environ.get('APP_NAME', 'LearnFlow AI')

            # Email content for PIN
            html_content = f"""
            <html>
            <body>
                <h2>Password Reset Request</h2>
                <p>Hello,</p>
                <p>Your password reset PIN for {app_name} is:</p>
                <div style="font-size: 28px; font-weight: bold; margin: 20px 0;">{pin}</div>
                <p>This PIN will expire in 10 minutes.</p>
                <p>If you didn't request a password reset, ignore this email.</p>
                <p>Best regards,<br>{app_name} Team</p>
            </body>
            </html>
            """

            params = {
                "from": f"{app_name} <onboarding@resend.dev>",
                "to": [email],
                "subject": f"{app_name} Password Reset PIN",
                "html": html_content
            }

            response = resend.Emails.send(params)
            print("📨 Resend response:", response)

            if isinstance(response, dict) and response.get("id"):
                print(f"✅ Password reset PIN sent to {email} (ID: {response['id']})")
                return True
            return False

        except Exception as e:
            print(f"❌ Error sending password reset PIN: {str(e)}")
            import traceback; traceback.print_exc()
            return False
    
    @staticmethod
    def generate_reset_token(user_id, email):
        """Generate JWT token for password reset"""
        try:
            payload = {
                'user_id': str(user_id),
                'email': email,
                'type': 'password_reset',
                'exp': datetime.utcnow() + timedelta(hours=1)  # Token expires in 1 hour
            }
            
            token = jwt.encode(payload, get_jwt_secret(), algorithm="HS256")
            return token
        except Exception as e:
            print(f"❌ Error generating reset token: {str(e)}")
            return None
    
    @staticmethod
    def verify_reset_token(token):
        """Verify password reset token"""
        try:
            payload = jwt.decode(token, get_jwt_secret(), algorithms=["HS256"])
            
            if payload.get('type') != 'password_reset':
                return None
            
            # Check if token is expired (jwt.decode will raise ExpiredSignatureError)
            return payload
        except jwt.ExpiredSignatureError:
            print("❌ Password reset token expired")
            return None
        except jwt.InvalidTokenError as e:
            print(f"❌ Invalid reset token: {str(e)}")
            return None
        except Exception as e:
            print(f"❌ Error verifying reset token: {str(e)}")
            return None