import os
import jwt
from datetime import datetime, timedelta
from flask import current_app
from resend import Resend
from bson import ObjectId
from app.utils.helpers import get_db, get_jwt_secret

class EmailService:
    @staticmethod
    def send_password_reset_email(email, reset_token):
        """Send password reset email using Resend.com"""
        try:
            # Get Resend API key from environment
            resend_api_key = os.environ.get('RESEND_API_KEY')
            if not resend_api_key:
                print("❌ RESEND_API_KEY not configured")
                return False
            
            resend = Resend(api_key=resend_api_key)
            
            # Get app name and URL from environment or config
            app_name = os.environ.get('APP_NAME', 'LearnFlow AI')
            app_url = os.environ.get('APP_URL', 'http://localhost:3000')
            
            reset_link = f"{app_url}/auth/reset-password?token={reset_token}"
            
            # Email content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                    .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                    .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
                    .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
                    .button {{ display: inline-block; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 12px 30px; text-decoration: none; border-radius: 5px; font-weight: bold; }}
                    .footer {{ text-align: center; margin-top: 30px; color: #666; font-size: 12px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1 style="color: white; margin: 0;">{app_name}</h1>
                    </div>
                    <div class="content">
                        <h2>Password Reset Request</h2>
                        <p>Hello,</p>
                        <p>You've requested to reset your password for your {app_name} account.</p>
                        <p>Click the button below to reset your password:</p>
                        <p style="text-align: center; margin: 30px 0;">
                            <a href="{reset_link}" class="button">Reset Password</a>
                        </p>
                        <p>Or copy and paste this link in your browser:</p>
                        <p style="background: #f0f0f0; padding: 10px; border-radius: 5px; word-break: break-all;">
                            {reset_link}
                        </p>
                        <p>This link will expire in 1 hour.</p>
                        <p>If you didn't request this password reset, please ignore this email.</p>
                    </div>
                    <div class="footer">
                        <p>&copy; {datetime.now().year} {app_name}. All rights reserved.</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Send email using Resend
            params = {
                "from": f"{app_name} <noreply@{os.environ.get('RESEND_DOMAIN', 'resend.dev')}>",
                "to": [email],
                "subject": f"Reset Your {app_name} Password",
                "html": html_content,
                "headers": {
                    "X-Entity-Ref-ID": f"password-reset-{datetime.now().timestamp()}"
                }
            }
            
            response = resend.Emails.send(params)
            print(f"✅ Password reset email sent to {email}")
            return True
            
        except Exception as e:
            print(f"❌ Error sending password reset email: {str(e)}")
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