import os
from typing import Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import asyncio
import json
from datetime import datetime
import secrets
import hashlib

# This is a simplified implementation since Better-Auth doesn't have a direct Python backend
# In a real implementation, you would use the official Better-Auth library
# For now, implementing a basic auth system that follows Better-Auth patterns

class SimpleAuth:
    def __init__(self, secret: str = "your-secret-key-change-in-production"):
        self.secret = secret
        self.users = {}  # In reality, this would be a database
        self.sessions = {}  # In reality, this would be a database
        self.user_fields = {
            "name": {"type": "string", "required": True},
            "email": {"type": "string", "required": True},
        }

    def add_user_fields(self, fields: Dict[str, Any]):
        """Add custom fields for personalization"""
        self.user_fields.update(fields)

    async def create_user(self, user_data: Dict[str, Any]):
        """Create a new user"""
        # Validate required fields
        for field, config in self.user_fields.items():
            if config.get("required", False) and field not in user_data:
                raise ValueError(f"Required field {field} is missing")

        # Create user ID
        user_id = f"user_{secrets.token_hex(16)}"

        # Hash password
        password_hash = hashlib.sha256(user_data["password"].encode()).hexdigest()

        # Create user object
        user = {
            "id": user_id,
            "email": user_data["email"],
            "name": user_data["name"],
            "password_hash": password_hash,
            "created_at": datetime.now().isoformat()
        }

        # Add custom fields
        for field in self.user_fields:
            if field in user_data and field not in ["email", "name", "password"]:
                user[field] = user_data[field]

        self.users[user_id] = user
        return user

    async def get_user(self, user_id: str):
        """Get user by ID"""
        user = self.users.get(user_id)
        if user:
            # Return a copy without password hash
            user_copy = user.copy()
            del user_copy["password_hash"]
            return type('User', (), user_copy)()
        return None

    async def update_user(self, user_id: str, update_data: Dict[str, Any]):
        """Update user data"""
        if user_id not in self.users:
            return None

        user = self.users[user_id]
        for key, value in update_data.items():
            if value is not None:  # Only update if value is provided
                user[key] = value

        # Return updated user (without password hash)
        user_copy = user.copy()
        if "password_hash" in user_copy:
            del user_copy["password_hash"]
        return type('User', (), user_copy)()

    def get_current_user(self, credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer())):
        """Get current user from session"""
        # In a real implementation, this would validate the token
        # For this example, we'll just return a mock user_id
        # In practice, you'd decode the JWT or validate the session
        token = credentials.credentials
        # This is simplified - in reality you'd decode the JWT
        user_id = self.sessions.get(token)
        if not user_id or user_id not in self.users:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
        return user_id

# Initialize auth system
auth = SimpleAuth()

# Add custom fields for personalization
custom_fields = {
    "software_background": {"type": "string", "required": False},
    "hardware_background": {"type": "string", "required": False},
    "content_depth_preference": {"type": "string", "required": False, "default": "standard"},
    "urdu_translation_enabled": {"type": "boolean", "required": False, "default": False}
}
auth.add_user_fields(custom_fields)

# Create signup form and personalization logic
async def create_user_with_profile(email: str, password: str, name: str, profile_data: Dict[str, Any]):
    """
    Create a new user with additional profile information for personalization
    """
    # Create user with Better Auth
    user_data = {
        "email": email,
        "password": password,
        "name": name,
        "software_background": profile_data.get("software_background", ""),
        "hardware_background": profile_data.get("hardware_background", ""),
        "content_depth_preference": profile_data.get("content_depth_preference", "standard"),
        "urdu_translation_enabled": profile_data.get("urdu_translation_enabled", False)
    }

    user = await auth.create_user(user_data)
    return user

async def update_user_profile(user_id: str, profile_data: Dict[str, Any]):
    """
    Update user profile for personalization
    """
    update_data = {}
    for field in ["software_background", "hardware_background", "content_depth_preference"]:
        if field in profile_data and profile_data[field] is not None:
            update_data[field] = profile_data[field]

    if "urdu_translation_enabled" in profile_data and profile_data["urdu_translation_enabled"] is not None:
        update_data["urdu_translation_enabled"] = profile_data["urdu_translation_enabled"]

    updated_user = await auth.update_user(user_id, update_data)
    return updated_user

async def get_user_profile(user_id: str):
    """
    Get user profile for personalization
    """
    user = await auth.get_user(user_id)
    if not user:
        return None

    return {
        "user_id": user.id,
        "name": user.name,
        "email": user.email,
        "software_background": getattr(user, 'software_background', ''),
        "hardware_background": getattr(user, 'hardware_background', ''),
        "content_depth_preference": getattr(user, 'content_depth_preference', 'standard'),
        "urdu_translation_enabled": getattr(user, 'urdu_translation_enabled', False)
    }

# Personalization functions
def get_content_level(user_profile: Dict[str, Any], default_level: str = "standard"):
    """
    Determine content depth based on user profile
    """
    if not user_profile:
        return default_level

    return user_profile.get("content_depth_preference", default_level)

def should_show_urdu_translation(user_profile: Dict[str, Any]) -> bool:
    """
    Check if Urdu translation should be shown based on user preference
    """
    if not user_profile:
        return False

    return user_profile.get("urdu_translation_enabled", False)

def personalize_content(content: str, user_profile: Optional[Dict[str, Any]]) -> str:
    """
    Apply personalization to content based on user profile
    """
    if not user_profile:
        return content

    # Adjust content based on user's background
    software_bg = user_profile.get("software_background", "").lower()
    hardware_bg = user_profile.get("hardware_background", "").lower()

    # Example: Add more technical details for users with software background
    if "software" in software_bg or "computer" in software_bg or "programming" in software_bg:
        content += "\n\n*Technical Note: This concept is particularly relevant for software engineers working with robotics frameworks.*"

    # Example: Add hardware-focused explanations for users with hardware background
    if "hardware" in hardware_bg or "electronics" in hardware_bg or "mechanical" in hardware_bg:
        content += "\n\n*Hardware Perspective: Consider the physical implementation challenges of this algorithm in real robotic systems.*"

    return content

# Example FastAPI integration
app = FastAPI()

@app.post("/api/auth/signup")
async def signup(
    email: str,
    password: str,
    name: str,
    software_background: Optional[str] = None,
    hardware_background: Optional[str] = None,
    content_depth_preference: Optional[str] = "standard",
    urdu_translation_enabled: Optional[bool] = False
):
    """
    Signup endpoint with profile information
    """
    profile_data = {
        "software_background": software_background,
        "hardware_background": hardware_background,
        "content_depth_preference": content_depth_preference,
        "urdu_translation_enabled": urdu_translation_enabled
    }

    user = await create_user_with_profile(email, password, name, profile_data)
    return {"user": user, "message": "User created successfully"}

@app.get("/api/auth/profile")
async def get_profile(user_id: str = Depends(auth.get_current_user)):
    """
    Get user profile
    """
    profile = await get_user_profile(user_id)
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    return profile

@app.put("/api/auth/profile")
async def update_profile(
    software_background: Optional[str] = None,
    hardware_background: Optional[str] = None,
    content_depth_preference: Optional[str] = None,
    urdu_translation_enabled: Optional[bool] = None,
    user_id: str = Depends(auth.get_current_user)
):
    """
    Update user profile
    """
    profile_data = {
        "software_background": software_background,
        "hardware_background": hardware_background,
        "content_depth_preference": content_depth_preference,
        "urdu_translation_enabled": urdu_translation_enabled
    }

    updated_user = await update_user_profile(user_id, profile_data)
    return {"user": updated_user, "message": "Profile updated successfully"}

# Example of how to use personalization in content delivery
@app.get("/api/content/{chapter_id}")
async def get_personalized_content(chapter_id: str, user_id: str = Depends(auth.get_current_user)):
    """
    Get personalized content for a chapter
    """
    # Get user profile
    user_profile = await get_user_profile(user_id)

    # Load chapter content (this would normally come from your docs)
    chapter_content = f"Content for chapter {chapter_id}"

    # Apply personalization
    personalized_content = personalize_content(chapter_content, user_profile)

    return {
        "chapter_id": chapter_id,
        "content": personalized_content,
        "personalization_applied": True,
        "urdu_translation_available": should_show_urdu_translation(user_profile)
    }

# Initialize the auth system
async def init_auth():
    print("Auth system initialized")

if __name__ == "__main__":
    asyncio.run(init_auth())