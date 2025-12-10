from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel
from typing import List, Optional
import os
from dotenv import load_dotenv
from datetime import datetime
from sqlmodel import Session
import asyncio

# Load environment variables
load_dotenv()

# Import RAG components
from rag import RAGChatbot
# Import database components
from models import User, UserPreferences, ChapterProgress, create_db_and_tables
from database import get_or_create_user, update_user_preferences, get_user_preferences, update_chapter_progress, get_user_chapter_progress
# Import subagents
from agents.subagents import ClaudeSubagent, QwenSubagent, MultiAgentOrchestrator

app = FastAPI(title="Physical AI & Humanoid Robotics RAG Chatbot",
              description="RAG-powered chatbot for the Physical AI & Humanoid Robotics book",
              version="1.0.0")

# Initialize RAG chatbot
rag_chatbot = RAGChatbot()

# Create database tables on startup
@app.on_event("startup")
def on_startup():
    create_db_and_tables()

# Pydantic models
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 500
    user_id: Optional[int] = None  # Optional user ID for tracking

class ChatResponse(BaseModel):
    response: str
    sources: Optional[List[str]] = []

class UserRegistrationRequest(BaseModel):
    email: str
    name: str
    auth_method: str
    is_software_focused: Optional[bool] = None
    learning_path: Optional[str] = None

class UserPreferencesRequest(BaseModel):
    language_preference: Optional[str] = None
    theme_preference: Optional[str] = None
    notification_enabled: Optional[bool] = None

class ChapterProgressRequest(BaseModel):
    chapter_id: str
    completed: bool = False

class SubagentRequest(BaseModel):
    task: str
    context: Optional[str] = ""
    preferred_agent: Optional[str] = "auto"  # "claude", "qwen", "auto", or "collaborative"

class SubagentResponse(BaseModel):
    response: str
    agent_used: Optional[str] = None
    agents_used: Optional[List[str]] = None
    task: str
    context: Optional[str] = ""

@app.get("/")
def read_root():
    return {"message": "Physical AI & Humanoid Robotics RAG Chatbot API"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(chat_request: ChatRequest):
    """
    Main chat endpoint that processes user queries using RAG
    """
    try:
        # Extract user query from the last message
        user_query = chat_request.messages[-1].content

        # Get response from RAG system
        result = rag_chatbot.get_response(user_query)

        # If user_id is provided, we could log the interaction or update stats
        if chat_request.user_id:
            # Here we could add logic to track user interactions
            pass

        return ChatResponse(response=result["response"], sources=result["sources"])

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/register")
async def register_user(registration: UserRegistrationRequest):
    """
    Register a new user or get existing user
    """
    try:
        user = get_or_create_user(
            email=registration.email,
            name=registration.name,
            auth_method=registration.auth_method
        )

        # Update user-specific preferences if provided
        if registration.is_software_focused is not None or registration.learning_path:
            with Session(engine) as session:
                user.is_software_focused = registration.is_software_focused
                user.learning_path = registration.learning_path
                session.add(user)
                session.commit()

        return {"user_id": user.id, "email": user.email, "name": user.name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.put("/users/{user_id}/preferences")
async def update_preferences(user_id: int, preferences: UserPreferencesRequest):
    """
    Update user preferences
    """
    try:
        updated_prefs = update_user_preferences(
            user_id=user_id,
            language_preference=preferences.language_preference,
            theme_preference=preferences.theme_preference,
            notification_enabled=preferences.notification_enabled
        )
        return updated_prefs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/preferences")
async def get_preferences(user_id: int):
    """
    Get user preferences
    """
    try:
        prefs = get_user_preferences(user_id=user_id)
        if not prefs:
            # Create default preferences if none exist
            prefs = update_user_preferences(user_id=user_id)
        return prefs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users/{user_id}/chapters/{chapter_id}/progress")
async def update_chapter_progress_endpoint(user_id: int, chapter_id: str, progress: ChapterProgressRequest):
    """
    Update chapter progress for a user
    """
    try:
        updated_progress = update_chapter_progress(
            user_id=user_id,
            chapter_id=chapter_id,
            completed=progress.completed
        )
        return updated_progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/users/{user_id}/chapters/{chapter_id}/progress")
async def get_chapter_progress_endpoint(user_id: int, chapter_id: str):
    """
    Get chapter progress for a user
    """
    try:
        progress = get_user_chapter_progress(user_id=user_id, chapter_id=chapter_id)
        return progress
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/subagents/process", response_model=SubagentResponse)
async def process_with_subagent(request: SubagentRequest):
    """
    Process a request using Claude/Qwen subagents
    """
    try:
        orchestrator = MultiAgentOrchestrator()

        if request.preferred_agent == "collaborative":
            result = await orchestrator.collaborative_response(
                task=request.task,
                context=request.context
            )
        else:
            result = await orchestrator.route_request(
                task=request.task,
                context=request.context,
                preferred_agent=request.preferred_agent
            )

        return SubagentResponse(
            response=result["response"],
            agent_used=result.get("agent_used"),
            agents_used=result.get("agents_used"),
            task=result["task"],
            context=result["context"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)