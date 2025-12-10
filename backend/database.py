from sqlmodel import Session, select
from models import User, ChapterProgress, UserPreferences, engine
from typing import Optional
from datetime import datetime

def get_or_create_user(email: str, name: str, auth_method: str) -> User:
    """Get existing user or create a new one"""
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        if not user:
            user = User(
                email=email,
                name=name,
                auth_method=auth_method,
                created_at=datetime.now().isoformat()
            )
            session.add(user)
            session.commit()
            session.refresh(user)
        return user

def update_user_preferences(user_id: int, language_preference: str = None,
                          theme_preference: str = None, notification_enabled: bool = None) -> UserPreferences:
    """Update user preferences"""
    with Session(engine) as session:
        # Check if preferences already exist
        user_prefs = session.exec(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        ).first()

        if user_prefs:
            # Update existing preferences
            if language_preference:
                user_prefs.language_preference = language_preference
            if theme_preference:
                user_prefs.theme_preference = theme_preference
            if notification_enabled is not None:
                user_prefs.notification_enabled = notification_enabled
        else:
            # Create new preferences
            user_prefs = UserPreferences(
                user_id=user_id,
                language_preference=language_preference or "en",
                theme_preference=theme_preference or "light",
                notification_enabled=notification_enabled if notification_enabled is not None else True
            )
            session.add(user_prefs)

        session.commit()
        session.refresh(user_prefs)
        return user_prefs

def get_user_preferences(user_id: int) -> Optional[UserPreferences]:
    """Get user preferences"""
    with Session(engine) as session:
        user_prefs = session.exec(
            select(UserPreferences).where(UserPreferences.user_id == user_id)
        ).first()
        return user_prefs

def update_chapter_progress(user_id: int, chapter_id: str, completed: bool = False) -> ChapterProgress:
    """Update chapter progress for a user"""
    with Session(engine) as session:
        # Check if progress record already exists
        progress = session.exec(
            select(ChapterProgress).where(
                ChapterProgress.user_id == user_id,
                ChapterProgress.chapter_id == chapter_id
            )
        ).first()

        if progress:
            # Update existing progress
            progress.completed = completed
            progress.last_accessed = datetime.now().isoformat()
        else:
            # Create new progress record
            progress = ChapterProgress(
                user_id=user_id,
                chapter_id=chapter_id,
                completed=completed,
                last_accessed=datetime.now().isoformat()
            )
            session.add(progress)

        session.commit()
        session.refresh(progress)
        return progress

def get_user_chapter_progress(user_id: int, chapter_id: str) -> Optional[ChapterProgress]:
    """Get chapter progress for a specific user and chapter"""
    with Session(engine) as session:
        progress = session.exec(
            select(ChapterProgress).where(
                ChapterProgress.user_id == user_id,
                ChapterProgress.chapter_id == chapter_id
            )
        ).first()
        return progress