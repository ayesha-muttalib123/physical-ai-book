from sqlmodel import SQLModel, Field, create_engine, Session
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    email: str = Field(unique=True, index=True)
    name: str
    auth_method: str  # 'email' or 'google' or 'github'
    created_at: Optional[str] = Field(default=None)
    is_software_focused: Optional[bool] = Field(default=None)  # User preference for software/hardware focus
    last_accessed_chapter: Optional[str] = Field(default=None)
    learning_path: Optional[str] = Field(default=None)  # 'beginner', 'intermediate', 'advanced', 'researcher', 'engineer'


class ChapterProgress(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    chapter_id: str
    completed: bool = False
    time_spent: Optional[int] = 0  # in seconds
    last_accessed: Optional[str] = Field(default=None)


class UserPreferences(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    user_id: int = Field(foreign_key="user.id")
    language_preference: str = "en"  # Default to English
    theme_preference: str = "light"  # 'light' or 'dark'
    notification_enabled: bool = True


# Database setup
DATABASE_URL = os.getenv("NEON_DB_URL", "postgresql://user:password@localhost/dbname")
engine = create_engine(DATABASE_URL, echo=True)

def create_db_and_tables():
    SQLModel.metadata.create_all(engine)

def get_session():
    with Session(engine) as session:
        yield session