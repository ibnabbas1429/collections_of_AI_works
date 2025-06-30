from sqlalchemy import Column, Integer, String, Text, DateTime, func
from chatbot.database import Base

class ConversationHistory(Base):
    __tablename__ = "conversation_history"
    id = Column(Integer, primary_key=True, index=True)
    session_id = Column(String, index=True)
    role = Column(String)  # 'user' or 'assistant'
    message = Column(Text)
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

class OnboardingStatus(Base):
    __tablename__ = "onboarding_status"
    id = Column(Integer, primary_key=True)
    session_id = Column(String, index=True)
    name = Column(String)
    goal = Column(Text)
    has_used_similar_tools = Column(String)
    step = Column(String)  # "ask_name", "ask_goal", etc.
    timestamp = Column(DateTime(timezone=True), server_default=func.now())

