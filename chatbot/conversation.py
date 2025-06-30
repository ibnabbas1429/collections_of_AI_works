from chatbot.models import ConversationHistory, OnboardingStatus
from chatbot.database import SessionLocal
from dotenv import load_dotenv
import os
#from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, AIMessage

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

def fetch_history(session_id: str):
    db = SessionLocal()
    history = db.query(ConversationHistory).filter_by(session_id=session_id).order_by(ConversationHistory.timestamp).all()
    db.close()

    messages = []
    for h in history:
        if h.role == 'user':
            messages.append(HumanMessage(content=h.message))
        elif h.role == 'assistant':
            messages.append(AIMessage(content=h.message))
    return messages

def save_message(session_id: str, role: str, message: str):
    db = SessionLocal()
    record = ConversationHistory(session_id=session_id, role=role, message=message)
    db.add(record)
    db.commit()
    db.close()
    
def get_onboarding_status(db, session_id):
    """Fetch the onboarding status for a given session ID."""
    return db.query(OnboardingStatus).filter_by(session_id=session_id).first()

def update_onboarding(db, session_id, user_input):
    onboarding = get_onboarding_status(db, session_id)
    if not onboarding:
        onboarding = OnboardingStatus(session_id=session_id, step="ask_name")
        db.add(onboarding)
        db.commit()
        return "Welcome! What's your name?"
    if onboarding.step == "ask_name":
        onboarding.name = user_input
        onboarding.step = "ask_goal"
        db.commit()
        return f"Nice to meet you, {onboarding.name}! What's your goal for using this chatbot?"
    if onboarding.step == "ask_goal":
        onboarding.goal = user_input
        onboarding.step = "ask_experience"
        db.commit()
        return "Have you used similar tools before? (yes/no)"
    if onboarding.step == "ask_experience":
        onboarding.has_used_similar_tools = user_input
        onboarding.step = "done"
        db.commit()
        return "Thanks for sharing! You can now chat with the bot."
    return None

def get_bot_response(session_id: str, user_input: str) -> str:
    db = SessionLocal()
    try:
        onboarding = get_onboarding_status(db, session_id)
        if not onboarding or onboarding.step != "done":
            reply = update_onboarding(db, session_id, user_input)
            save_message(session_id, "user", user_input)
            save_message(session_id, "assistant", reply)
            return reply
        # Onboarding complete: normal chat
        # Prepare onboarding info as a system message
        onboarding_info = []
        if onboarding.name:
            onboarding_info.append(f"Name: {onboarding.name}")
        if onboarding.goal:
            onboarding_info.append(f"Goal: {onboarding.goal}")
        if onboarding.has_used_similar_tools:
            onboarding_info.append(f"Has used similar tools: {onboarding.has_used_similar_tools}")
        system_message = None
        if onboarding_info:
            system_message = f"User info: {', '.join(onboarding_info)}. Use this information to personalize your responses."
        history = fetch_history(session_id)
        if system_message:
            # Insert system message at the start
            from langchain.schema import SystemMessage
            history.insert(0, SystemMessage(content=system_message))
        history.append(HumanMessage(content=user_input))
        response = llm(history)
        save_message(session_id, "user", user_input)
        save_message(session_id, "assistant", response.content)
        return response.content
    finally:
        db.close()
