from fastapi import FastAPI
from pydantic import BaseModel
from chatbot.conversation import get_bot_response

app = FastAPI()

class UserMessage(BaseModel):
    session_id: str
    message: str

@app.post("/chat/")
def chat(user_msg: UserMessage):
    reply = get_bot_response(user_msg.session_id, user_msg.message)
    return {"reply": reply}
