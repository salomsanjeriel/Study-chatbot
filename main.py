import os
from datetime import datetime
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pymongo import MongoClient
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# 1. Setup Environment and Variables [cite: 9, 11]
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MONGO_URL = os.getenv("MONGODB_URL")

# 2. Database Connection (MongoDB) [cite: 15, 16]
client = MongoClient(MONGO_URL)
db = client["study_bot_db"]
collection = db["chat_history"]

# 3. Initialize FastAPI [cite: 4, 10]
app = FastAPI(title="Study Bot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    user_id: str
    question: str

# 4. Study Bot Logic & Prompting [cite: 19, 20, 21]
# The system prompt ensures it stays an Academic Assistant [cite: 5, 21]
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AI Study helper. Help users with academic questions and remember previous context."),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

# Initialize LLM (Ensure you use a valid Groq model name) [cite: 13]
llm = ChatGroq(api_key=GROQ_API_KEY, model="llama-3.1-8b-instant")
chain = prompt | llm

# 5. Helper function to retrieve history [cite: 18]
def get_history_from_db(user_id: str):
    # Fetch last 10 messages to maintain context [cite: 7, 18]
    docs = collection.find({"user_id": user_id}).sort("timestamp", 1).limit(10)
    history = []
    for doc in docs:
        if doc["role"] == "user":
            history.append(HumanMessage(content=doc["message"]))
        else:
            history.append(AIMessage(content=doc["message"]))
    return history

# 6. API Endpoints [cite: 4, 6]
@app.get("/")
def home():
    return {"status": "Study Bot API is running"}

@app.post("/chat")
def chat_endpoint(request: ChatRequest):
    # Retrieve previous context [cite: 18]
    history = get_history_from_db(request.user_id)
    
    # Generate response using LLM and history [cite: 6, 14]
    response = chain.invoke({"history": history, "question": request.question})
    bot_message = response.content

    # Store messages in MongoDB for persistence [cite: 15, 17]
    timestamp = datetime.utcnow()
    collection.insert_many([
        {
            "user_id": request.user_id,
            "role": "user",
            "message": request.question,
            "timestamp": timestamp
        },
        {
            "user_id": request.user_id,
            "role": "assistant",
            "message": bot_message,
            "timestamp": timestamp
        }
    ])

    return {
        "user_id": request.user_id,
        "response": bot_message
    }