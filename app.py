from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
from uuid import uuid4
import json
import os

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage
)

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

app = Flask(__name__)

llm = ChatGoogleGenerativeAI(
    model="gemini-3.6-flash"
)

# File to persist chat history across server restarts
HISTORY_FILE = "chat_history.json"

def load_conversations():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_conversations(convos):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(convos, f, indent=4)
    except Exception as e:
        print("Error saving conversation history:", e)

# Load conversations into memory on startup
conversations = load_conversations()


# EcoAI's instructions
ECOAI_PROMPT = """
You are EcoAI, a friendly and practical environmental assistant.

Your job is to help people understand how to reuse, recycle, sort,
and safely dispose of waste materials.

Give clear, practical advice that ordinary people can follow.

When someone describes a waste item, organize your response using
these sections where appropriate:

### 1. What it is
Briefly identify the material and any important considerations.

### 2. What you should do
Recommend the best practical option:
- Reuse
- Recycle
- Compost
- Safe disposal

### 3. How to do it
Give simple, step-by-step instructions.

### 4. What to avoid
Mention mistakes that could harm people or the environment.

Important rules:
- Never claim that an item is recyclable everywhere.
- Recycling and disposal options vary by location.
- Ask for the user's location when local disposal instructions
  or recycling facilities are needed.
- Treat batteries, chemicals, paint, medicines, and electronics
  with appropriate caution.
- Never recommend burning waste or pouring hazardous substances
  into drains, waterways, or soil.
- If you are unsure what an item contains, ask the user to clarify.
- Keep answers practical, friendly, and easy to understand.
"""


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard/dash.html")


@app.route("/api/eco-chat", methods=["POST"])
def eco_chat():
    data = request.get_json(silent=True) or {}

    message = data.get("message", "").strip()
    chat_id = data.get("chat_id")

    if not message:
        return jsonify({
            "error": "Message cannot be empty."
        }), 400

    # Create a conversation if necessary or if chat_id is unknown
    if not chat_id or chat_id not in conversations:
        chat_id = str(uuid4())
        conversations[chat_id] = []

    try:
        messages = [
            SystemMessage(content=ECOAI_PROMPT)
        ]

        # Restore previous conversation history for context memory
        for chat in conversations[chat_id]:
            messages.append(HumanMessage(content=chat["user"]))
            messages.append(AIMessage(content=chat["assistant"]))

        # Add current user message
        messages.append(HumanMessage(content=message))

        # Ask Gemini with full history context
        response = llm.invoke(messages)
        reply = response.content

        # Handle responses that come as content blocks
        if isinstance(reply, list):
            reply = "\n".join(
                block.get("text", "")
                for block in reply
                if isinstance(block, dict)
            )

        if not isinstance(reply, str):
            reply = str(reply)

        # Save this exchange to memory and persist to file
        conversations[chat_id].append({
            "user": message,
            "assistant": reply
        })
        save_conversations(conversations)

        return jsonify({
            "chat_id": chat_id,
            "reply": reply
        }), 200

    except Exception as e:
        print("EcoAI Gemini error:", e)
        return jsonify({
            "error": "Something went wrong while getting a response."
        }), 500


@app.route("/api/new-chat", methods=["POST"])
def new_chat():
    chat_id = str(uuid4())
    conversations[chat_id] = []
    save_conversations(conversations)

    return jsonify({
        "chat_id": chat_id,
        "message": "New chat started."
    }), 200


if __name__ == "__main__":
    app.run(debug=True, port=2000)