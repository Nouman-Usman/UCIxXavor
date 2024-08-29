import os
import json
from datetime import datetime
import uuid
from flask import Flask, render_template, request, jsonify
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq
import markdown
from dotenv import load_dotenv

load_dotenv()

credentials_info = {
    "type": os.getenv("GOOGLE_TYPE"),
    "project_id": os.getenv("GOOGLE_PROJECT_ID"),
    "private_key_id": os.getenv("GOOGLE_PRIVATE_KEY_ID"),
    "private_key": os.getenv("GOOGLE_PRIVATE_KEY").replace("\\n", "\n"),
    "client_email": os.getenv("GOOGLE_CLIENT_EMAIL"),
    "client_id": os.getenv("GOOGLE_CLIENT_ID"),
    "auth_uri": os.getenv("GOOGLE_AUTH_URI"),
    "token_uri": os.getenv("GOOGLE_TOKEN_URI"),
    "auth_provider_x509_cert_url": os.getenv("GOOGLE_AUTH_PROVIDER_X509_CERT_URL"),
    "client_x509_cert_url": os.getenv("GOOGLE_CLIENT_X509_CERT_URL"),
    "universe_domain": os.getenv("GOOGLE_UNIVERSE_DOMAIN"),
}
credentials = Credentials.from_service_account_info(credentials_info)
SCOPES = ['https://www.googleapis.com/auth/spreadsheets']
SERVICE = build('sheets', 'v4', credentials=credentials)
spreadsheet_id = os.getenv("FORM")
sheet_name = 'Logs'

app = Flask(__name__)

groq_api_key = os.getenv("GROQ_API")
model = 'llama3-8b-8192'

groq_chat = ChatGroq(
    groq_api_key=groq_api_key, 
    temperature=0.0,
    model_name=model
)

# Blood pressure evaluation rules
evaluation_rules = {
    "on_treatment": {
        "severe": "Your blood pressure is very high. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the severe range, contact your local hospital’s maternity unit immediately and go in for an urgent assessment today at the local hospital.",
        "high": "Your blood pressure is high. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the high range, contact your provider urgently and arrange assessment today.",
        "raised": "Your blood pressure is raised. No change in your medication yet.",
        "high_normal": "Your blood pressure is in the target range when on treatment. This is fine provided that you have no side effects.",
        "low_normal": "Your blood pressure is normal but you may require less treatment. Follow your medication change instructions if your blood pressure remains in this range for 2 days in a row.",
        "low": "Your blood pressure is too low. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the low range, contact your provider urgently and arrange assessment today."
    },
    "not_on_treatment": {
        "severe": "Your blood pressure is very high. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the severe range, immediately contact your local hospital’s maternity unit for urgent assessment today at the hospital.",
        "high": "Your blood pressure is high. Sit quietly for 5 minutes and repeat the blood pressure reading. If 2 or more consecutive readings are in this high range, contact your provider or local hospital’s maternity assessment unit for review within 48 hours.",
        "normal": "Your blood pressure is normal."
    }
}
system_prompt = (
    "Your Role: You help nurses in hypertension assessment. Details must ask: "
    "Name, Age, Gender, Weight, Marital Status, measured your blood pressure recently, "
    "currently on anti-hypertensive treatment. Your responses must be concise, under 300 characters, "
    "and adhere strictly to the evaluation rules provided. Always confirm the user's gender and anti-hypertensive "
    "treatment status before evaluating blood pressure. Each step must contain one question. The blood pressure reading "
    "must be in the following format: 'SYS/DIA'. For example, '120/80'. Never ask systolic and diastolic blood pressure "
    "readings separately. If the user is on treatment or not, ask for the most recent blood pressure reading. "
    "Evaluate the blood pressure and prescribe the patient with the next step mentioned below."
)

# Initialize or load BP logs from the JSON file
# def load_bp_logs():
#     if os.path.exists(BP_LOGS_FILE):
#         with open(BP_LOGS_FILE, 'r') as file:
#             return json.load(file)
#     return {}
conversational_memory_length = 20  # number of previous messages the chatbot will remember during the conversation and
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

def get_mac_address():
    mac_num = hex(uuid.getnode()).replace('0x', '').upper()
    return ':'.join(mac_num[i:i+2] for i in range(0, len(mac_num), 2))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])

@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    chat_context = request.json.get('context', {})  # Store the conversation context

    if user_question:
        # Store the on_treatment status if mentioned
        if "on treatment" in user_question.lower() or "treatment" in user_question.lower():
            chat_context["on_treatment"] = "yes" in user_question.lower()

        if "bp" in user_question.lower() or "blood pressure" in user_question.lower() or "/" in user_question:
            systolic, diastolic = None, None
            parts = user_question.lower().split()
            for part in parts:
                if '/' in part:
                    try:
                        systolic, diastolic = map(int, part.split('/'))
                        break
                    except ValueError:
                        pass
            
            if systolic and diastolic:
                on_treatment = chat_context.get("on_treatment", False)
                if on_treatment:
                    print("on_treatment")
                    if systolic >= 160 or diastolic >= 110:
                        response = evaluation_rules["on_treatment"]["severe"]
                    elif 150 <= systolic <= 159 or 100 <= diastolic <= 109:
                        response = evaluation_rules["on_treatment"]["high"]
                    elif 140 <= systolic <= 149 or 90 <= diastolic <= 99:
                        response = evaluation_rules["on_treatment"]["raised"]
                    elif 130 <= systolic <= 139 or 80 <= diastolic <= 89:
                        response = evaluation_rules["on_treatment"]["high_normal"]
                    elif 100 <= systolic <= 129 and diastolic < 80:
                        response = evaluation_rules["on_treatment"]["low_normal"]
                    else:
                        response = evaluation_rules["on_treatment"]["low"]
                else:
                    print("not_on_treatment")
                    # Determine the category
                    if systolic >= 160 or diastolic >= 110:
                        response = evaluation_rules["not_on_treatment"]["severe"]
                    elif 140 <= systolic <= 159 or 90 <= diastolic <= 109:
                        response = evaluation_rules["not_on_treatment"]["high"]
                    else:
                        response = evaluation_rules["not_on_treatment"]["normal"]

                response_markdown = markdown.markdown(response)
                return jsonify({"answer": response_markdown, "context": chat_context})
        else:
            # If no BP reading is provided, continue with the chat flow
            prompt = ChatPromptTemplate.from_messages(
                [
                    SystemMessage(content=f"{system_prompt}\nEvaluation rules: {evaluation_rules}"),
                    MessagesPlaceholder(variable_name="chat_history"),
                    HumanMessagePromptTemplate.from_template("{human_input}")
                ]
            )

            conversation = LLMChain(
                llm=groq_chat,
                prompt=prompt,
                verbose=False,
                memory=memory,
            )

            response = conversation.predict(human_input=user_question)
            response_markdown = markdown.markdown(response)
            return jsonify({"answer": response_markdown, "context": chat_context})

    return jsonify({"answer": "Sorry, I didn't understand that."})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000)
