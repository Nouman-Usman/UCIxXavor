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

# CREDENTIALS_FILE = 'plenary-caster-412619-3b8e91b60470.json'
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
sheet_name = 'Logs'  # Name of the sheet where BP logs are stored
def retrieve_data(spreadsheet_id, range_name):
    result = SERVICE.spreadsheets().values().get(spreadsheetId=spreadsheet_id, range=range_name).execute()
    return result.get('values', [])

def append_data(spreadsheet_id, range_name, values):
    body = {'values': values}
    result = SERVICE.spreadsheets().values().append(
        spreadsheetId=spreadsheet_id,
        range=range_name,
        valueInputOption='RAW',
        body=body
    ).execute()
    return result

def find_next_available_row(spreadsheet_id, sheet_name):
    range_name = f'{sheet_name}!A:A'
    data = retrieve_data(spreadsheet_id, range_name)
    return len(data) + 1

def filter_data_by_mac(data, mac_address):
    header = data[0]
    filtered_rows = [header]
    for row in data[1:]:
        if len(row) > 2 and row[2] == mac_address:
            filtered_rows.append(row)
    return filtered_rows
app = Flask(__name__)
# Get Groq API key
groq_api_key = os.getenv("GROQ_API")

model = 'llama3-8b-8192'

# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
)

with open('steps.json', 'r') as file:
    assessment_data = json.load(file)

# Get system prompt from JSON
system_prompt = assessment_data['system_prompt']
conversational_memory_length = 20  # number of previous messages the chatbot will remember during the conversation and
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)
BP_LOGS_FILE = 'bp_logs.json'

# Initialize or load BP logs from the JSON file
def load_bp_logs():
    if os.path.exists(BP_LOGS_FILE):
        with open(BP_LOGS_FILE, 'r') as file:
            return json.load(file)
    return {}

# Save BP logs to JSON file
def save_bp_logs(logs):
    with open(BP_LOGS_FILE, 'w') as file:
        json.dump(logs, file)

# Function to add a BP log
def add_bp_log(mac_address, systolic, diastolic):
    logs = load_bp_logs()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if mac_address not in logs:
        logs[mac_address] = []
    logs[mac_address].append({"systolic": systolic, "diastolic": diastolic, "timestamp": timestamp})
    save_bp_logs(logs)

# Function to retrieve BP logs for a MAC address
def get_bp_logs(mac_address):
    logs = load_bp_logs()
    return logs.get(mac_address, [])

# Function to get MAC address of the user's machine
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
def chat():
    user_question = request.json.get('question')
    
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=f"{system_prompt}\nEvaluation rules: {json.dumps(assessment_data['steps']['blood_pressure_evaluation'])}"),
                MessagesPlaceholder(variable_name="chat_history"),
                HumanMessagePromptTemplate.from_template("{human_input}")
            ]
        )

        # Create a conversation chain
        conversation = LLMChain(
            llm=groq_chat,
            prompt=prompt,
            verbose=False,
            memory=memory,
        )

        response = conversation.predict(human_input=user_question)
        response_markdown = markdown.markdown(response)
        if "bp" in user_question.lower() or "blood pressure" in user_question.lower():
            parts = user_question.lower().split()
            for part in parts:
                if '/' in part:
                    try:
                        email = "testing.xyz"
                        systolic, diastolic = map(int, part.split('/'))
                        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        mac_address = get_mac_address()
                        values_to_write = [[timestamp, email, mac_address, systolic, diastolic]]
                        next_row = find_next_available_row(spreadsheet_id, sheet_name)
                        append_range = f'{sheet_name}!A{next_row}'
                        append_result = append_data(spreadsheet_id, append_range, values_to_write)
                        print("Append result:", append_result)
                    except ValueError:
                        pass
        # Return the formatted response
        return jsonify({"answer": response_markdown})
        # return jsonify({"answer": response})

    return jsonify({"answer": "Sorry, I didn't understand that."})

@app.route('/get-bp-logs', methods=['GET'])
def get_bp_logs_route():
    mac_address = get_mac_address()
    range_name = f'{sheet_name}!A:F'
    data = retrieve_data(spreadsheet_id, range_name)
    filtered_logs = filter_data_by_mac(data, mac_address)
    logs = []
    for row in filtered_logs[1:]:  # Skip header row
        if len(row) >= 5:
            logs.append({
                'timestamp': row[0],
                'email': row[1],
                'mac_address': row[2],
                'diastolic': row[4],
                'systolic': row[3]
            })

    return jsonify({"logs": logs})

if __name__ == "__main__":
    app.run(host='0.0.0.0',port=8000)