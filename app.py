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

# with open('steps.json', 'r') as file:
#     assessment_data = json.load(file)
system_prompt = (
    '''
**Role Description:**

You are a helpful AI assistant guiding users through a hypertension assessment. Follow these guidelines:

1. **Information Gathering:**
   - Ask for the following details in a clear and concise manner:
     - Name
     - Age
     - Gender
     - Anti-hypertensive medication status (e.g., "Are you currently taking any blood pressure medication?")
     - Most recent blood pressure reading

2. **Question Clarity:** Ensure each question is easy to understand and avoids ambiguity.

3. **Blood Pressure Reading Format:** Insist on the format 'SYS/DIA' (e.g., '120/80'). Avoid asking for systolic and diastolic readings separately.

4. **Concise Responses:** Keep your responses informative but brief, aiming for under 300 characters.

5. **Evaluation Criteria:**
   - **On Treatment:**
     - Severe: SYS ≥ 160 or DIA ≥ 110
     - High: 150 ≤ SYS ≤ 159 or 100 ≤ DIA ≤ 109
     - Raised: 140 ≤ SYS ≤ 149 or 90 ≤ DIA ≤ 99
     - High Normal: 130 ≤ SYS ≤ 139 or 80 ≤ DIA ≤ 89
     - Low Normal: 100 ≤ SYS ≤ 129 and DIA < 80
     - Low: SYS < 100 and DIA < 80
   - **Not On Treatment:**
     - Severe: SYS ≥ 160 or DIA ≥ 110
     - High: 140 ≤ SYS ≤ 159 or 90 ≤ DIA ≤ 109
     - Normal: SYS < 140 and DIA < 90
6. Evaluation and Guidance:

   If On Treatment:
     Severe: Your blood pressure is very high. Sit quietly for 5 minutes and repeat the reading. If it remains high, contact your local hospital's maternity unit immediately.
     High: Your blood pressure is high. Sit quietly for 5 minutes and repeat the reading. If it remains high, contact your provider urgently.
     Raised: Your blood pressure is raised. No change in medication yet.
     High Normal: Your blood pressure is in the target range when on treatment. This is fine if you have no side effects.
     Low Normal: Your blood pressure is normal but you may need less treatment. Follow your medication change instructions if this persists for 2 days.
     Low: Your blood pressure is too low. Sit quietly for 5 minutes and repeat the reading. If it remains low, contact your provider urgently.

   If Not On Treatment:
    Severe: Your blood pressure is very high. Sit quietly for 5 minutes and repeat the reading. If it remains high, contact your local hospital's maternity unit for urgent assessment.
     High: Your blood pressure is high. Sit quietly for 5 minutes and repeat the reading. If 2 or more consecutive readings are high, contact your provider or local hospital’s maternity unit within 48 hours.
     Normal: Your blood pressure is normal.

6. Adherence: Ensure responses align with the evaluation rules and guidance provided.
'''
)
# Blood pressure evaluation rules
evaluation_rules = {
    "on_treatment": {
        "severe": "If SYS is 160 or more, or DIA is 110 or more, or if severe symptoms are present, respond with: "
                  "'Your blood pressure is very high. Sit quietly for 5 minutes and repeat the blood pressure reading. "
                  "If this is a repeat reading in the severe range, contact your local hospital’s maternity unit "
                  "immediately and go in for an urgent assessment today at the local hospital.'",
        "high": "If SYS is 150-159 or DIA is 100-109, respond with: "
                "'Your blood pressure is high. Sit quietly for 5 minutes and repeat the blood pressure reading. "
                "If this is a repeat reading in the high range, contact your provider urgently and arrange assessment today.'",
        "raised": "If SYS is 140-149 or DIA is 90-99, respond with: "
                  "'Your blood pressure is raised. No change in your medication yet.'",
        "high_normal": "If SYS is 130-139 or DIA is 80-89, respond with: "
                       "'Your blood pressure is in the target range when on treatment. This is fine provided that you "
                       "have no side effects.'",
        "low_normal": "If SYS is 100-129 and DIA is less than 80, respond with: "
                      "'Your blood pressure is normal but you may require less treatment. Follow your medication change "
                      "instructions if your blood pressure remains in this range for 2 days in a row.'",
        "low": "If SYS is less than 100 and DIA is less than 80, respond with: "
               "'Your blood pressure is too low. Sit quietly for 5 minutes and repeat the blood pressure reading. "
               "If this is a repeat reading in the low range, contact your provider urgently and arrange assessment today.'"
    },
    "not_on_treatment": {
        "severe": "If SYS is 160 or more or DIA is 110 or more, respond with: "
                  "'Your blood pressure is very high. Sit quietly for 5 minutes and repeat the blood pressure reading. "
                  "If this is a repeat reading in the severe range, immediately contact your local hospital’s maternity "
                  "unit for urgent assessment today at the hospital.'",
        "high": "If SYS is 140-159 or DIA is 90-109, respond with: "
                "'Your blood pressure is high. Sit quietly for 5 minutes and repeat the blood pressure reading. "
                "If 2 or more consecutive readings are in this high range, contact your provider or local hospital’s "
                "maternity assessment unit for review within 48 hours.'",
        "normal": "If SYS is less than 140 and DIA is less than 90, respond with: "
                  "'Your blood pressure is normal.'"
    }
}

# Get system prompt from JSON
# system_prompt = assessment_data['system_prompt']
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
                SystemMessage(content=f"{system_prompt}"),
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
        if "bp" in user_question.lower() or "blood pressure" in user_question.lower() or "/" in user_question:
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