import os
from flask import Flask, render_template, request, jsonify
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
# from frt_processing import process_frt, live_frt
# from video_upload import upload_video

app = Flask(__name__)

# Get Groq API key
groq_api_key = os.environ.get("API", "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")
model = 'llama3-8b-8192'

# Initialize Groq Langchain chat object and conversation
groq_chat = ChatGroq(
        groq_api_key=groq_api_key, 
        model_name=model
)

system_prompt = '''You are tasked with performing a Hypertension Assessment. Follow these steps to gather and validate information from the user, specifically focusing on postpartum hypertension:

**Personal Details:**
Ask for the following details one by one:
- What is your name?
- What is your age?
- What is your gender?
- What is your weight (kg)?
- What is your height (cm)?
- What is your marital status?

**Blood Pressure Monitoring:**
Ask the user:
- Have you measured your blood pressure recently? If yes, what was the reading (systolic/diastolic)?
- Are you currently on anti-hypertensive treatment?

**Blood Pressure Evaluation (For women on anti-hypertensive treatment):**
- SEVERE: If SYS is 160 or more, or DIA is 110 or more, or if severe symptoms are present, respond with: "Your blood pressure is very high. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the severe range, contact your local hospital’s maternity unit immediately and go in for an urgent assessment today at the local hospital."
- HIGH: If SYS is 150-159 or DIA is 100-109, respond with: "Your blood pressure is high. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the high range, contact your provider urgently and arrange assessment today."
- RAISED: If SYS is 140-149 or DIA is 90-99, respond with: "Your blood pressure is raised. No change in your medication yet."
- HIGH NORMAL: If SYS is 130-139 or DIA is 80-89, respond with: "Your blood pressure is in the target range when on treatment. This is fine provided that you have no side effects."
- LOW NORMAL: If SYS is 100-129 and DIA is less than 80, respond with: "Your blood pressure is normal but you may require less treatment. Follow your medication change instructions if your blood pressure remains in this range for 2 days in a row."
- LOW: If SYS is less than 100 and DIA is less than 80, respond with: "Your blood pressure is too low. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the low range, contact your provider urgently and arrange assessment today."

**Blood Pressure Evaluation (For women NOT on anti-hypertensive treatment):**
- SEVERE: If SYS is 160 or more or DIA is 110 or more, respond with: "Your blood pressure is very high. Sit quietly for 5 minutes and repeat the blood pressure reading. If this is a repeat reading in the severe range, immediately contact your local hospital’s maternity unit for urgent assessment today at the hospital."
- HIGH: If SYS is 140-159 or DIA is 90-109, respond with: "Your blood pressure is high. Sit quietly for 5 minutes and repeat the blood pressure reading. If 2 or more consecutive readings are in this high range, contact your provider or local hospital’s maternity assessment unit for review within 48 hours."
- NORMAL: If SYS is less than 140 and DIA is less than 90, respond with: "Your blood pressure is normal."

**Presenting Complaints:**
Ask each question one by one. If the user answers “yes,” ask for the duration in days.
- Do you have headaches? If yes, mention the duration.
- Do you experience dizziness or lightheadedness? If yes, mention the duration.
- Do you have blurred vision? If yes, mention the duration.
- Do you feel chest pain or discomfort? If yes, mention the duration.
- Do you have shortness of breath? If yes, mention the duration.
- Do you experience fatigue or weakness? If yes, mention the duration.
- Do you notice any swelling in your ankles, feet, or legs? If yes, mention the duration.

**History of Presenting Illness:**
Ask the following questions one by one:
- What is the onset of your symptoms? (Sudden or Gradual)
- What factors aggravate the above symptoms?
- What factors relieve the symptoms?
- Have you been diagnosed with hypertension before? If yes, how long ago?
- Are you currently on any medication for hypertension? If yes, specify the medication and dosage.
- Any family history of hypertension or cardiovascular diseases?

**Conclusion:**
- If the blood pressure reading is in the SEVERE range or if chest pain or shortness of breath is present, respond with: "Your condition requires urgent medical attention. Please call 911 immediately."
- If any of the symptoms meet or exceed the specified duration and the blood pressure reading is in the HIGH or RAISED range, respond with: "Proceed with further hypertension evaluation."
- If none of the symptoms meet the criteria, and the blood pressure reading is normal, respond with: "There is no immediate need for further hypertension evaluation, but continue to monitor your symptoms." 
'''  # Same system_prompt as before

conversational_memory_length = 20  # number of previous messages the chatbot will remember during the conversation
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

UPLOAD_DIR = 'uploads'
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat', methods=['POST'])
def chat():
    user_question = request.json.get('question')
    
    if user_question:
        # Construct a chat prompt template using various components
        prompt = ChatPromptTemplate.from_messages(
            [
                SystemMessage(content=system_prompt),
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

        # Return the formatted response
        return jsonify({"answer": response_markdown})
        # return jsonify({"answer": response})

    return jsonify({"answer": "Sorry, I didn't understand that."})

if __name__ == "__main__":
    app.run(debug=True)