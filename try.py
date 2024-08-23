import os
import json
import chainlit as cl
from langchain.chains import LLMChain
from langchain_core.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import SystemMessage
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain_groq import ChatGroq

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the Chainlit application
api_key = os.getenv("API", "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")
model_name = "llama3-8b-8192"

groq_chat = ChatGroq(
    groq_api_key=api_key,
    model_name=model_name
)

# Load assessment data from prompt.json
with open('prompt.json', 'r') as file:
    assessment_data = json.load(file)

system_prompt = assessment_data['system_prompt']
conversational_memory_length = 20  # Number of previous messages the chatbot will remember during the conversation
memory = ConversationBufferWindowMemory(k=conversational_memory_length, memory_key="chat_history", return_messages=True)

# Initialize BP logs storage
bp_logs = []

@cl.on_message
async def handle_message(message: cl.Message):
    user_question = message.content

    if user_question:
        # Log BP if the message contains BP information
        if "bp" in user_question.lower() or "blood pressure" in user_question.lower():
            bp_logs.append(user_question)

        # Check if the user wants to see their BP logs
        if "view bp logs" in user_question.lower():
            logs = "<br>".join(bp_logs) if bp_logs else "No BP logs available."
            await cl.Message(content=f"Your BP logs:<br>{logs}").send()
            return

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

        # Get response from the conversation chain
        response = conversation.predict(human_input=user_question)

        # Shorten the response
        max_length = 500  # Set max length for response
        if len(response) > max_length:
            response = response[:max_length] + "..."

        # Apply HTML formatting
        response_decorated = f"{response}"

        # Send the formatted response back to the user
        await cl.Message(content=response_decorated).send()

    else:
        await cl.Message(content="Sorry, I didn't understand that.").send()

@cl.on_chat_start
async def start():
    welcome_message = """
    I'm here to help you with any Hypertension Query 💡
    """
    await cl.Message(content=welcome_message).send()

if __name__ == "__main__":
    cl.run()
