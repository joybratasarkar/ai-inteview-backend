# langchain.py
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
import os  # Import the os module to use getenv for environment variables
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from langchain.memory import ConversationSummaryBufferMemory
from langchain.chains import ConversationChain

# Initialize ChatOpenAI and ChatPromptTemplate objects
load_dotenv()  # Load environment variables from the .env file

openai_api_key = os.getenv("OPENAI_API_KEY")

# Conditional logging for development environment
if os.getenv("ENV") == "development":
    print(f"OpenAI API Key: {openai_api_key}")  # Print the OpenAI API key

llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=openai_api_key,temperature=0)

output_parser = StrOutputParser()

# Define prompt templates
interview_prompt_template = """
Acknowledge the candidate's response briefly and provide a short review.
Keep it concise, like "OK", "Good", or "Interesting perspective", and then move on to the next question.
Give the answer in 20 words max in brief.
{response}
"""

general_prompt_template = """
Answer the following general question.
You are an AI bot of Supersourcing from Indore, India.
You can answer a wide range of questions, but you prefer to keep the conversation related to interviews.
Give the answer in 20 words max in brief.
<context>
{context}
</context>
"""
interview_end_prompt_template = """
Conclude the interview by acknowledging the candidate's participation.
You are an AI bot of Supersourcing from Indore, India.
Thank the candidate and provide a brief summary or closing remark.
Keep it concise, and limit it to 20 words max in brief.
{summary}
"""


memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=10)



output_parser = StrOutputParser()
interview_prompt = ChatPromptTemplate.from_template(interview_prompt_template)
chain_interview = interview_prompt | llm | output_parser 

interview_prompt_end = ChatPromptTemplate.from_template(interview_end_prompt_template)
chain_interview_end = interview_prompt_end | llm | output_parser 
