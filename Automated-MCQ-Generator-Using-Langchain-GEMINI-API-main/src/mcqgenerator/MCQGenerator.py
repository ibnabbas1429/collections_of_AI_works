import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file, get_data
from src.mcqgenerator.logger import logging

# import necessary packages

from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain

#Load environment variables from .env file
load_dotenv()
logging.info("Loading environment variables from .env file")

#Accessing the environmnt variables just like you would with os.environ

key = os.getenv("GOOGLE_API_KEY")
logging.info("Google Gemini API Key logging successfully")

llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

template = """
Text:{text} 
You are an expert Multi Choice Question maker, Given the above, it is your job to\
create a question of number multiple choice questions for {subject} students in {tone} tone.
Make sure the questions are not repeated and check all the questions to be confirming the text as well.
Make sure to format your response like RESPONSE_JSON \ below and use it as a guide. \ 
Ensure to make {number}  
 """


test_generation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template= template 

)

test_chain = LLMChain(llm=llm, prompt=test_generation_prompt, output_key="quiz", verbose=True)

logging.info("Setting up quiz generation chain")

template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:


"""

test_Evaluation_prompt = PromptTemplate(
    input_variables = ["text", "number", "subject", "tone", "response_json"],
    template= template2 

)

logging.info("Setting up test evaluation chain. ")
review_chain = LLMChain(llm=llm, prompt=test_generation_prompt, output_key="review", verbose=True )

#This is an overall Chain where we run the two chains in Sequence
logging.info("Setting up sequential chain for quiz generation and evaluation.")
generate_evaluate_chain = SequentialChain(
                    chains=[test_chain, review_chain], 
                    input_variables=["text", "number","subject", "tone", "response_json"],
                    output_variables=["quiz", "review"],
                     verbose = True)



