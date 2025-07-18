import os
import PyPDF2
import json
import traceback
from src.mcqgenerator.logger import logging


def read_file(file):
    logging.info(f"Reading file: %s", file.name)
    if file.name.endswith(".pdf"):
        try:
            pdf_reader=PyPDF2.PdfFileReader(file)
            text=""
            for page in pdf_reader.pages:
                text+=page.extract_text()
            logging.info("successfully read PDF file")
            return text
            
        except Exception as e:
            logging.error(f"Error reading the PDF file:{e}")
            raise Exception("error reading the PDF file")
        
    elif file.name.endswith(".txt"):
        try:
            content = file.read().decode("utf-8")
            logging.info("Successfully read TXT file.")
        
            return content
        except Exception as e:
            logging.error(f"Error reading the TXT file: {e}")
            raise Exception("error reading the TXT file")
    
    else:
        
        raise Exception(
            "unsupported file format only pdf and text file suppoted"
            )

def get_data(quiz_str):
    try:
        # convert the quiz from a str to dict
        quiz_dict=json.loads(quiz_str)
        quiz_table_data=[]
        
        # iterate over the quiz dictionary and extract the required information
        for key,value in quiz_dict.items():
            multiple_choice_question =value[" multiple_choice_question"]
            options=" || ".join(
                [
                    f"{option}-> {option_value}" for option, option_value in value["options"].items()
                 
                 ]
            )
            
            correct=value["correct"]
            quiz_table_data.append({" multiple_choice_question":  multiple_choice_question,"Choices": options, "Correct": correct})
        
        return quiz_table_data
        
    except Exception as e:
        traceback.print_exception(type(e), e, e.__traceback__)
        return False


