import streamlit as st
from src.mcqgenerator.utils import read_file, get_data
from src.mcqgenerator.MCQGenerator import llm, template
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
import json

st.title("Hello, Welcome to a Multiple Choice Question Generator")

uploaded_file = st.file_uploader("Upload a file in .txt or .pdf", type=["pdf", "txt"])

subject = st.text_input("Enter the subject of the question (e.g., Mathematics, Science, History)")
tone = st.selectbox("Select the tone of the question", ["Formal", "Informal", "Neutral"])
number = st.number_input("Enter the number of MCQs you want to generate", min_value=1, max_value=100, value=5)

if uploaded_file is not None:
    try:
        text = read_file(uploaded_file)
        st.success("File uploaded successfully!")

        if st.button("Generate MCQs"):
            with st.spinner("Generating MCQs..."):

                response_json = {
                    "1": {
                        "mcq": "multiple choice question",
                        "options": {
                            "a": "choice here",
                            "b": "choice here",
                            "c": "choice here",
                            "d": "choice here",
                        },
                        "correct": "correct answer",
                    },
                    "2": {
                        "mcq": "multiple choice question",
                        "options": {
                            "a": "choice here",
                            "b": "choice here",
                            "c": "choice here",
                            "d": "choice here",
                        },
                        "correct": "correct answer",
                    }
                }

                prompt = PromptTemplate(
                    input_variables=["text", "subject", "tone", "number", "response_json"],
                    template=template
                )

                def clean_gemini_output(ai_output):
                    content = ai_output.content.strip()
                    if content.startswith("```json"):
                        content = content.removeprefix("```json").removesuffix("```").strip()
                    elif content.startswith("```"):
                        content = content.removeprefix("```").removesuffix("```").strip()
                    return {"quiz": content}

                chain = prompt | llm | RunnableLambda(clean_gemini_output)

                result = chain.invoke({
                    "text": text,
                    "number": number,
                    "subject": subject,
                    "tone": tone,
                    "response_json": json.dumps(response_json)
                })

                st.subheader("Generated MCQs")
                try:
                    table_data = get_data(result["quiz"])
                    st.table(table_data)
                except Exception:
                    print(result)
                    st.write(result)

    except Exception as e:
        st.error(f"Error: {e}")
