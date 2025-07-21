Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/.venv Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/experiment Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/log Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/logs Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/src Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/.env Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/.gitignore Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/data.txt Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/Dockerfile Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/README.md Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/requirements.txt Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/Response.json Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/setup.py Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/StreamlitAPP.py Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/test.py Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/unused_setup.py# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

# Copy the rest of the application code
#COPY Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/ ./

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variables (optional, for Streamlit)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501

# Run the Streamlit app
CMD ["streamlit", "run", "StreamlitAPP.py"]
