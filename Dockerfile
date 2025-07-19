# Use the official Python image as a base
FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code
COPY Automated-MCQ-Generator-Using-Langchain-GEMINI-API-main/ ./

# Expose the port Streamlit will run on
EXPOSE 8501

# Set environment variables (optional, for Streamlit)
ENV PYTHONUNBUFFERED=1
ENV STREAMLIT_SERVER_PORT=8501

# Run the Streamlit app
CMD ["streamlit", "run", "StreamlitAPP.py"]
