""" An Automated Mutiple choice question 
Generator using Langchain and Gemini API"""

import os
import sys





def create_directories():
    directories = [
    "src",
    "src/mcqgenerator",
    "logs",
    "experiment",
    "mcqgenrator.egg-info"
]

    


    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created directory: {dir_path}")

def create_files():
    files = {
    "src/mcqgenerator/__init__.py": "",
    "src/mcqgenerator/logger.py": "# Logger setup\n",
    "src/mcqgenerator/utils.py": "# Utility functions\n",
    "src/mcqgenerator/MCQGenerator.py": "# MCQ Generator logic\n",
    "README.md": "# Project README\n",
    "requirements.txt": "",
    "setup.py": "# Setup script\n"
}



    for file_path, content in files.items():
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"Created file: {file_path}")

if __name__ == "__main__":
    create_directories()
    create_files()
    print("All required directories and files have been created.")


from setuptools import find_packages,setup

setup(
    name='mcqgenrator',
    version='0.0.1',
    author='ismail Abbas Aminu',
    author_email='ibnabbas19@amail.com',
    install_requires=["googlegenerativeai","langchain","streamlit","python-dotenv","PyPDF2"],
    packages=find_packages()
)


