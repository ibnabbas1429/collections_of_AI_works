import os 

def create_directories():
    directories = [
        'experiment',
        'logs',
        'src',
        os.path.join('src', 'mcqgenerator')
    ]
    for directory in directories:
        os.makedirs(directory, exist_ok=True)

def create_files():
    files = [
        'data.txt',
        'README.md',
        'requirements.txt',
        'Response.json',
        'setup.py',
        'StreamlitAPP.py',
        'test.py',
        os.path.join('experiment', 'machinelearning.csv'),
        os.path.join('experiment', 'mcq.ipynb'),
        os.path.join('src', '__init__.py'),
        os.path.join('src', 'mcqgenerator', '__init__.py'),
        os.path.join('src', 'mcqgenerator', 'logger.py'),
        os.path.join('src', 'mcqgenerator', 'MCQGenerator.py'),
        os.path.join('src', 'mcqgenerator', 'utils.py')
    ]
    for file in files:
        if not os.path.exists(file):
            with open(file, 'w') as f:
                f.write('')
def setup():
    from setuptools import setup, find_packages
    setup(
    name='automated_mcq_generator',
    version='1.0.0',
    description='Automated MCQ Generator using Langchain and Google Gemini',
    author='Ismail Abbas Aminu ',
    author_email='ibnabbas1981a@gmail.com',
    packages=find_packages(),
    install_requires=[
        'langchain[google-genai]',
        'langchain',
        'streamlit',
        'python-dotenv',
        'PyPDF2'
    ],
    include_package_data=True,
    python_requires='>=3.7',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
)

if __name__ == "__main__":
    create_directories()
    create_files()
    setup()


