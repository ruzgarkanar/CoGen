from setuptools import setup, find_packages

setup(
    name="chatbot-genai",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        'nltk>=3.8',
        'sentence-transformers>=2.2.2',
        'numpy>=1.24',
        'pathlib>=1.0',
        'python-dotenv>=1.0',
        'torch>=2.0',
        'transformers>=4.30',
        'pdf2image>=1.16.3',  
        'pytesseract>=0.3.10',  
        'python-docx>=0.8.11',  
        'redis>=4.5.0',  
        'Pillow>=9.5.0',
        'aiohttp>=3.8.0',
        'aiofiles>=23.1.0',
        'asyncio>=3.4.3'
    ]
)
