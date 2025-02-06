# Chatbot GenAI

<div align="center">

![Python Version](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Dependencies](https://img.shields.io/badge/dependencies-up%20to%20date-brightgreen)

An advanced document understanding and question-answering system powered by state-of-the-art NLP and computer vision technologies.

</div>

## 🌟 Features

- **Multi-Format Document Processing**: Automated processing of PDF, DOCX, and images
- **Advanced OCR**: Enhanced text extraction with OpenCV preprocessing
- **Intelligent Text Analysis**: NLP-based understanding and chunking
- **Vector Database**: Efficient semantic search capabilities
- **Multi-Language Support**: Built-in support for English and Turkish
- **Smart Q&A**: Context-aware responses with confidence scoring

## 🚀 Quick Start Guide

### 1. Clone & Setup
```bash
git clone https://github.com/yourusername/CoGen.git
cd CoGen

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Spacy model
python -m spacy download en_core_web_lg
```

### 2. Configuration
1. Copy `.env.example` to `.env`
2. Get HuggingFace token from https://huggingface.co/settings/tokens
3. Update `.env` file with your token:
```
HF_TOKEN=your_token_here
```

### 3. Prepare Documents
1. Create `demo-data` folder in project root
2. Add your documents (supported formats: PDF, DOCX, TXT, PNG, JPG)

### 4. Run System
Run these scripts in order:

```bash
# 1. Process and vectorize documents
python -m src.scripts.prepare_data

# 2. Initialize system and create vector store
python -m src.scripts.initialize_system

# 3. Train the model
python -m src.scripts.train_model

# 4. Start chatbot
python -m src.scripts.test_chatbot
```

### 5. Usage
- Type your questions in natural language
- Type 'quit' to exit
- System will provide answers with confidence scores and sources

### System Requirements
- Python 3.8+
- 8GB RAM recommended
- SSD storage recommended
- CUDA-compatible GPU (optional)

### Troubleshooting
1. If you see OCR errors:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install tesseract-ocr
   # On macOS
   brew install tesseract
   ```

2. If you see PDF processing errors:
   ```bash
   # On Ubuntu/Debian
   sudo apt-get install poppler-utils
   # On macOS
   brew install poppler
   ```

3. For GPU support:
   - Install CUDA toolkit
   - Update torch with: `pip install torch --upgrade`

### Note
- First run will download required models (~2GB)
- Redis cache is optional but recommended for better performance

## Project Purpose
This project is an AI system capable of automatically understanding, processing, and answering questions about documents. Its main objectives are:

- Automated processing of documents in various formats (PDF, DOCX, images)
- Understanding and analyzing document contents
- Generating intelligent answers to questions about documents
- Multi-language support (Turkish and English)
- Efficiently storing document content in a vector database

## Key Features

### 1. Document Processing
- Extracting text from PDF files
- Extracting text from images using OCR
- Processing DOCX documents
- Automatic language detection
- Text standardization

### 2. Text Analysis
- Advanced text preprocessing
- Technical term recognition
- NLP-based semantic analysis
- Keyword extraction
- Text chunking

### 3. Vector Database
- Efficient document indexing
- Semantic search
- Similarity calculation
- Fast query response

### 4. Chat Capabilities
- Natural language understanding
- Context-aware responses
- Confidence score calculation
- Source document referencing

## Installation

### Prerequisites

## How It Works

### 1. Document Processing Workflow
1. Place PDF/DOCX/Image files in the `demo-data/` folder
2. Perform OCR and text extraction
3. Standardize and clean the text
4. Split the text into logical chunks
5. Convert each chunk into a vector
6. Store the vectors in the vector database

### 2. Chat Flow
1. User asks a question
2. Convert the question into a vector
3. Find the most relevant document chunks
4. AI model generates a response
5. Reference the source documents

## Usage Examples

### 1. System Startup and Document Processing

## System Architecture and Working Principle

### Models Used
1. **Text Understanding**: 
   - Model: `all-MiniLM-L6-v2` (SentenceTransformers)
   - Feature: Text vectorization and semantic analysis
   - Deployment: Runs locally, internet connection required only for initial download

2. **OCR Operations**:
   - Model: Tesseract OCR
   - Language Support: Turkish and English
   - Feature: Extracting text from images and PDFs

3. **NLP Operations**:
   - Model: SpaCy `en_core_web_lg`
   - Feature: Language detection, named entity recognition
   - Deployment: Runs locally

### Execution Order

1. **System Setup**:

Document Processing Features:

PDF text extraction and OCR
Image OCR and preprocessing
DOCX parsing
Automatic language detection
Noise cleaning
Text normalization
Vector Operations:

Semantic vector transformation
Fast similarity search
Chunk-based indexing
Vector caching
AI Features:

Context-based question answering
Multi-document analysis
Confidence score calculation
Source referencing
Transfer learning support
Performance Features:

Batch processing
GPU support (CUDA)
Parallel processing
Memory optimization
Deployment Options
On-Premise Deployment:

All models run locally
No internet connection required (except for initial setup)
Data privacy is maintained
Customizable model training
Hybrid Deployment:

Main system runs locally
Internet connection required for HuggingFace models
Wider model selection
Lower resource usage
Cloud Deployment (Optional):

Entire system can run in the cloud
Access via API endpoints
Scalable architecture
Load balancing support
System Requirements
Minimum Requirements:

Python 3.8+
4GB RAM
2GB disk space
Poppler and Tesseract
Recommended Requirements:

Python 3.10+
8GB RAM
SSD storage
NVIDIA GPU (optional)

