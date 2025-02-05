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

## 🚀 Quick Start

### Prerequisites

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

## 📋 Script Execution Guide

### Step by Step Run

- prepare_data.py → Processes documents, converts text to vectors.
- train_model.py → Fine-tune the DistilBERT model, making it produce better answers.
- test_chatbot.py → Receives user questions, finds relevant documents and generates answers.
