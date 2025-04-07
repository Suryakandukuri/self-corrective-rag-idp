# Self-Corrective RAG IDP

A Retrieval-Augmented Generation (RAG) framework for Intelligent Document Processing (IDP) that combines ChromaDB for vector storage and Groq's Llama3-70b-8192 model for reasoning. This project enables efficient document retrieval, embedding, and question-answering with self-corrective mechanisms.

---

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Folder Structure](#folder-structure)
4. [Setup Instructions](#setup-instructions)
5. [Usage](#usage)
6. [How It Works](#how-it-works)
7. [Contributing](#contributing)
8. [License](#license)

---

## Project Overview

This project implements a self-corrective RAG framework for processing and querying documents stored in ChromaDB. The system uses:
- **ChromaDB** as a persistent vector database for storing embeddings.
- **Groq's Llama3-70b-8192 model** for reasoning and language understanding.
- **LangChain** for managing embeddings, vector stores, and retrieval workflows.

The system supports:
1. Persistent storage of document embeddings.
2. Retrieval of relevant documents based on user queries.
3. A grading mechanism to assess the relevance of retrieved documents.

---

## Features

- **ChromaDB Integration**: Persistent vector database to store and retrieve embeddings.
- **Groq's Llama3 Model**: Advanced reasoning capabilities with Llama3-70b-8192.
- **Self-Corrective Mechanism**: Grading retrieved documents for relevance.
- **JSON Data Handling**: Fetches and processes JSON responses from APIs.

---

## Folder Structure

```

self-corrective-rag-idp/
├── app.py                 \# Main application script
├── chromadb/              \# Persistent ChromaDB storage
│   ├── 1b89f3d0-a622-4e93-936d-fb0c25382900/
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── length.bin
│   │   └── link_lists.bin
│   ├── 076d46b2-8442-4792-b0fc-d748b47c9e25/
│   │   ├── data_level0.bin
│   │   ├── header.bin
│   │   ├── length.bin
│   │   └── link_lists.bin
│   └── chroma.sqlite3     \# ChromaDB SQLite database file
├── create_chromadb.py     \# Script to populate ChromaDB with embeddings
├── LICENSE                \# License file for the project
├── metadata_store.json    \# Metadata for ChromaDB collections
├── poetry.lock            \# Poetry lock file for dependencies
├── pyproject.toml         \# Poetry project configuration file
├── README.md              \# Project documentation (this file)
└── utils/                 \# Utility scripts and modules
├── build_rag.py       \# RAG implementation (vector DB management)
├── data_gatherer.py   \# Fetches JSON data for embedding and processing
└── llm.py             \# LLM initialization (Groq's Llama3 model)

```

---

## Setup Instructions

### Prerequisites

1. Python 3.8 or higher.
2. Install dependencies using Poetry:
```

poetry install

```
3. Set up environment variables in a `.env` file:
```

VECTOR_STORE=./chromadb/
GROQ_API_KEY=your_groq_api_key_here

```

### Install Required Libraries

Ensure you have the necessary libraries installed:
```

pip install chromadb langchain langchain_chroma langchain_community python-dotenv

```

---

## Usage

### Step 1: Populate the Vector Database

Run the `create_chromadb.py` script to fetch data and populate the vector database:
```

python create_chromadb.py

```

### Step 2: Run the Application

Start the main application to query documents:
```

python app.py

```

### Step 3: Query Documents

In `app.py`, modify the `question` variable to query specific topics:
```

question = "hospitals"

```
The system will retrieve relevant documents and grade their relevance.

---

## How It Works

1. **Data Gathering**:
   - The `data_gatherer.py` script fetches JSON responses from APIs.
   - These responses are processed into LangChain-compatible document objects.

2. **Embedding and Storage**:
   - The `build_rag.py` script uses HuggingFace BGE embeddings ("BAAI/bge-base-en-v1.5") to create document embeddings.
   - Embeddings are stored persistently in ChromaDB.

3. **Document Retrieval**:
   - A retriever is created using LangChain's Chroma integration.
   - Documents are retrieved based on semantic similarity to user queries.

4. **Grading Mechanism**:
   - The Groq LLM grades retrieved documents for relevance using a structured prompt.

---

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork this repository.
2. Create a new branch: `git checkout -b feature-name`.
3. Commit your changes: `git commit -m 'Add feature-name'`.
4. Push to your branch: `git push origin feature-name`.
5. Submit a pull request.

---

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.