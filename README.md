# Physical AI & Humanoid Robotics Book Project

This project creates a full AI-native textbook with integrated RAG chatbot and user personalization via Better-Auth, fully compliant with hackathon requirements.

## Project Structure

```
├── constitution.yml                 # Project constitution and rules
├── package.json                    # Project dependencies
├── docusaurus/                     # Book content and frontend
│   ├── docs/                       # Textbook chapters
│   ├── src/                        # React components and pages
│   ├── static/                     # Static assets
│   └── docusaurus.config.js        # Docusaurus configuration
├── rag/                           # RAG ingestion and API
│   ├── main.py                     # FastAPI application
│   ├── ingest.py                   # Content ingestion pipeline
│   ├── embed.py                    # Embedding and vector store management
│   └── utils/                      # Utility modules
├── auth/                          # Better-Auth integration
│   ├── better_auth_integration.py  # Authentication backend
│   ├── signup_form.py              # Signup form component
│   └── personalization_logic.py    # Content personalization
├── agents/                        # AI agents and subagents
│   ├── rag-assistant.yaml          # RAG assistant configuration
│   ├── book-writer.yaml            # Book writer configuration
│   ├── editor.yaml                 # Editor configuration
│   └── subagents/                  # Subagent configurations
└── scripts/                       # Build and deployment scripts
```

## Features

### 1. AI-Native Textbook
- Built with Docusaurus for modern documentation
- 11 comprehensive chapters covering Physical AI and Humanoid Robotics
- Each chapter follows constitutional requirements:
  - Overview
  - Why it matters
  - Key Concepts
  - Code Examples (ROS2, Gazebo, Isaac)
  - Practical Examples
  - Summary
  - Quiz

### 2. RAG Chatbot Integration
- FastAPI-powered backend
- Qdrant Cloud Free Tier vector store
- 800-chunk size with 100-overlap
- Retrieves top 5 chunks for answer generation
- Strictly answers from book content only

### 3. Authentication & Personalization
- Better-Auth integration
- Collects user background information
- Personalizes content depth
- Urdu translation toggle
- Signup & Signin forms in Docusaurus

### 4. AI Agent System
- **RAG Assistant**: Answers questions from book content
- **Book Writer**: Generates textbook chapters from outlines
- **Editor**: Improves clarity without adding facts
- **Subagents**:
  - Chapter Writer: Creates chapters from outlines
  - Code Writer: Generates ROS2/Gazebo/Isaac examples
  - Summarizer: Produces chapter summaries
  - Citation Checker: Ensures outline compliance

## Technology Stack

- **Book**: Docusaurus, Spec-Kit Plus
- **RAG**: FastAPI, Neon Serverless Postgres, Qdrant Cloud Free Tier
- **Auth**: Better-Auth
- **Agents**: YAML-based configuration system

## Setup Instructions

### 1. Install Dependencies

```bash
# For Docusaurus
npm install

# For RAG backend
cd rag
pip install -r requirements.txt
```

### 2. Environment Variables

Create a `.env` file in the root directory:

```env
# Qdrant Configuration
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key

# Better-Auth Configuration
BETTER_AUTH_SECRET=your_secret_key
DATABASE_URL=your_database_url
```

### 3. Initialize the System

```bash
# Start the RAG API
cd rag
python main.py

# Build and start the Docusaurus site
cd docusaurus
npm start
```

### 4. Process Book Content

```bash
# In the rag directory
python ingest.py
python embed.py
```

## Constitutional Compliance

This project strictly follows the constitutional requirements:
- Only uses approved technologies
- No hallucinated content
- Agents follow constitutional guidelines
- Strict adherence to course outline
- All content follows chapter structure requirements

## Deployment

- **Book Hosting**: GitHub Pages
- **RAG Hosting**: FastAPI backend
- **Auth Hosting**: Better-Auth

Build commands:
```bash
npm run build
npm run deploy
```

## Constraints

- Never invent ROS2, Gazebo, Isaac commands
- Never invent hardware or backend APIs outside the constitution
- All agents must follow the constitution strictly
- When unsure, respond: "Information not available in project context"