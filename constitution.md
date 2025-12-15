# Physical AI & Humanoid Robotics Book Project

**Version:** 2.0.0
**Author:** Ayesha Muttalib

**Description:** Constitution file for creating a full AI-native textbook with integrated RAG chatbot and user personalization via Better-Auth. Fully compliant with hackathon requirements. Agents and subagents must strictly follow rules to avoid hallucination.

## 1. Global Rules

- Only use technologies explicitly allowed by the hackathon:
  - Book: Docusaurus, Spec-Kit Plus
  - RAG: FastAPI, Neon Serverless Postgres, Qdrant Cloud Free Tier
  - Auth: Better-Auth
- Do not invent hardware, ROS2 commands, or APIs outside the course material
- Agents must strictly follow constitution and chapter content
- When unsure, respond: "Information not available in project context"
- All generated code must be clean, modular, and maintainable
- All content must follow course outline and references

## 2. Project Structure

### Root Directory
- `docusaurus/` - Book content
- `rag/` - RAG ingestion pipeline
- `auth/` - Better-Auth integration
- `agents/` - Main agents & subagents
- `scripts/` - Build and deploy scripts
- `.env.example`
- `package.json`
- `README.md`

### Docusaurus Structure
- `docs/`
- `src/`
  - `pages/`
  - `components/`
  - `styles/`
- `static/`
- `sidebars.js`
- `docusaurus.config.js`

### RAG Structure
- `ingest.py`
- `embed.py`
- `vector_store/`
  - `store.json`
- `utils/`
  - `chunker.py`
  - `loader.py`
  - `embeddings.py`

### Auth Structure
- `better_auth_integration.py`
- `signup_form.py`
- `personalization_logic.py`

### Agents Structure
- `rag-assistant.yaml`
- `book-writer.yaml`
- `editor.yaml`
- `subagents/`
  - `chapter-writer.yaml`
  - `code-writer.yaml`
  - `summarizer.yaml`
  - `citation-checker.yaml`

## 3. Book Specification

**Title:** Physical AI & Humanoid Robotics
**Subtitle:** From Digital Agents to Real-World Humanoids

**Purpose:** Full textbook to teach students Physical AI principles, ROS2, Gazebo, NVIDIA Isaac, humanoid locomotion, perception, and embodied intelligence.

### Chapters
- 00-introduction
- 01-physical-ai-basics
- 02-sensing-and-embodied-intelligence
- 03-ros2-nervous-system
- 04-digital-twin-gazebo-unity
- 05-nvidia-isaac-brain
- 06-vision-language-action
- 07-humanoid-locomotion
- 08-rag-chatbot-integration
- 09-auth-personalization
- 10-capstone-project

### Chapter Placeholders

#### 00-introduction
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 01-physical-ai-basics
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 02-sensing-and-embodied-intelligence
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 03-ros2-nervous-system
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 04-digital-twin-gazebo-unity
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 05-nvidia-isaac-brain
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 06-vision-language-action
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 07-humanoid-locomotion
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 08-rag-chatbot-integration
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 09-auth-personalization
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

#### 10-capstone-project
- **Overview:**
- **Why it matters:**
- **Key Concepts:** []
- **Code Examples:** []
- **Practical Examples:** []
- **Summary:**
- **Quiz:** []

### Chapter Rules
- Each chapter must include:
  - Overview
  - Why it matters
  - Key Concepts
  - Code Examples (ROS2, Gazebo, Isaac)
  - Practical Examples
  - Summary
  - Quiz
- Chapters must strictly follow course outline and references
- No hallucinated content

## 4. RAG Chatbot Pipeline

**Vector Store:** Qdrant Cloud Free Tier
**Chunk Size:** 800
**Overlap:** 100

### Retrieval Process
- Load embeddings from vector store
- Perform similarity search
- Return top 5 chunks for answer generation

### Constraints
- Only answer questions from book content
- If no relevant content found, respond: "Information not available in project context"

### RAG API Specification
**Framework:** FastAPI

**Endpoints:**
- `POST /api/rag/query`
  - Input: userQuery(string)
  - Output: answer, context_chunks

## 5. Authentication and Personalization

**Provider:** Better-Auth

### Fields Collected
- name
- email
- software_background
- hardware_background

### Personalization Features
- content_depth_toggle
- urdu_translation_toggle

### Integration
- Signup & Signin forms in Docusaurus pages
- Background info used to personalize chapter content

## 6. Agents & Subagents

### Main Agents
- **rag-assistant**
  - Role: Answer questions strictly from book content via RAG pipeline
  - No general knowledge: true
- **book-writer**
  - Role: Generate accurate textbook chapters from outlines
- **editor**
  - Role: Improve clarity and readability without adding facts

### Subagents
- **chapter-writer**: writes chapters from outlines
- **code-writer**: writes ROS2 / Gazebo / Isaac code examples
- **summarizer**: produces chapter summaries
- **citation-checker**: ensures content matches course outline

## 7. Deployment

### Hosting
- **Book Hosting:** GitHub Pages
- **RAG Hosting:** FastAPI backend
- **Auth Hosting:** Better-Auth

### Build Commands
- `npm run build`
- `npm run deploy`

## 8. Constraints

- Never invent ROS2, Gazebo, Isaac commands
- Never invent hardware or backend APIs outside the constitution
- All agents must follow the constitution strictly
- When unsure, respond: "Information not available in project context"

## 9. Prompt History Record (PHR)

**Description:** Log of all actions, prompts, and content edits for audit trail

### Schema
- **timestamp:** ISO 8601 timestamp
- **action_type:** create/update/delete/edit
- **content_type:** chapter/code/example/text/etc.
- **agent_responsible:** Agent name that performed action
- **prompt_used:** Exact prompt that triggered the action
- **changes_made:** Brief description of changes made
- **approval_status:** Draft/Reviewed/Published

### Example Entry
```
{
  "timestamp": "2025-01-15T10:30:00Z",
  "action_type": "create",
  "content_type": "chapter",
  "agent_responsible": "book-writer",
  "prompt_used": "Create introduction chapter with overview, key concepts, and examples",
  "changes_made": "Created 00-introduction chapter with all required sections",
  "approval_status": "Draft"
}
```

### PHR Log Entries will be appended here as actions occur:
```yaml
phr_entries: []
```

---

*End of Constitution*