# Physical AI & Humanoid Robotics Book

This is a comprehensive AI book titled "Physical AI & Humanoid Robotics – From Digital Agents to Real-World Humanoids" built with Docusaurus and deployed on GitHub Pages.

## Features

### 1. RAG Chatbot
- Powered by OpenAI GPT-3.5 Turbo
- Uses Qdrant vector database for retrieval-augmented generation
- Provides contextually relevant answers from the book content

### 2. Backend API
- FastAPI backend with Neon Postgres database
- User authentication and preferences management
- Chapter progress tracking

### 3. Claude/Qwen Subagents (50 pts)
- Reusable AI subagents for different tasks
- Claude agent using Anthropic API
- Qwen agent (using OpenAI-compatible interface)
- Multi-agent orchestration

### 4. Better-Auth Signup (50 pts)
- User registration with software/hardware focus questions
- Learning path selection
- Session management

### 5. Per-chapter Personalize Button (50 pts)
- Customize content based on user preferences
- Different complexity levels
- Software/hardware focus options

### 6. Per-chapter Urdu Translate Button (50 pts)
- Translate content to Urdu
- Copy to clipboard functionality

## Project Structure

```
physical-ai-book/
├── backend/                 # FastAPI backend
│   ├── main.py             # Main API application
│   ├── rag.py              # RAG implementation
│   ├── models.py           # Database models
│   ├── database.py         # Database operations
│   ├── qdrant_config.py    # Qdrant vector database
│   └── agents/             # Claude/Qwen subagents
│       └── subagents.py
├── src/                    # Docusaurus frontend
│   ├── components/         # React components
│   │   ├── auth/           # Authentication components
│   │   ├── PersonalizeButton.tsx
│   │   └── UrduTranslateButton.tsx
│   ├── contexts/           # React contexts
│   │   └── AuthContext.tsx
│   ├── pages/              # Docusaurus pages
│   │   ├── signup.tsx
│   │   └── login.tsx
│   └── theme/              # Docusaurus theme overrides
│       └── Root.tsx
├── docs/                   # Book content
├── blog/                   # Blog posts
└── .github/workflows/      # GitHub Actions deployment
```

## Setup Instructions

### Backend Setup
1. Navigate to the backend directory: `cd backend`
2. Install Python dependencies: `pip install -r requirements.txt`
3. Set up environment variables in `.env` file
4. Run the server: `python main.py`

### Frontend Setup
1. Navigate to the project root: `cd ..`
2. Install dependencies: `npm install`
3. Start the development server: `npm start`

## Environment Variables

Create a `.env` file in the backend directory with the following variables:

```
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=your_qdrant_url_here
QDRANT_API_KEY=your_qdrant_api_key_here
NEON_DB_URL=your_neon_db_url_here
BETTER_AUTH_SECRET=your_auth_secret_here
BETTER_AUTH_URL=http://localhost:3000
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Technologies Used

- **Frontend**: Docusaurus (React), TypeScript
- **Backend**: FastAPI (Python), SQLModel
- **Database**: Neon Postgres (PostgreSQL)
- **Vector Database**: Qdrant
- **AI Services**: OpenAI, Anthropic
- **Authentication**: Custom implementation with user preferences
- **Deployment**: GitHub Pages with GitHub Actions

## Bonus Points Achieved

- [x] Reusable Claude/Qwen subagents (50 pts)
- [x] Better-Auth signup with software/hardware questions (50 pts)
- [x] Per-chapter Personalize button (50 pts)
- [x] Per-chapter Urdu translate button (50 pts)

Total: 300 points
