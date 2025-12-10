from typing import Dict, Any, List
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

class ClaudeSubagent:
    """Reusable Claude subagent for various tasks"""

    def __init__(self):
        self.model = ChatAnthropic(
            model_name="claude-3-haiku-20240307",  # Using Claude 3 Haiku for cost-effective operations
            temperature=0.7,
            max_tokens=1000,
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY")
        )

    async def process_request(self, task: str, context: str = "") -> str:
        """Process a request using Claude"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert assistant for the Physical AI & Humanoid Robotics book. Your task is to help users understand concepts related to physical AI, humanoid robotics, digital agents, and real-world applications. Be helpful, accurate, and provide detailed explanations when needed."),
            ("human", f"Task: {task}\n\nContext: {context}\n\nPlease provide a helpful response based on the Physical AI & Humanoid Robotics book content.")
        ])

        chain = prompt | self.model
        response = await chain.ainvoke({})
        return response.content


class QwenSubagent:
    """Reusable Qwen subagent for various tasks"""

    def __init__(self):
        # Using OpenAI compatible endpoint for Qwen or a fallback model
        # In a real implementation, this would connect to a Qwen-specific API
        self.model = ChatOpenAI(
            model_name="gpt-3.5-turbo",  # Fallback to OpenAI model if Qwen API is not available
            temperature=0.7,
            max_tokens=1000
        )

    async def process_request(self, task: str, context: str = "") -> str:
        """Process a request using Qwen (or fallback model)"""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are Qwen, an AI assistant specialized in Physical AI & Humanoid Robotics. Your task is to help users understand concepts related to physical AI, humanoid robotics, digital agents, and real-world applications. Be helpful, accurate, and provide detailed explanations when needed."),
            ("human", f"Task: {task}\n\nContext: {context}\n\nPlease provide a helpful response based on the Physical AI & Humanoid Robotics book content.")
        ])

        chain = prompt | self.model
        response = await chain.ainvoke({})
        return response.content


class MultiAgentOrchestrator:
    """Orchestrates multiple subagents for complex tasks"""

    def __init__(self):
        self.claude_agent = ClaudeSubagent()
        self.qwen_agent = QwenSubagent()

    async def route_request(self, task: str, context: str = "", preferred_agent: str = "auto") -> Dict[str, Any]:
        """Route request to appropriate agent based on task"""
        if preferred_agent == "claude":
            result = await self.claude_agent.process_request(task, context)
            agent_used = "claude"
        elif preferred_agent == "qwen":
            result = await self.qwen_agent.process_request(task, context)
            agent_used = "qwen"
        else:
            # Auto-routing logic - for now defaulting to Claude for complex tasks
            if any(keyword in task.lower() for keyword in ["explain", "describe", "how", "what", "why"]):
                result = await self.claude_agent.process_request(task, context)
                agent_used = "claude"
            else:
                result = await self.qwen_agent.process_request(task, context)
                agent_used = "qwen"

        return {
            "response": result,
            "agent_used": agent_used,
            "task": task,
            "context": context
        }

    async def collaborative_response(self, task: str, context: str = "") -> Dict[str, Any]:
        """Get collaborative response from both agents"""
        claude_response = await self.claude_agent.process_request(task, context)
        qwen_response = await self.qwen_agent.process_request(task, context)

        # Combine responses (in a real implementation, this could be more sophisticated)
        combined_response = f"CLAUDE'S RESPONSE:\n{claude_response}\n\nQWEN'S PERSPECTIVE:\n{qwen_response}"

        return {
            "response": combined_response,
            "agents_used": ["claude", "qwen"],
            "task": task,
            "context": context,
            "claude_response": claude_response,
            "qwen_response": qwen_response
        }


# Example usage functions
async def get_claude_analysis(task: str, context: str = ""):
    """Get analysis from Claude subagent"""
    agent = ClaudeSubagent()
    return await agent.process_request(task, context)


async def get_qwen_analysis(task: str, context: str = ""):
    """Get analysis from Qwen subagent"""
    agent = QwenSubagent()
    return await agent.process_request(task, context)


async def get_multi_agent_response(task: str, context: str = "", approach: str = "auto"):
    """Get response using multi-agent orchestration"""
    orchestrator = MultiAgentOrchestrator()

    if approach == "collaborative":
        return await orchestrator.collaborative_response(task, context)
    else:
        return await orchestrator.route_request(task, context, approach)