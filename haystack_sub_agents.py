# haystack_sub_agents.py
# Multi-agent pipeline with supervisor delegation architecture

import requests
import os
import json
import re
from typing import List, Dict, Optional, Any
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.core.errors import BreakpointException

# MCP server configuration
MCP_ENDPOINT = "http://localhost:8000/mcp"

# ============================================================================
# SUB-AGENT CAPABILITIES REGISTRY
# ============================================================================

SUB_AGENT_CAPABILITIES = {
    "math": {
        "name": "Math Agent",
        "skills": ["arithmetic", "calculations", "mathematical operations", "numbers"],
        "tools": ["add_numbers", "multiply_numbers"],
        "system_prompt": """You are a Math Specialist Agent. Your expertise is in mathematical calculations and numerical operations.
When asked to perform calculations:
1. Use the available tools: add_numbers(a, b) and multiply_numbers(a, b)
2. Format tool requests clearly like: "I'll use add_numbers with a=5, b=3"
3. Provide the final answer to the user after receiving tool results.""",
        "keywords": ["calculate", "add", "multiply", "sum", "product", "math", "number", "compute"]
    },
    "research": {
        "name": "Research Agent",
        "skills": ["information gathering", "web search", "fact finding", "documentation"],
        "tools": [],  # Will add web search tools later
        "system_prompt": """You are a Research Specialist Agent. Your expertise is in gathering information, finding facts, and conducting research.
Provide well-researched, factual answers with proper context and sources when available.""",
        "keywords": ["research", "find", "search", "information", "what is", "who is", "tell me about", "explain"]
    },
    "code": {
        "name": "Code Agent", 
        "skills": ["programming", "code generation", "debugging", "code review"],
        "tools": [],  # Will add code tools later
        "system_prompt": """You are a Code Specialist Agent. Your expertise is in programming, code generation, and debugging.
Help users with code-related tasks including writing functions, debugging errors, and explaining code.""",
        "keywords": ["code", "function", "program", "debug", "write", "implement", "python", "javascript"]
    },
    "general": {
        "name": "General Agent",
        "skills": ["general conversation", "fallback handling"],
        "tools": [],
        "system_prompt": """You are a General Assistant Agent. Handle general queries and conversations that don't fit specialized categories.
Be helpful, informative, and conversational.""",
        "keywords": []  # Fallback agent
    }
}

# ============================================================================
# MCP TOOL FUNCTIONS
# ============================================================================

def add_numbers(a: float, b: float) -> float:
    """Adds two numbers using MCP protocol"""
    import uuid
    req_id = str(uuid.uuid4())
    payload = {
        "id": req_id,
        "method": "add_numbers",
        "params": {"a": a, "b": b}
    }
    response = requests.post(MCP_ENDPOINT, json=payload, timeout=10).json()
    if response.get("error"):
        return f"Error: {response['error']}"
    return response.get("result")

def multiply_numbers(a: float, b: float) -> float:
    """Multiplies two numbers using MCP protocol"""
    import uuid
    req_id = str(uuid.uuid4())
    payload = {
        "id": req_id,
        "method": "multiply_numbers",
        "params": {"a": a, "b": b}
    }
    response = requests.post(MCP_ENDPOINT, json=payload, timeout=10).json()
    if response.get("error"):
        return f"Error: {response['error']}"
    return response.get("result")

# ============================================================================
# COMPONENT 1: AGENT ROUTER
# ============================================================================

@component
class AgentRouter:
    """
    Routes messages to the appropriate agent based on agent_name.
    Unlike ConditionalRouter, this preserves List[ChatMessage] type.
    """
    
    @component.output_types(
        math=Optional[List[ChatMessage]],
        research=Optional[List[ChatMessage]],
        code=Optional[List[ChatMessage]],
        general=Optional[List[ChatMessage]]
    )
    def run(self, messages: List[ChatMessage], agent_name: str) -> Dict[str, Any]:
        """
        Route messages to the appropriate agent output.
        """
        result = {
            "math": None,
            "research": None,
            "code": None,
            "general": None
        }
        
        # Route to the appropriate agent
        if agent_name in result:
            result[agent_name] = messages
        else:
            # Fallback to general
            result["general"] = messages
        
        return result

# ============================================================================
# COMPONENT 2: TASK CLASSIFIER
# ============================================================================

@component
class TaskClassifier:
    """Classifies user queries to determine which sub-agent should handle them"""
    
    def __init__(self):
        self.capabilities = SUB_AGENT_CAPABILITIES
    
    @component.output_types(agent_name=str, confidence=float, messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage]) -> Dict[str, Any]:
        """
        Analyze the last user message and classify which agent should handle it.
        Returns: agent_name, confidence score, and messages list
        """
        if not messages:
            return {
                "agent_name": "general",
                "confidence": 0.5,
                "messages": []
            }
        
        # Get the last user message
        last_message = messages[-1]
        query = last_message.content if hasattr(last_message, 'content') else str(last_message)
        query_lower = query.lower()
        
        # Check for math queries first (higher priority)
        math_keywords = SUB_AGENT_CAPABILITIES["math"]["keywords"]
        math_score = sum(1 for keyword in math_keywords if keyword in query_lower)
        
        # Check for numbers in the query (strong indicator of math)
        has_numbers = bool(re.search(r'\d+', query))
        if has_numbers and any(keyword in query_lower for keyword in ['add', 'multiply', 'sum', 'product', '+', '*', 'calculate', 'compute']):
            math_score += 3  # Boost math score significantly
        
        # Keyword-based classification for other agents
        scores = {"math": math_score}
        
        for agent_name, config in self.capabilities.items():
            if agent_name in ["general", "math"]:
                continue  # Skip general (fallback) and math (already scored)
            
            keywords = config.get("keywords", [])
            score = sum(1 for keyword in keywords if keyword in query_lower)
            if score > 0:
                scores[agent_name] = score
        
        # Determine best agent
        if scores and max(scores.values()) > 0:
            best_agent = max(scores, key=scores.get)
            max_score = scores[best_agent]
            confidence = min(0.9, 0.5 + (max_score * 0.1))  # Cap at 0.9
            
            print(f"🎯 TaskClassifier: Routing to '{best_agent}' agent (confidence: {confidence:.2f})")
            return {
                "agent_name": best_agent,
                "confidence": confidence,
                "messages": messages
            }
        else:
            # Fallback to general agent
            print(f"🎯 TaskClassifier: Routing to 'general' agent (fallback)")
            return {
                "agent_name": "general",
                "confidence": 0.5,
                "messages": messages
            }

# ============================================================================
# COMPONENT 2: SUB-AGENT EXECUTORS
# ============================================================================

@component
class MathAgentExecutor:
    """Executes math-related queries using specialized tools"""
    
    def __init__(self):
        self.tools = {
            "add_numbers": add_numbers,
            "multiply_numbers": multiply_numbers
        }
        self.system_prompt = SUB_AGENT_CAPABILITIES["math"]["system_prompt"]
        self.llm = OllamaChatGenerator(
            model="llama3.1:8b",
            url="http://localhost:11434"
        )
    
    @component.output_types(
        agent_name=str,
        response=str,
        tool_results=List[Dict],
        messages=List[ChatMessage]
    )
    def run(self, messages: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """Execute math agent with tool support"""
        if messages is None:
            return {
                "agent_name": "math",
                "response": "",
                "tool_results": [],
                "messages": []
            }
        
        print(f"🧮 MathAgent: Processing query...")
        
        # Add system prompt
        agent_messages = [ChatMessage.from_system(self.system_prompt)] + messages
        
        tool_results = []
        max_iterations = 3
        
        for iteration in range(max_iterations):
            # Call LLM
            llm_result = self.llm.run(messages=agent_messages)
            replies = llm_result.get("replies", [])
            
            if not replies:
                break
            
            reply = replies[0]
            content = reply.content if hasattr(reply, 'content') else str(reply)
            
            # Check for tool calls
            tool_pattern = r'(\w+_numbers)\s*(?:with|using)?\s*(?:a|arguments)?[=:\s]*(\d+(?:\.\d+)?)[,\s]+(?:b[=:\s]*)?(\d+(?:\.\d+)?)'
            match = re.search(tool_pattern, content.lower())
            
            if match:
                tool_name = match.group(1)
                a = float(match.group(2))
                b = float(match.group(3))
                
                if tool_name in self.tools:
                    print(f"  🔧 Executing {tool_name}(a={a}, b={b})")
                    result = self.tools[tool_name](a, b)
                    print(f"  📊 Result: {result}")
                    
                    tool_results.append({
                        "tool": tool_name,
                        "params": {"a": a, "b": b},
                        "result": result
                    })
                    
                    # Add tool result to conversation
                    tool_msg = ChatMessage.from_assistant(
                        f"Tool {tool_name} returned: {result}. Please provide the final answer."
                    )
                    agent_messages.append(reply)
                    agent_messages.append(tool_msg)
                    continue  # Run LLM again with tool result
            
            # No tool call or final response
            return {
                "agent_name": "math",
                "response": content,
                "tool_results": tool_results,
                "messages": agent_messages
            }
        
        # Max iterations reached
        return {
            "agent_name": "math",
            "response": "Unable to complete calculation",
            "tool_results": tool_results,
            "messages": agent_messages
        }

@component
class ResearchAgentExecutor:
    """Executes research-related queries"""
    
    def __init__(self):
        self.system_prompt = SUB_AGENT_CAPABILITIES["research"]["system_prompt"]
        self.llm = OllamaChatGenerator(
            model="llama3.1:8b",
            url="http://localhost:11434"
        )
    
    @component.output_types(
        agent_name=str,
        response=str,
        sources=List[str],
        messages=List[ChatMessage]
    )
    def run(self, messages: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """Execute research agent"""
        if messages is None:
            return {
                "agent_name": "research",
                "response": "",
                "sources": [],
                "messages": []
            }
        
        print(f"🔍 ResearchAgent: Processing query...")
        
        # Add system prompt
        agent_messages = [ChatMessage.from_system(self.system_prompt)] + messages
        
        # Call LLM
        llm_result = self.llm.run(messages=agent_messages)
        replies = llm_result.get("replies", [])
        
        if replies:
            reply = replies[0]
            content = reply.content if hasattr(reply, 'content') else str(reply)
            
            return {
                "agent_name": "research",
                "response": content,
                "sources": [],  # TODO: Add actual source tracking
                "messages": agent_messages + replies
            }
        
        return {
            "agent_name": "research",
            "response": "Unable to process research query",
            "sources": [],
            "messages": agent_messages
        }

@component
class CodeAgentExecutor:
    """Executes code-related queries"""
    
    def __init__(self):
        self.system_prompt = SUB_AGENT_CAPABILITIES["code"]["system_prompt"]
        self.llm = OllamaChatGenerator(
            model="llama3.1:8b",
            url="http://localhost:11434"
        )
    
    @component.output_types(
        agent_name=str,
        response=str,
        code_snippets=List[str],
        messages=List[ChatMessage]
    )
    def run(self, messages: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """Execute code agent"""
        if messages is None:
            return {
                "agent_name": "code",
                "response": "",
                "code_snippets": [],
                "messages": []
            }
        
        print(f"💻 CodeAgent: Processing query...")
        
        # Add system prompt
        agent_messages = [ChatMessage.from_system(self.system_prompt)] + messages
        
        # Call LLM
        llm_result = self.llm.run(messages=agent_messages)
        replies = llm_result.get("replies", [])
        
        if replies:
            reply = replies[0]
            content = reply.content if hasattr(reply, 'content') else str(reply)
            
            # Extract code blocks
            code_blocks = re.findall(r'```[\w]*\n(.*?)\n```', content, re.DOTALL)
            
            return {
                "agent_name": "code",
                "response": content,
                "code_snippets": code_blocks,
                "messages": agent_messages + replies
            }
        
        return {
            "agent_name": "code",
            "response": "Unable to process code query",
            "code_snippets": [],
            "messages": agent_messages
        }

@component
class GeneralAgentExecutor:
    """Handles general queries and fallback cases"""
    
    def __init__(self):
        self.system_prompt = SUB_AGENT_CAPABILITIES["general"]["system_prompt"]
        self.llm = OllamaChatGenerator(
            model="llama3.1:8b",
            url="http://localhost:11434"
        )
    
    @component.output_types(
        agent_name=str,
        response=str,
        messages=List[ChatMessage]
    )
    def run(self, messages: Optional[List[ChatMessage]] = None) -> Dict[str, Any]:
        """Execute general agent"""
        if messages is None:
            return {
                "agent_name": "general",
                "response": "",
                "messages": []
            }
        
        print(f"💬 GeneralAgent: Processing query...")
        
        # Add system prompt
        agent_messages = [ChatMessage.from_system(self.system_prompt)] + messages
        
        # Call LLM
        llm_result = self.llm.run(messages=agent_messages)
        replies = llm_result.get("replies", [])
        
        if replies:
            reply = replies[0]
            content = reply.content if hasattr(reply, 'content') else str(reply)
            
            return {
                "agent_name": "general",
                "response": content,
                "messages": agent_messages + replies
            }
        
        return {
            "agent_name": "general",
            "response": "I'm here to help! How can I assist you?",
            "messages": agent_messages
        }

# ============================================================================
# COMPONENT 3: RESULT AGGREGATOR
# ============================================================================

@component
class ResultAggregator:
    """Aggregates results from sub-agents and formats final response"""
    
    @component.output_types(final_response=str, agent_used=str, metadata=Dict)
    def run(
        self,
        agent_name: Optional[str] = None,
        response: Optional[str] = None,
        tool_results: Optional[List[Dict]] = None,
        sources: Optional[List[str]] = None,
        code_snippets: Optional[List[str]] = None,
        messages: Optional[List[ChatMessage]] = None
    ) -> Dict[str, Any]:
        """
        Aggregate results from whichever sub-agent was invoked.
        All parameters are optional since only one agent will provide output.
        """
        
        if not agent_name or not response:
            return {
                "final_response": "No response generated",
                "agent_used": "none",
                "metadata": {}
            }
        
        # Format response with agent attribution
        agent_display = SUB_AGENT_CAPABILITIES.get(agent_name, {}).get("name", agent_name)
        
        # Build metadata
        metadata = {
            "agent": agent_name,
            "agent_display": agent_display
        }
        
        if tool_results:
            metadata["tool_results"] = tool_results
        if sources:
            metadata["sources"] = sources
        if code_snippets:
            metadata["code_snippets"] = code_snippets
        
        print(f"✅ {agent_display} completed successfully")
        
        return {
            "final_response": response,
            "agent_used": agent_name,
            "metadata": metadata
        }

# ============================================================================
# PIPELINE CONSTRUCTION
# ============================================================================

def create_supervisor_pipeline() -> Pipeline:
    """Create the multi-agent supervisor pipeline"""
    
    pipeline = Pipeline()
    
    # Add components
    pipeline.add_component("classifier", TaskClassifier())
    pipeline.add_component("router", AgentRouter())
    pipeline.add_component("math_agent", MathAgentExecutor())
    pipeline.add_component("research_agent", ResearchAgentExecutor())
    pipeline.add_component("code_agent", CodeAgentExecutor())
    pipeline.add_component("general_agent", GeneralAgentExecutor())
    
    # Connect classifier to router
    pipeline.connect("classifier.agent_name", "router.agent_name")
    pipeline.connect("classifier.messages", "router.messages")
    
    # Connect router to agents (only one will execute based on routing)
    pipeline.connect("router.math", "math_agent.messages")
    pipeline.connect("router.research", "research_agent.messages")
    pipeline.connect("router.code", "code_agent.messages")
    pipeline.connect("router.general", "general_agent.messages")
    
    return pipeline

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("MULTI-AGENT SUPERVISOR PIPELINE")
    print("=" * 70)
    print("\nInitializing pipeline...")
    
    pipeline = create_supervisor_pipeline()
    
    print("\n✅ Pipeline ready!")
    print("\nAvailable agents:")
    for agent_name, config in SUB_AGENT_CAPABILITIES.items():
        print(f"  • {config['name']}: {', '.join(config['skills'])}")
    
    print("\nCommands:")
    print("  - 'quit' or 'exit' - Exit the program")
    print("  - Type your query and the supervisor will route it to the appropriate agent")
    print("-" * 70)
    
    conversation_history = []
    
    while True:
        try:
            user_input = input("\n🧑 You: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n👋 Goodbye!")
                break
            
            # Add user message to history
            user_message = ChatMessage.from_user(user_input)
            conversation_history.append(user_message)
            
            # Run pipeline
            print("\n" + "─" * 70)
            result = pipeline.run(
                data={
                    "classifier": {"messages": conversation_history}
                }
            )
            
            # Extract response from whichever agent executed
            agent_result = None
            agent_name = None
            
            # Check which agent produced output
            for agent in ["math_agent", "research_agent", "code_agent", "general_agent"]:
                if agent in result and result[agent]:
                    agent_result = result[agent]
                    agent_name = agent.replace("_agent", "")
                    break
            
            if agent_result:
                final_response = agent_result.get("response", "No response")
                agent_display = SUB_AGENT_CAPABILITIES.get(agent_name, {}).get("name", agent_name)
                
                print("─" * 70)
                print(f"🤖 {agent_display}: {final_response}")
                
                # Show metadata if available
                if agent_result.get("tool_results"):
                    print(f"\n📊 Tool Results:")
                    for tool_result in agent_result["tool_results"]:
                        print(f"   {tool_result['tool']}{tool_result['params']} = {tool_result['result']}")
                
                if agent_result.get("code_snippets"):
                    print(f"\n💻 Code snippets generated: {len(agent_result['code_snippets'])}")
                
                if agent_result.get("sources"):
                    print(f"\n📚 Sources: {len(agent_result['sources'])}")
                
                # Add assistant response to history
                assistant_message = ChatMessage.from_assistant(final_response)
                conversation_history.append(assistant_message)
            else:
                print("❌ No response from pipeline")
            
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()
