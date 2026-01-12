# haystack_mcp_pipeline.py
import requests
import os
import json
import re
from typing import List
from haystack import Pipeline, component
from haystack.components.builders import ChatPromptBuilder
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.dataclasses.breakpoints import Breakpoint
from haystack.core.errors import BreakpointException
from haystack.core.pipeline.breakpoint import load_pipeline_snapshot

# MCP server configuration
MCP_ENDPOINT = "http://localhost:8000/mcp"

# Module-level functions for MCP tools (serializable)
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


@component
class StateManager:
    """Component that manages pipeline state and context"""
    
    def __init__(self):
        self.state = {
            "user_preferences": {},
            "conversation_context": {},
            "tool_history": [],
            "last_calculation": None,
            "variables": {}
        }
    
    @component.output_types(state=dict, updated_messages=List[ChatMessage])
    def run(self, messages: List[ChatMessage]):
        """
        Process messages and update state based on content.
        Extract and store variables, preferences, and context.
        """
        updated_messages = []
        
        for msg in messages:
            content = msg.content if hasattr(msg, 'content') else str(msg)
            
            # Extract variable assignments like "let x = 5"
            var_pattern = r'(?:let|set|store)\s+(\w+)\s*=\s*([0-9.]+)'
            var_match = re.search(var_pattern, content.lower())
            if var_match:
                var_name = var_match.group(1)
                var_value = float(var_match.group(2))
                self.state["variables"][var_name] = var_value
                print(f"📌 Stored variable: {var_name} = {var_value}")
            
            # Replace variable references in messages
            for var_name, var_value in self.state["variables"].items():
                pattern = r'\b' + var_name + r'\b'
                content = re.sub(pattern, str(var_value), content, flags=re.IGNORECASE)
            
            # Update message with resolved variables
            if hasattr(msg, 'content'):
                updated_msg = ChatMessage(
                    role=msg.role if hasattr(msg, 'role') else msg._role,
                    content=content,
                    meta=msg.meta if hasattr(msg, 'meta') else {}
                )
                updated_messages.append(updated_msg)
            else:
                updated_messages.append(msg)
        
        # Add state context to messages if variables exist
        if self.state["variables"] and updated_messages:
            context_info = f"\n[Context: Variables = {self.state['variables']}]"
            last_msg = updated_messages[-1]
            if hasattr(last_msg, 'content'):
                updated_content = last_msg.content + context_info
                updated_messages[-1] = ChatMessage(
                    role=last_msg.role if hasattr(last_msg, 'role') else last_msg._role,
                    content=updated_content,
                    meta=last_msg.meta if hasattr(last_msg, 'meta') else {}
                )
        
        return {"state": self.state, "updated_messages": updated_messages}


@component
class MCPToolInvoker:
    """Component that detects and executes MCP tool calls from LLM responses"""
    
    def __init__(self):
        self.tools = {
            "add_numbers": add_numbers,
            "multiply_numbers": multiply_numbers
        }
    
    @component.output_types(messages=list, needs_llm=bool, tool_result=dict)
    def run(self, replies: list, state: dict = None):
        """
        Check LLM replies for tool calls and execute them.
        Returns updated messages and whether we need another LLM call.
        """
        messages = []
        needs_llm = False
        tool_result = {}
        
        for reply in replies:
            content = reply.content if hasattr(reply, 'content') else str(reply)
            
            # Check if LLM is requesting a tool call
            # Look for patterns like: "I'll use add_numbers with a=5, b=3"
            tool_pattern = r'(\w+_numbers)\s*(?:with|using)?\s*(?:a|arguments)?[=:\s]*(\d+(?:\.\d+)?)[,\s]+(?:b[=:\s]*)?(\d+(?:\.\d+)?)'
            match = re.search(tool_pattern, content.lower())
            
            if match:
                tool_name = match.group(1)
                a = float(match.group(2))
                b = float(match.group(3))
                
                if tool_name in self.tools:
                    print(f"🔧 Invoking tool: {tool_name}(a={a}, b={b})")
                    result = self.tools[tool_name](a, b)
                    print(f"📊 Tool result: {result}")
                    
                    # Store result in tool_result
                    tool_result = {
                        "tool": tool_name,
                        "params": {"a": a, "b": b},
                        "result": result
                    }
                    
                    # Update state with tool history if provided
                    if state:
                        state["tool_history"].append(tool_result)
                        state["last_calculation"] = result
                    
                    # Add tool result as a message
                    tool_msg = ChatMessage.from_assistant(
                        f"Tool {tool_name} returned: {result}. Please provide the final answer to the user."
                    )
                    messages.append(tool_msg)
                    needs_llm = True  # Need LLM to generate final response
                else:
                    messages.append(reply)
            else:
                # No tool call detected, pass through
                messages.append(reply)
        
        return {"messages": messages, "needs_llm": needs_llm, "tool_result": tool_result}


# Create Pipeline
pipeline = Pipeline()

# Add components
state_manager = StateManager()
prompt_builder = ChatPromptBuilder()
llm = OllamaChatGenerator(
    model="llama3.1:8b",
    url="http://localhost:11434"
)
tool_invoker = MCPToolInvoker()

# Add components to pipeline
pipeline.add_component("state_manager", state_manager)
pipeline.add_component("prompt_builder", prompt_builder)
pipeline.add_component("llm", llm)
pipeline.add_component("tool_invoker", tool_invoker)

# Connect components
pipeline.connect("state_manager.updated_messages", "prompt_builder.template")
pipeline.connect("prompt_builder", "llm")
pipeline.connect("llm.replies", "tool_invoker.replies")
pipeline.connect("state_manager.state", "tool_invoker.state")

# Breakpoint configuration
SNAPSHOT_DIR = "./pipeline_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

print("Pipeline ready! Type 'quit' or 'exit' to stop.")
print("Commands:")
print("  - 'pause' or 'bp' - Enable breakpoint on LLM call")
print("  - 'pause tool' - Enable breakpoint on tool invocation")
print("  - 'resume <file>' - Resume from snapshot file")
print("  - 'list' - List available snapshots")
print("  - 'state' - Show current state (variables, history)")
print("  - 'clear' - Clear all state")
print("Available tools: add_numbers, multiply_numbers")
print("Variable syntax: 'let x = 5' to store, then use 'x' in queries")
print("-" * 50)

breakpoint_enabled = False
breakpoint_component = "llm"  # "llm" or "tool_invoker"
conversation_history = []
system_prompt = "You are a helpful assistant with access to math tools. When asked to perform calculations, use the available tools: add_numbers(a, b) and multiply_numbers(a, b). Format your tool requests clearly."

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Handle pause command
        if user_input.lower().startswith('pause') or user_input.lower().startswith('bp'):
            parts = user_input.lower().split()
            if len(parts) == 1:
                breakpoint_enabled = not breakpoint_enabled
                breakpoint_component = "llm"
                status = "enabled (LLM)" if breakpoint_enabled else "disabled"
            elif parts[1] == 'tool':
                breakpoint_enabled = True
                breakpoint_component = "tool_invoker"
                status = "enabled (tool invoker)"
            else:
                print(f"Unknown breakpoint target: {parts[1]}")
                continue
            
            print(f"Breakpoint {status}")
            continue
        
        # Handle list snapshots
        if user_input.lower() == 'list':
            if os.path.exists(SNAPSHOT_DIR):
                files = sorted([f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.json')])
                if files:
                    print("\nAvailable snapshots:")
                    for f in files:
                        print(f"  - {f}")
                else:
                    print("No snapshots found")
            continue
        
        # Handle resume command
        if user_input.lower().startswith('resume '):
            snapshot_file = user_input[7:].strip()
            snapshot_path = os.path.join(SNAPSHOT_DIR, snapshot_file)
            if os.path.exists(snapshot_path):
                snapshot = load_pipeline_snapshot(snapshot_path)
                print(f"Loaded snapshot: {snapshot_file}")
                
                try:
                    print("Resuming execution...")
                    response = pipeline.run(data={}, pipeline_snapshot=snapshot)
                    
                    if response.get("llm", {}).get("replies"):
                        for reply in response["llm"]["replies"]:
                            content = reply.content if hasattr(reply, 'content') else str(reply)
                            print(f"\n✅ Pipeline (resumed): {content}")
                            conversation_history.append(ChatMessage.from_assistant(content))
                    else:
                        print(f"\n✅ Response: {response}")
                        
                except Exception as e:
                    print(f"Error resuming: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Snapshot not found: {snapshot_path}")
            continue
            
        if not user_input:
            continue
        
        # Add system prompt on first message
        if not conversation_history:
            conversation_history.append(ChatMessage.from_system(system_prompt))
        
        # Add user message to history
        conversation_history.append(ChatMessage.from_user(user_input))
        
        # Prepare breakpoint if enabled
        bp = None
        if breakpoint_enabled:
            print(f"⏸️  Breakpoint active - will pause at {breakpoint_component}")
            bp = Breakpoint(
                component_name=breakpoint_component,
                visit_count=0,
                snapshot_file_path=SNAPSHOT_DIR
            )
        
        # Run pipeline - may need multiple iterations if tools are invoked
        max_iterations = 3
        for iteration in range(max_iterations):
            try:
                result = pipeline.run(
                    data={
                        "state_manager": {"messages": conversation_history}
                    },
                    break_point=bp
                )
                
                # Check if tool invoker detected a tool call
                if result.get("tool_invoker", {}).get("needs_llm"):
                    # Tool was executed, add result to conversation and continue
                    tool_messages = result["tool_invoker"]["messages"]
                    for msg in tool_messages:
                        conversation_history.append(msg)
                    print("🔄 Running LLM again with tool result...")
                    continue  # Run pipeline again with tool result
                
                # Extract final response
                if result.get("llm", {}).get("replies"):
                    for reply in result["llm"]["replies"]:
                        content = reply.content if hasattr(reply, 'content') else str(reply)
                        print(f"\nPipeline: {content}")
                        conversation_history.append(ChatMessage.from_assistant(content))
                    break  # Done with this query
                elif result.get("tool_invoker", {}).get("messages"):
                    # No LLM needed, just show tool result
                    for msg in result["tool_invoker"]["messages"]:
                        content = msg.content if hasattr(msg, 'content') else str(msg)
                        print(f"\nPipeline: {content}")
                    break
                    
            except BreakpointException as e:
                print(f"\n🔴 Breakpoint triggered!")
                print(f"Component: {e.component}")
                
                # Find latest snapshot
                latest_snapshot = None
                if os.path.exists(SNAPSHOT_DIR):
                    files = sorted(
                        [f for f in os.listdir(SNAPSHOT_DIR) if f.endswith('.json')],
                        key=lambda x: os.path.getmtime(os.path.join(SNAPSHOT_DIR, x)),
                        reverse=True
                    )
                    if files:
                        latest_snapshot = files[0]
                        print(f"📸 Snapshot saved: {latest_snapshot}")
                        print(f"Location: {os.path.join(SNAPSHOT_DIR, latest_snapshot)}")
                
                print("\n💡 Commands:")
                if latest_snapshot:
                    print(f"  - Type 'resume {latest_snapshot}' to continue")
                else:
                    print(f"  - Type 'resume <filename>' to continue")
                print("  - Type 'list' to see all snapshots")
                
                breakpoint_enabled = False
                break  # Exit iteration loop
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
