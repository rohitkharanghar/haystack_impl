# haystack_mcp_agent.py
import requests
import os
from haystack.components.agents import Agent
from haystack_integrations.components.generators.ollama import OllamaChatGenerator
from haystack.tools import Tool
from haystack.dataclasses.breakpoints import AgentBreakpoint, Breakpoint, ToolBreakpoint
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


# Create MCP tools with module-level functions (serializable)
add_tool = Tool(
    name="add_numbers",
    description="Adds two numbers using MCP protocol",
    parameters={"a": {"type": "number"}, "b": {"type": "number"}},
    function=add_numbers
)

multiply_tool = Tool(
    name="multiply_numbers",
    description="Multiplies two numbers using MCP protocol",
    parameters={"a": {"type": "number"}, "b": {"type": "number"}},
    function=multiply_numbers
)



# Create Haystack Agent
llm = OllamaChatGenerator(
    model="llama3.1:8b",
    url="http://localhost:11434"
)

agent = Agent(
    chat_generator=llm,
    system_prompt="You are a coordinator that delegates research tasks to a specialist. if you are unable to asnwer using the tools, respond with 'I don't know'. just reply soory without explaination.",
    tools=[add_tool, multiply_tool],
    state_schema={
        "calc_result": {"type": float},
    }
)

# Interactive CLI loop
from haystack.dataclasses import ChatMessage

# Breakpoint configuration
SNAPSHOT_DIR = "./agent_snapshots"
os.makedirs(SNAPSHOT_DIR, exist_ok=True)

# Create breakpoints for debugging
# Uncomment to enable breakpoints on chat generator (LLM calls)
# chat_bp = Breakpoint(
#     component_name="chat_generator",
#     visit_count=0,
#     snapshot_file_path=SNAPSHOT_DIR
# )
# agent_breakpoint = AgentBreakpoint(
#     break_point=chat_bp,
#     agent_name="agent"
# )

# Uncomment to enable breakpoints on tool calls
# tool_bp = ToolBreakpoint(
#     component_name="tool_invoker",
#     visit_count=0,
#     tool_name="multiply_numbers",  # Specific tool, or None for any tool
#     snapshot_file_path=SNAPSHOT_DIR
# )
# agent_breakpoint = AgentBreakpoint(
#     break_point=tool_bp,
#     agent_name="agent"
# )

print("Agent ready! Type 'quit' or 'exit' to stop.")
print("Commands:")
print("  - 'pause' or 'bp' - Enable breakpoint on next tool call")
print("  - 'pause add' - Enable breakpoint only for add_numbers tool")
print("  - 'pause multiply' - Enable breakpoint only for multiply_numbers tool")
print("  - 'pause llm' - Enable breakpoint on LLM (chat generator) calls")
print("  - 'resume <file>' - Resume from snapshot file")
print("  - 'list' - List available snapshots")
print("-" * 50)

breakpoint_enabled = False
breakpoint_type = "tool"  # "tool", "llm", or specific tool name
breakpoint_tool = None
snapshot_to_resume = None

while True:
    try:
        user_input = input("\nYou: ").strip()
        
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
        
        # Handle pause command variations
        if user_input.lower().startswith('pause') or user_input.lower().startswith('bp'):
            parts = user_input.lower().split()
            if len(parts) == 1:
                # Toggle general tool breakpoint
                breakpoint_enabled = not breakpoint_enabled
                breakpoint_type = "tool"
                breakpoint_tool = None
                status = "enabled (all tools)" if breakpoint_enabled else "disabled"
            elif parts[1] == 'llm':
                # LLM breakpoint
                breakpoint_enabled = True
                breakpoint_type = "llm"
                status = "enabled (LLM calls)"
            elif parts[1] in ['add', 'add_numbers']:
                # Specific tool breakpoint
                breakpoint_enabled = True
                breakpoint_type = "tool"
                breakpoint_tool = "add_numbers"
                status = "enabled (add_numbers only)"
            elif parts[1] in ['multiply', 'multiply_numbers']:
                # Specific tool breakpoint
                breakpoint_enabled = True
                breakpoint_type = "tool"
                breakpoint_tool = "multiply_numbers"
                status = "enabled (multiply_numbers only)"
            else:
                print(f"Unknown breakpoint target: {parts[1]}")
                continue
            
            print(f"Breakpoint {status}")
            continue
        
        # Handle list snapshots command
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
                
                # Debug: inspect snapshot structure
                print(f"Snapshot type: {type(snapshot)}")
                if hasattr(snapshot, '__dict__'):
                    print(f"Snapshot attributes: {list(snapshot.__dict__.keys())}")
                
                # Resume immediately
                try:
                    print("Resuming execution...")
                    
                    # Agent requires messages parameter, pass empty list - snapshot has the state
                    response = agent.run(data={}, pipeline_snapshot=snapshot)
                    
                    print(f"Response keys: {response.keys()}")
                    
                    # Extract and print the reply - check all possible locations
                    found_response = False
                    
                    if response.get("replies"):
                        found_response = True
                        for reply in response["replies"]:
                            content = reply.content if hasattr(reply, 'content') else (reply.text if hasattr(reply, 'text') else str(reply))
                            print(f"\n✅ Agent (resumed): {content}")
                    
                    if response.get("messages"):
                        # Find the assistant's last message
                        for msg in reversed(response["messages"]):
                            role = msg.role if hasattr(msg, 'role') else msg._role if hasattr(msg, '_role') else None
                            if role == 'assistant':
                                found_response = True
                                content = msg.content if hasattr(msg, 'content') else (msg.text if hasattr(msg, 'text') else str(msg))
                                print(f"\n✅ Agent (resumed): {content}")
                                break
                    
                    # Check for agent-specific outputs
                    for key in response.keys():
                        if key not in ['messages', 'replies', 'calc_result']:
                            print(f"Additional output [{key}]: {response[key]}")
                    
                    if not found_response:
                        print(f"\n⚠️  No response found. Full output: {response}")
                    
                    # Print any state information if available
                    if response.get("calc_result"):
                        print(f"Calc Result: {response['calc_result']}")
                        
                except Exception as e:
                    print(f"Error resuming: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"Snapshot not found: {snapshot_path}")
            continue
            
        if not user_input:
            continue
        
        # Prepare breakpoint if enabled
        agent_bp = None
        if breakpoint_enabled:
            if breakpoint_type == "llm":
                # Breakpoint on chat generator (LLM calls)
                print(f"⏸️  Breakpoint active - will pause before LLM call")
                chat_bp = Breakpoint(
                    component_name="chat_generator",
                    visit_count=0,
                    snapshot_file_path=SNAPSHOT_DIR
                )
                agent_bp = AgentBreakpoint(
                    break_point=chat_bp,
                    agent_name="agent"
                )
            else:
                # Breakpoint on tool execution
                tool_msg = f"tool: {breakpoint_tool}" if breakpoint_tool else "any tool"
                print(f"⏸️  Breakpoint active - will pause before {tool_msg}")
                tool_bp = ToolBreakpoint(
                    component_name="tool_invoker",
                    visit_count=0,
                    tool_name=breakpoint_tool,  # None for any tool, or specific tool name
                    snapshot_file_path=SNAPSHOT_DIR
                )
                agent_bp = AgentBreakpoint(
                    break_point=tool_bp,
                    agent_name="agent"
                )
        
        # Run the agent with user input
        try:
            if snapshot_to_resume:
                # This shouldn't happen anymore since resume is handled above
                response = agent.run(messages=[], pipeline_snapshot=snapshot_to_resume)
                snapshot_to_resume = None
                print("✅ Resumed from snapshot")
            elif agent_bp:
                # Run with breakpoint
                response = agent.run(
                    messages=[ChatMessage.from_user(user_input)],
                    break_point=agent_bp
                )
            else:
                # Normal run
                response = agent.run(messages=[ChatMessage.from_user(user_input)])
        except BreakpointException as e:
            print(f"\n🔴 Breakpoint triggered!")
            print(f"Component: {e.component}")
            
            # Find the most recent snapshot file
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
                else:
                    print(f"Snapshot directory: {SNAPSHOT_DIR}")
            
            if hasattr(e, 'inputs') and e.inputs:
                print(f"Tool inputs: {e.inputs}")
            
            print("\n💡 Commands:")
            if latest_snapshot:
                print(f"  - Type 'resume {latest_snapshot}' to continue")
            else:
                print(f"  - Type 'resume <filename>' to continue")
            print("  - Type 'list' to see all snapshots")
            breakpoint_enabled = False
            continue
            if e.inputs:
                print(f"Inputs: {e.inputs}")
            print("\nUse 'list' to see snapshots, 'resume <file>' to continue")
            breakpoint_enabled = False
            continue
        
        # Extract and print the reply
        if response.get("messages"):
            last_message = response["messages"][-1]
            print(f"\nAgent: {last_message.text}")
        
        # Print any state information if available
        if response.get("calc_result"):
            print(f"Calc Result: {response['calc_result']}")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
        break
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
