# mcp_server_protocol.py
from fastapi import FastAPI, Request
import uuid

app = FastAPI(title="Local MCP Server (Protocol)")

# Tool registry
def add_numbers(a: float, b: float) -> float:
    return a + b

def multiply_numbers(a: float, b: float) -> float:
    return a * b

TOOLS = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers
}

@app.post("/mcp")
async def mcp_endpoint(req: Request):
    """
    Accepts MCP protocol requests in JSON:
    {
        "id": "uuid",
        "method": "tool_name",
        "params": {...}
    }
    """
    payload = await req.json()
    
    req_id = payload.get("id", str(uuid.uuid4()))
    method = payload.get("method")
    params = payload.get("params", {})

    if method not in TOOLS:
        return {"id": req_id, "result": None, "error": f"Tool '{method}' not found"}

    try:
        result = TOOLS[method](**params)
        return {"id": req_id, "result": result, "error": None}
    except Exception as e:
        return {"id": req_id, "result": None, "error": str(e)}
