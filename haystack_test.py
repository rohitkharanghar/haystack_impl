from haystack.components.generators.chat import OpenAIChatGenerator
from haystack.dataclasses import ChatMessage
from haystack.utils import Secret
import os

# Configure AWS Bedrock via Kong AI Gateway (OpenAI-compatible endpoint)
# Kong ai-proxy plugin should be configured with:
# - route_type: llm/v1/chat (OpenAI-compatible)
# - model.provider: bedrock
# - model.name: anthropic.claude-3-sonnet-20240229-v1:0
# - Optional: bedrock_guardrail_id and bedrock_guardrail_version for guardrails


KONG_AI_GATEWAY_URL = os.getenv("KONG_AI_GATEWAY_URL", "****")
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY", "*****")
# Use OpenAIChatGenerator with Kong's OpenAI-compatible API
# Kong translates OpenAI format to Bedrock format automatically
print("Using Kong AI Gateway at:", os.getenv("OPENAI_API_KEY"))

# Define streaming callback
def stream_callback(chunk):
    """Callback function to handle streaming tokens"""
    if chunk.content:
        print(chunk.content, end="", flush=True)

generator = OpenAIChatGenerator(
    api_base_url="https://9cb1c8dbab.aws-us-east-2.edge.gateways.konggateway.com/us-east-2/dr-who/sonnet-4",
    model="us.anthropic.claude-sonnet-4-20250514-v1:0",  # Bedrock model ID
    generation_kwargs={
        "max_tokens": 512,
        "temperature": 0.9,
    },
    streaming_callback=stream_callback,  # Enable streaming
    timeout=30,
    max_retries=3,
    # Add custom HTTP headers
    http_client_kwargs={
        "headers": {
            "apiKey": os.getenv("OPENAI_API_KEY", "****")
        }
    }
)

messages = [
    ChatMessage.from_system("\nYou are a helpful, respectful and honest assistant"),
    ChatMessage.from_user("What's Natural Language Processing?")
]

print("\n🤖 Assistant (streaming): ", end="", flush=True)
result = generator.run(messages=messages)
print("\n\n📊 Full result:")
print(result)