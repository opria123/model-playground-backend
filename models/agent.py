from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.graph import START, StateGraph
from langgraph.prebuilt import tools_condition
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from models.reasoning_model import ReasoningModel
from tools.guest_info_retriever import guest_info_tool 
from tools.search_tool import search_tool 
from tools.weather_tool import weather_info_tool
from tools.hugging_face_hub_stats_tool import hub_stats_tool 
from smolagents import CodeAgent, LiteLLMModel
import logging
import os

logger = logging.getLogger(__name__)

# Retrieve the API base URL from environment variables
api_base_url = os.environ.get("API_BASE", "http://127.0.0.1:11435")  # Default to localhost if not set

model = LiteLLMModel(
    model_id="ollama_chat/qwen2.5:7b",  # Can try different model here I am using qwen2.5 7B model
    api_base=api_base_url,  # Use the environment variable for the API base
    num_ctx=8192,
)

alfred = CodeAgent(
    tools=[guest_info_tool, weather_info_tool, hub_stats_tool], 
    model=model, 
    add_base_tools=False
)

def invoke_agent(message: str) -> dict:
    human_message = {"role": "user", "content": message}
    messages = [human_message]

    response = alfred.run(message)
    logger.info("Raw response from agent: %s", response)

    return {
        response: response
    }
