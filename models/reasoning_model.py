from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch
import logging
from typing import Dict, Any, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_core.tools import BaseTool

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ReasoningModel:
    def __init__(self, model_name="opria123/SmolGRPO-135M"):
        """
        Initialize the reasoning model with a pretrained model from Hugging Face.
        
        Args:
            model_name (str): The name or path of the pretrained model
        """
        logger.info(f"Loading model {model_name}...")
        
        try:
            # Load base model and tokenizer first
            logger.info("Loading base model and tokenizer...")
            base_model_id = "HuggingFaceTB/SmolLM-135M-Instruct"
            self.tokenizer = AutoTokenizer.from_pretrained(base_model_id)
            logger.info("Tokenizer loaded successfully")
            
            # Load model
            logger.info("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            logger.info("Model loaded successfully")
            
            # Initialize the generator pipeline
            logger.info("Initializing generator pipeline...")
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                trust_remote_code=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            logger.info("Generator pipeline initialized successfully")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def bind_tools(self, tools: List[BaseTool]):
        """
        Bind tools to the model for use in the agent.
        
        Args:
            tools (List[BaseTool]): List of tools to bind
            
        Returns:
            ReasoningModel: The model instance with tools bound
        """
        self.tools = tools
        return self

    def invoke(self, messages: List[BaseMessage]) -> AIMessage:
        """
        Process a list of messages and generate a response.
        
        Args:
            messages (List[BaseMessage]): List of messages to process
            
        Returns:
            AIMessage: The generated response
        """
        try:
            # Convert messages to a prompt
            prompt = self._format_messages(messages)
            
            # Generate response
            generated_text = self.generator(
                prompt,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            raw_response = generated_text[0]['generated_text']
            
            # Log the raw response for debugging
            logger.info("\n=== RAW MODEL RESPONSE ===")
            logger.info(raw_response)
            logger.info("=== END RAW RESPONSE ===\n")
            
            # Parse the response
            parsed_response = self.parse_response(raw_response, prompt)
            
            # Return as AIMessage
            return AIMessage(content=parsed_response["answer"])
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def _format_messages(self, messages: List[BaseMessage]) -> str:
        """
        Format a list of messages into a prompt string.
        
        Args:
            messages (List[BaseMessage]): List of messages to format
            
        Returns:
            str: Formatted prompt string
        """
        formatted_messages = []
        for message in messages:
            if isinstance(message, HumanMessage):
                formatted_messages.append(f"user: {message.content}")
            elif isinstance(message, AIMessage):
                formatted_messages.append(f"assistant: {message.content}")
        
        return "\n".join(formatted_messages)

    def generate_response(self, prompt: str, role: str = "assistant", max_new_tokens: int = 256) -> Dict[str, Any]:
        """
        Generate a response using the model.
        
        Args:
            prompt (str): The input prompt
            role (str): The role of the model (e.g., "assistant", "user")
            max_new_tokens (int): Maximum number of new tokens to generate
            
        Returns:
            Dict[str, Any]: Dictionary containing the parsed response
        """
        try:
            # Format the prompt with role
            formatted_prompt = f"{role}: {prompt}\n"
            logger.info(f"Generating response for prompt: {formatted_prompt}")
            
            generated_text = self.generator(
                formatted_prompt,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            raw_response = generated_text[0]['generated_text']
            
            # Log the raw response for debugging
            logger.info("\n=== RAW MODEL RESPONSE ===")
            logger.info(raw_response)
            logger.info("=== END RAW RESPONSE ===\n")
            
            # Parse the response
            parsed_response = self.parse_response(raw_response, prompt)
            
            # Log the parsed response for debugging
            logger.info("\n=== PARSED RESPONSE ===")
            logger.info(parsed_response)
            logger.info("=== END PARSED RESPONSE ===\n")
            
            return parsed_response
            
        except Exception as e:
            logger.error(f"Error during generation: {str(e)}")
            raise

    def parse_response(self, response: str, original_prompt: str) -> Dict[str, Any]:
        """
        Parse the model's response into role, question, and answer.
        
        Args:
            response (str): The raw response from the model
            original_prompt (str): The original prompt/question
            
        Returns:
            Dict[str, Any]: Dictionary containing role, question, and answer
        """
        try:
            # Split on the first colon to separate role and content
            role_part, content = response.split(":", 1)
            role = role_part.strip()
            
            # Clean up the content
            content = content.strip()
            
            # Remove the repeated question if it exists at the beginning
            if content.startswith(original_prompt):
                content = content[len(original_prompt):].strip()
            
            # If the content starts with "Answer:", remove it
            if content.startswith("Answer:"):
                content = content[7:].strip()
            
            return {
                "role": {
                    "role_name": role,
                    "question": original_prompt
                },
                "answer": content
            }
        except Exception as e:
            logger.error(f"Error parsing response: {str(e)}")
            # Return the original response if parsing fails
            return {
                "role": {
                    "role_name": "assistant",
                    "question": original_prompt
                },
                "answer": response
            } 