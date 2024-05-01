from typing import List, Dict, Any

import google.generativeai as genai
import google.generativeai.types as gtypes
from PIL import Image


class GenerativeAI:
    def __init__(self, api_key: str) -> None:
        genai.configure(api_key=api_key)

    def list_models(self) -> List[gtypes.Model]:
        return genai.list_models()

    def generate_text(self, model_name: str, prompt: str) -> str:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        return response.text

    def generate_text_from_image(
        self, model_name: str, prompt: str, image_path: str
    ) -> str:
        img = Image.open(image_path)
        model = genai.GenerativeModel(model_name)
        response = model.generate_content([prompt, img], stream=True)
        response.resolve()
        return response.text

    def start_chat(self, model_name: str, history: List[str] = []) -> genai.chat:
        model = genai.GenerativeModel(model_name)
        chat = model.start_chat(history=history)
        return chat

    def count_tokens(self, model_name: str, text: str) -> int:
        model = genai.GenerativeModel(model_name)
        return model.count_tokens(text)

    def embed_content(
        self, model_name: str, content: str, task_type: str, title: str
    ) -> Dict[str, Any]:
        return genai.embed_content(
            model=model_name, content=content, task_type=task_type, title=title
        )

    def generate_content_advanced(
        self, model_name: str, prompt: str, safety_settings: Dict[str, str] = None
    ) -> str:
        model = genai.GenerativeModel(model_name)
        if safety_settings:
            response = model.generate_content(prompt, safety_settings=safety_settings)
        else:
            response = model.generate_content(prompt)
        return response.text

    def generate_content_multi_turn(
        self, model_name: str, messages: List[Dict[str, List[str]]]
    ) -> str:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(messages)
        return response.text

    def generate_content_with_config(
        self,
        model_name: str,
        prompt: str,
        generation_config: genai.types.GenerationConfig,
    ) -> str:
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt, generation_config=generation_config)
        if response.candidates:
            print("Safety ratings:", response.candidates[0].safety_ratings)
            if hasattr(response.candidates[0], "parts"):
                text = (
                    response.candidates[0].parts[0].text
                    if response.candidates[0].parts
                    else "No valid part in response"
                )
                if response.candidates[0].finish_reason.name == "MAX_TOKENS":
                    text += "..."
                print(text)
            else:
                print("No 'parts' attribute in Candidate")
        else:
            print("No candidates in response")
        return response.text
def main() -> None:
    api_key = "AIzaSyDF5zuBmIBLZaUFEmuu2ajLJrFcBaSkFF4"  # Add your API key here
    generative_ai = GenerativeAI(api_key)

    # List available models
    for m in generative_ai.list_models():
        if "generateContent" in m.supported_generation_methods:
            print(m.name)

    # Generate text from text inputs
    text_response = generative_ai.generate_text(
        "gemini-pro", "Explain table tennis in one sentence."
    )
    print(text_response)

    # Generate text from image and text inputs
    img_path = "image.jpg"
    image_response = generative_ai.generate_text_from_image(
        "gemini-pro-vision",
        "Write a short, engaging blog post based on this picture. It should include a description of the meal in the photo and talk about my journey meal prepping.",
        img_path,
    )
    print(image_response)

    # Chat conversations
    chat = generative_ai.start_chat("gemini-pro")
    chat_response = chat.send_message(
        "In one sentence, explain how a computer works to a young child."
    )
    print(chat_response.text)

    # Count tokens
    token_count = generative_ai.count_tokens(
        "gemini-pro", "What is the meaning of life?"
    )
    print(token_count)

    # Use embeddings
    embedding_result = generative_ai.embed_content(
        model_name="models/embedding-001",
        content="What is the meaning of life?",
        task_type="retrieval_document",
        title="Embedding of single string",
    )
    print(embedding_result["embedding"])

    # Advanced use cases
    advanced_response = generative_ai.generate_content_advanced(
        "gemini-pro", "[Questionable prompt here]"
    )
    print(advanced_response)

    # Multi-turn conversations
    messages = [
        {
            "role": "user",
            "parts": ["Briefly explain how a computer works to a young child."],
        }
    ]
    multi_turn_response = generative_ai.generate_content_multi_turn(
        "gemini-pro", messages
    )
    print(multi_turn_response)

    # Generation configuration
    generation_config = genai.types.GenerationConfig(
        candidate_count=1, stop_sequences=["x"], max_output_tokens=20, temperature=1.0
    )
    config_response = generative_ai.generate_content_with_config(
        "gemini-pro",
        "Explain the meaning of life in one sentence. But for varying levels of understanding and complexity.",
        generation_config,
    )
    print(config_response)


if __name__ == "__main__":
    main()
