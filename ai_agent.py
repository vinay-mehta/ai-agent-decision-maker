import requests
from transformers import pipeline, GPT2Tokenizer, GPT2LMHeadModel

# Load pre-trained GPT-2 model and tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Initialize the text generation pipeline
text_generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Define keywords for identifying math-related queries
math_keywords = ["calculate", "sum", "subtract", "multiply", "divide", "math", "equation", "solve"]

def is_math_query(query):
    return any(keyword in query.lower() for keyword in math_keywords)

SERPAPI_API_KEY = "your_serpapi_api_key"

def search_serpapi(query):
    params = {
        "q": query,
        "api_key": SERPAPI_API_KEY,
        "engine": "google"
    }
    response = requests.get("https://serpapi.com/search", params=params)
    return response.json()

def perform_math(query):
    prompt = f"Solve the following math problem: {query}"
    results = text_generator(prompt, max_length=50)
    return results[0]["generated_text"]

def ai_agent(query):
    if is_math_query(query):
        result = perform_math(query)
    else:
        result = search_serpapi(query)
    return result

# Example queries
query1 = "What is the capital of France?"
query2 = "Calculate the sum of 234 and 456"

print("Query 1 Result:", ai_agent(query1))
print("Query 2 Result:", ai_agent(query2))
