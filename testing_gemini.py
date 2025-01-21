import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()

genai.configure(api_key=os.getenv("Gemini_API_Key"))

model = genai.GenerativeModel("gemini-1.5-flash")
response = model.generate_content("can you follow the same instructions while users are inputing varing requests/prompts ")
print(response.text)