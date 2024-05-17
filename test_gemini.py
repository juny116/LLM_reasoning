from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai


GOOGLE_API_KEY = "AIzaSyCc-F_Bq_k3HFbAv3ZjA6xd5e-AeYyds3E"

llm = GoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
print(llm("What are some of the pros and cons of Python as a programming language?"))
