import transformers
print(transformers.__version__)

import os
from google import genai
from google.genai import types

# 1. Initialize Client (Put your key here or in env vars)
client = genai.Client(api_key="AIzaSyCO1iuF6k9sJ7egnSoBG6RSqGKnRlTA-_E") 

# 2. Use the CORRECT model string for the new SDK
# It prefers just "gemini-1.5-flash" but requires the right method call
response = client.models.generate_content(
    model="gemini-2.5-flash", 
    contents="Hello, are you working?"
)

print(response.text)