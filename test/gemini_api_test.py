import google.generativeai as genai

API_KEY = "AIzaSyC1ETJPKVmprlqFvQWx8-JSvWd8WS5HJII"

# Configure the API
genai.configure(api_key=API_KEY)

# Use Gemini 2.0 Flash (free tier model)
model = genai.GenerativeModel('gemini-2.0-flash-lite')

# Simple test prompt
test_prompt = """
You are a helpful assistant. Answer this question briefly:
What is the capital of France?
"""

try:
    # Generate response
    response = model.generate_content(test_prompt)
    
    print("✅ API Connection Successful!")
    print("\nResponse:")
    print(response.text)
    
except Exception as e:
    print("❌ Error:", str(e))
    print("\nTroubleshooting:")
    print("1. Check your API key is correct")
    print("2. Visit: https://makersuite.google.com/app/apikey")
    print("3. Ensure API is enabled in your Google Cloud project")
