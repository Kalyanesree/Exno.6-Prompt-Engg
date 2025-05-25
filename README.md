# Exno.6-Prompt-Engg
# Date:14/05/2025
# Register no. 212222050028         # Name: KALYANE SREE M
# Aim: Development of Python Code Compatible with Multiple AI Tools


# Algorithm: Write and implement Python code that integrates with multiple AI tools to automate the task of interacting with APIs, comparing outputs, and generating actionable insights.
This Python script integrates with multiple Al tools to automate API interactions, compare outputs, and generate insights. Below is a comprehensive implementation with four powerful Al tools: Tools Integrated:
1.	OpenAI GPT-4 - For natural language processing and insight generation
2.	Hugging Face Transformers - For alternative NLP processing and comparison
3.	Google Gemini API - For multimodal analysis capabilities
4.	LangChain - For orchestration and workflow management

# PYTHON CODE:

import os import requests importjson from typing import List, Diet, Any import pandas as pd from langchain.chains import LLMChain from langchain.prompts import PromptTemplate from langchain.llms import OpenAI from langchain_community.llms import HuggingFaceHub import google.generativeai as genai from dotenv import load_dotenv import matplotlib.pyplot as pit from concurrent.futures import ThreadPoolExecutor

Load environment variables

load_dotenv()

class APIAnalyzer: def init(self): # Initialize all Al tools with API keys self.openai_key = os.getenv("OPENAI_API_KEY") self.huggingface_key = os.getenv("HUGGINGFACE_API_KEY") self.google_key = os.getenv("GOOGLE_API_KEY")

```# Configure Google Gemini	c9
genai.configure(api_key=self.google_key) self.gemini = genai.GenerativeModel('gemini-pro')

# Configure LangChain with OpenAI self.llm_openai = OpenAI(temperature=0.7,
openai_api_key=self.openai_key)

# Configure Hugging Face self.hf_llm = HuggingFaceHub(
repo_id="google/flan-tS-xxl", huggingfacehub_api_token=self.huggingface_key


# Initialize prompt templates self.init_prompts()```
 

def init_prompts(self):
"""Initialize prompt templates for different tasks""" self.insight_prompt = PromptTemplate( input_variables=["api_responses", "user_query"], template="""
Analyze the following API responses and provide actionable insights based on the user's	query: {user_query}

API Responses:
{api_responses}

# Provide:
1.	Key similarities and differences
2.	Potential issues or anomalies
3.	Recommendations for action
4.	Confidence score (1-10) in your analysis



self.comparison_prompt = PromptTemplate( input_variables=["responses"], template="""
Compare these API responses and highlight:
1.	Data consistency across responses
2.	Response time differences
3.	Data completeness
4.	Any contradictory information

Responses:
{responses}



```def call_api(self, url: str, params: Diet= None, headers: Diet= None) ->
Diet:
"""Generic API call function""" try:
response= requests.get(url, params=params, headers=headers) response.raise_for_status()
return response.json()
except requests.exceptions.RequestException as e: return {"error": str(e)}

def analyze_with_openai(self, prompt: str) -> str: """Analyze text using OpenAI GPT-4"""
chain= LLMChain(llm=self.llm_openai, prompt=PromptTemplate.from_template("{input}"))
return chain.run(input=prompt)
 

def analyze_with_huggingface(self, prompt: str) -> str: '"'"Analyze text using Hugging Face model"""
return self.hf_llm(prompt)

def analyze_with_gemini(self, prompt: str) -> str: "'"'Analyze text using Google Gemini""" response= self.gemini.generate_content(prompt) return response.text

def compare_responses(self, responses: List[Dict], method: str = "openai")
-> Diet:
"""Compare multiple API responses using specified AI tool""" comparison_text = "\n\n".join([json.dumps(res, indent=2) for res in
responses])

if method== "openai":
analysis= self.analyze_with_openai(
f"Compare these API responses:\n{comparison_text}"
)
elif method== "huggingface":
analysis= self.analyze_with_huggingface(
f"Compare these API responses:\n{comparison_text}"
)
elif method== "gemini":
analysis= self.analyze_with_gemini(
f"Compare these API responses:\n{comparison_text}"
)
else:
analysis	"Invalid analysis method specified"

return {
"method": method, "analysis": analysis, "responses": responses
}

def generate_insights(self, api_data: Any, user_query: str) -> Diet: """Generate insights from API data using multiple AI tools"""
# Convert data to string if not already
if not isinstance(api_data, str):
api_data = json.dumps(api_data, indent=2)

# Prepare results dictionary insights={}

# Analyze with all tools in parallel with ThreadPoolExecutor() as executor:
futures= {
 
"openai": executor.submit( self.analyze_with_openai, self.insight_prompt.format(
api_responses=api_data, user_query=user_query
)
) ,
"huggingface": executor.submit( self.analyze_with_huggingface, self.insight_prompt.format(
api_responses=api_data, user_query=user_query
)
) ,
"gemini": executor.submit( self.analyze_with_gemini, self.insight_prompt.format(
api_responses=api_data, user_query=user_query
)

}

for tool, future in futures.items(): insights[tool] = future.result()

return insights

def visualize_comparison(self, comparisons: Diet) -> None: """Create visualizations from comparison data"""
# This is a simple example - you would expand based on your specific
data
tools= list(comparisons.keys())
analysis_lengths = [len(comparisons[tool]['analysis']) for tool in
tools]

plt.figure(figsize=(10, 5)) plt.bar(tools, analysis_lengths)
plt.title("Analysis Output Length by AI Tool") plt.ylabel("Character Count")
plt.xlabel("AI Tool") plt.show()


Example Usage

if name = = "main": analyzer= APIAnalyzer()
 
# Example API calls (replace with your actual APis)	c9
apil_response = analyzer.call_api("https://api.example.com/datal") api2_response = analyzer.call_api("https://api.example.com/data2")

# Compare responses using different AI tools comparison_results = {
"openai": analyzer.compare_responses([apil_response, api2_response], "openai"),
"huggingface": analyzer.compare_responses([apil_response, api2_response], "huggingface"),
"gemini": analyzer.compare_responses([apil_response, api2_response], "gemini")
}

# Generate insights from the first API response insights= analyzer.generate_insights(
apil_response,
"What trends can you identify in this data?"
)

# Print results print("\nComparison Results:")
for tool, result in comparison_results.items(): print(f"\n{tool.upper()} Analysis:") print(result['analysis'])

print("\ninsights:")
for tool, insight in insights.items(): print(f"\n{tool.upper()} Insights:") print(insight)

# Visualize comparison analyzer.visualize_comparison(comparison_results) ```


# Detailed Explanation of the Four Al Tools:

1.	OpenAI GPT-4 (via LangChain)

o	Purpose: Primary natural language processing for analysis and insight generation o Strengths: Most advanced reasoning capabilities, best for complex analysis o Implementation: Used through LangChain for prompt engineering and chaining
 
2.	Hugging Face Transformers (FLAN-TS­ XXL)
o	Purpose: Alternative NLP processing for comparison and validation o Strengths: Open­ source, good for factual comparison tasks o Implementation: Using their inference API with the FLAN-TS-xx! model

3.	Google Gemini API
o	Purpose: Multimodal analysis and alternative perspective o Strengths: Strong at data interpretation and synthesis o Implementation: Direct integration with Google's SDK

4.	LangChain Framework

o Purpose: Orchestration and workflow management o Strengths: Simplifies complex chains of operations with multiple Al tools o Implementation: Used to manage prompts and chains between different components

#Result:
The corresponding Prompt is executed successfully
 




















# Result: The corresponding Prompt is executed successfully
