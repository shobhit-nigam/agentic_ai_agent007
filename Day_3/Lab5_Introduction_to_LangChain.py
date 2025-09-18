# Lab 5: Introduction to LangChain
# --------------------------------
# Objective:
# Build a simple LangChain program using LLM, PromptTemplate, and LLMChain.

# Step 1: Import required modules
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Step 2: Define the LLM (the "brain")
# Replace 'gpt-3.5-turbo' with the model you want to use
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

# Step 3: Create a Prompt Template
# A template is a structured way of writing prompts with placeholders
template = """You are an AI assistant.
Explain the concept of Agentic AI in {num_sentences} simple sentences."""

prompt = PromptTemplate(
    input_variables=["num_sentences"],
    template=template
)

# Step 4: Build the Chain
# LLMChain connects: PromptTemplate -> LLM -> Output
chain = LLMChain(llm=llm, prompt=prompt)

# Step 5: Run the Chain
# Provide a value for 'num_sentences'
result = chain.run(num_sentences=3)

# Display the result
print("=== Agentic AI Explanation ===")
print(result)
