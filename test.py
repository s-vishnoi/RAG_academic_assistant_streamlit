from langchain_ollama import OllamaLLM

llm = OllamaLLM(model="deepseek-r1")
response = llm.invoke("What is 2 + 2?")
print(response)

#runs! (model is loading)