import ollama

response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": "Say hello in 3 words"}]
)

print(response["message"]["content"])