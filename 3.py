import ollama

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "The city name"}
                },
                "required": ["city"]
            }
        }
    }
]

response = ollama.chat(
    model="llama3.1",
    messages=[{"role": "user", "content": "What's the weather like in Paris?"}],
    tools=tools
)

print("Full response message:")
print(response["message"])
print()
print("Did it call a tool?", bool(response["message"].get("tool_calls")))