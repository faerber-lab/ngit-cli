import torch
# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

start_time = time.time()

tokenizer = AutoTokenizer.from_pretrained("Salesforce/xLAM-2-1b-fc-r")
model = AutoModelForCausalLM.from_pretrained("Salesforce/xLAM-2-1b-fc-r")

# Example conversation with a tool call
messages = [
    {"role": "user", "content": "Hi, how are you?"},
    {"role": "assistant", "content": "Thanks. I am doing well. How can I help you?"},
    {"role": "user", "content": "What's the weather like in London?"},
]

tools = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "The city and state, e.g. San Francisco, CA"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    }
]

print("====== prompt after applying chat template ======")
print(tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False))

inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True, return_tensors="pt")
input_ids_len = inputs["input_ids"].shape[-1] # Get the length of the input tokens
inputs = {k: v.to(model.device) for k, v in inputs.items()}
print("====== model response ======")
outputs = model.generate(**inputs, max_new_tokens=256)
generated_tokens = outputs[:, input_ids_len:] # Slice the output to get only the newly generated tokens
print(tokenizer.decode(generated_tokens[0], skip_special_tokens=True))
end_time = time.time()  # record end
duration = end_time - start_time  # duration in seconds
print(f"Duration: {duration:.2f} seconds")
# [{"name": "get_weather", "arguments": {"location": "London"}}] .. 50.62 S