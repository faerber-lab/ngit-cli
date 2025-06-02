import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# ===================== PERFORMANCE OPTIMIZATIONS =====================

# 1. Set optimal thread count for your 64-core CPU
torch.set_num_threads(32)  # Often physical cores work best, adjust between 16-64
os.environ["OMP_NUM_THREADS"] = "32"
os.environ["MKL_NUM_THREADS"] = "32"

# 2. Enable optimized CPU kernels
torch.set_flush_denormal(True)

# 3. Set memory format for better cache performance
torch.backends.mkldnn.enabled = True

print("Loading model with optimizations...")
start_time = time.time()

# 4. Load with optimized settings
tokenizer = AutoTokenizer.from_pretrained("Salesforce/xLAM-2-1b-fc-r")
model = AutoModelForCausalLM.from_pretrained(
    "Salesforce/xLAM-2-1b-fc-r",
    torch_dtype=torch.float32,  # or torch.bfloat16 if supported
    device_map="cpu",
    low_cpu_mem_usage=True
)

# 5. Set model to evaluation mode
model = model.eval()

# 6. Enable CPU optimizations for transformers
if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
    torch.backends.mkldnn.enabled = True

# 7. Optimize memory layout
model = model.to(memory_format=torch.channels_last) if hasattr(model, 'to') else model

print(f"Model loading time: {time.time() - start_time:.2f}s")

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
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"],
                         "description": "The unit of temperature to return"}
            },
            "required": ["location"]
        }
    }
]

print("====== prompt after applying chat template ======")
prompt = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, tokenize=False)
print(prompt)

# Prepare inputs
inputs = tokenizer.apply_chat_template(messages, tools=tools, add_generation_prompt=True, return_dict=True,
                                       return_tensors="pt")
input_ids_len = inputs["input_ids"].shape[-1]

print("====== model response ======")
inference_start = time.time()

# 8. Optimized generation with CPU-specific settings
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=False,  # Greedy decoding is faster
        pad_token_id=tokenizer.eos_token_id,
        use_cache=True,  # Enable KV cache
        num_beams=1
    )

generated_tokens = outputs[:, input_ids_len:]
response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
print(response)

inference_time = time.time() - inference_start
total_time = time.time() - start_time

print(f"Inference time: {inference_time:.2f} seconds")
print(f"Total time: {total_time:.2f} seconds")
print(f"Tokens generated: {len(generated_tokens[0])}")
print(f"Tokens/second: {len(generated_tokens[0]) / inference_time:.2f}")
# [{"name": "get_weather", "arguments": {"location": "London"}}] .. 10 Second
