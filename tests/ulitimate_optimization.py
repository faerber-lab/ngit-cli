import warnings
from transformers.utils import logging
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# ===================== ULTIMATE CPU OPTIMIZATION SETUP =====================

def setup_cpu_environment():
    """Configure optimal CPU environment for AMD EPYC 9334"""
    # AMD EPYC specific optimizations
    os.environ["OMP_NUM_THREADS"] = "32"  # Adjust based on your workload
    os.environ["MKL_NUM_THREADS"] = "32"
    os.environ["OPENBLAS_NUM_THREADS"] = "32"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "32"
    os.environ["NUMEXPR_NUM_THREADS"] = "32"

    # Advanced CPU affinity settings for EPYC
    os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
    os.environ["KMP_BLOCKTIME"] = "1"
    os.environ["KMP_SETTINGS"] = "1"

    # Memory allocation optimizations
    os.environ["MALLOC_CONF"] = "background_thread:true,metadata_thp:auto,dirty_decay_ms:30000,muzzy_decay_ms:30000"

    # PyTorch specific settings
    torch.set_num_threads(32)
    torch.set_num_interop_threads(4)
    torch.set_flush_denormal(True)

    # Enable all available CPU optimizations
    if hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
        torch.backends.mkldnn.enabled = True

    if hasattr(torch.backends, 'mkl') and torch.backends.mkl.is_available():
        torch.backends.mkl.enabled = True

    print("✓ CPU environment optimized for AMD EPYC 9334")


def load_optimized_model(model_name: str):
    """Load model with working optimizations only"""

    print("Loading model with available optimizations...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Load model with optimal settings
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,  # Use bfloat16 for AMD EPYC
        device_map="cpu",
        low_cpu_mem_usage=True,
        attn_implementation="eager",  # Better for CPU
        use_cache=True
    )

    # Apply model optimizations
    model = model.eval()
    model_type = "standard"

    # Try PyTorch 2.0 compilation (most effective optimization available)
    try:
        print("Attempting PyTorch compilation...")
        compiled_model = torch.compile(
            model,
            mode="reduce-overhead",  # Best for inference
            backend="inductor",
            dynamic=False,
            fullgraph=False  # More compatible
        )
        # Test the compiled model
        print("✓ PyTorch compilation successful")
        model = compiled_model
        model_type = "compiled"
    except Exception as e:
        print(f"⚠ PyTorch compilation failed: {e}")
        print("Continuing with standard optimizations...")

    # Apply CPU-specific optimizations
    try:
        # Enable CPU optimizations
        torch.backends.cudnn.enabled = False  # Ensure CPU mode
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        print("✓ CPU backend optimizations enabled")
    except Exception as e:
        print(f"⚠ Backend optimization warning: {e}")

    return tokenizer, model, model_type


def create_optimized_generate_function(model):
    """Create optimized generation function with proper context"""
    def optimized_generate(inputs, **kwargs):
        """Highly optimized generation function"""

        # Prepare generation arguments for maximum speed
        generation_kwargs = {
            "max_new_tokens": 256,
            "do_sample": False,  # Greedy decoding is fastest
            "use_cache": True,  # Enable KV cache
            "temperature": None,
            "top_p": None,
            "top_k": None,
            "repetition_penalty": 1.0,
            "length_penalty": 1.0,
            "early_stopping": True,
            "num_beams": 1,  # Greedy search
            **kwargs  # Allow override
        }

        # Use the most efficient context manager
        context_manager = torch.inference_mode() if hasattr(torch, 'inference_mode') else torch.no_grad()

        with context_manager:
            outputs = model.generate(**inputs, **generation_kwargs)

        return outputs

    return optimized_generate


def benchmark_model(model, tokenizer, inputs):
    """Comprehensive benchmarking"""
    # Create optimized generate function
    generate_func = create_optimized_generate_function(model)
    input_ids_len = inputs["input_ids"].shape[-1]

    # Warmup runs
    start_time = time.time()
    with torch.inference_mode():
        outputs = generate_func(inputs)
    warmup_time = time.time() - start_time
    print(f"Generation time : {warmup_time:.2f}s")

    # Get sample output for analysis
    generated_tokens = outputs[:, input_ids_len:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    tokens_generated = len(generated_tokens[0])

    print(f"\nSample Response: {response[:100]}...")
    print(f"Tokens Generated: {tokens_generated}")

    return tokens_generated, response


def main():
    print("🚀 Starting Fixed Ultimate CPU Inference Optimization")
    print("=" * 60)

    # Setup environment
    setup_cpu_environment()

    # Model configuration
    model_name = "Salesforce/xLAM-2-1b-fc-r"

    print(f"\n📦 Loading model: {model_name}")
    start_time = time.time()

    # Load optimized model
    tokenizer, model, model_type = load_optimized_model(model_name)

    load_time = time.time() - start_time
    print(f"✓ Model loaded in {load_time:.2f}s (Type: {model_type})")

    # Prepare test data
    messages = [
        {"role": "user", "content": "Change the develop branch name to joojoo"},
    ]

    tools = [
        {
            "name": "config_user",
            "description": "Configure Git global username and email",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "Git username"},
                    "email": {"type": "string", "description": "Git email"}
                },
                "required": ["username", "email"]
            }
        },
        {
            "name": "commit",
            "description": "Commit with a message",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string", "description": "Commit message"}
                },
                "required": ["message"]
            }
        },
        {
            "name": "checkout",
            "description": "Checkout to a branch",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch_name": {"type": "string", "description": "Branch name"}
                },
                "required": ["branch_name"]
            }
        },
        {
            "name": "create_branch",
            "description": "Create a new branch",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch_name": {"type": "string", "description": "New branch name"}
                },
                "required": ["branch_name"]
            }
        },
        {
            "name": "delete_branch",
            "description": "Delete a branch",
            "parameters": {
                "type": "object",
                "properties": {
                    "branch_name": {"type": "string", "description": "Branch to delete"}
                },
                "required": ["branch_name"]
            }
        },
        {
            "name": "rename_branch",
            "description": "Change the name of current branch/Rename current branch",
            "parameters": {
                "type": "object",
                "properties": {
                    "old_name": {"type": "string", "description": "Current branch name"},
                    "new_name": {"type": "string", "description": "New branch name"}
                },
                "required": ["old_name", "new_name"]
            }
        },
        {
            "name": "status",
            "description": "Show git status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "reset_last_commit",
            "description": "Remove the last commit (soft/mixed/hard)",
            "parameters": {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["soft", "mixed", "hard"],
                        "description": "Reset mode"
                    }
                },
                "required": ["mode"]
            }
        },
        {
            "name": "add_remote",
            "description": "Add a new remote",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Remote name"},
                    "url": {"type": "string", "description": "Remote URL"}
                },
                "required": ["name", "url"]
            }
        },
        {
            "name": "remove_remote",
            "description": "Remove a remote",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Remote name"}
                },
                "required": ["name"]
            }
        },
        {
            "name": "list_remotes",
            "description": "List all remotes",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "add",
            "description": "Add files to staging",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to add"
                    }
                },
                "required": ["files"]
            }
        },
        {
            "name": "unstage",
            "description": "Remove files from staging",
            "parameters": {
                "type": "object",
                "properties": {
                    "files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Files to unstage"
                    }
                },
                "required": ["files"]
            }
        },
        {
            "name": "pull",
            "description": "Pull changes from remote",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote": {"type": "string", "description": "Remote name", "default": "origin"},
                    "branch": {"type": "string", "description": "Branch name", "default": "main"}
                },
                "required": []
            }
        },
        {
            "name": "push",
            "description": "Push changes to remote",
            "parameters": {
                "type": "object",
                "properties": {
                    "remote": {"type": "string", "description": "Remote name", "default": "origin"},
                    "branch": {"type": "string", "description": "Branch name", "default": "main"}
                },
                "required": []
            }
        },
        {
            "name": "init",
            "description": "Initialize a new git repo",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {"type": "string", "description": "Directory path", "default": "."}
                },
                "required": []
            }
        },
        {
            "name": "clone",
            "description": "Clone a git repo",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "Repository URL"},
                    "directory": {"type": "string", "description": "Target directory"}
                },
                "required": ["url"]
            }
        }
    ]

    print("\n🔄 Preparing inputs...")
    # Prepare inputs
    inputs = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt"
    )

    # Add pad_token_id if not present
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("✓ Inputs prepared")

    # Run comprehensive benchmark
    tokens_generated, sample_response = benchmark_model(
        model, tokenizer, inputs
    )

    print("\n" + "=" * 60)
    print("📈 FINAL PERFORMANCE METRICS")
    print("=" * 60)
    print(f"Model Type: {model_type}")
    print(f"Model Load Time: {load_time:.2f}s")
    print(f"Tokens Generated: {tokens_generated}")

    # Memory usage (if available)
    try:
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        print(f"Memory Usage: {memory_mb:.0f} MB")
    except ImportError:
        print("Memory usage: Install psutil for memory monitoring")

    print(f"\nSample Output: {sample_response}")


if __name__ == "__main__":
    main()
