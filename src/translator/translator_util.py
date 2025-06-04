import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM


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

    # print("✓ CPU environment optimized for AMD EPYC 9334")


def load_optimized_model(model_name: str):
    """Load model with working optimizations only"""

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
        compiled_model = torch.compile(
            model,
            mode="reduce-overhead",  # Best for inference
            backend="inductor",
            dynamic=False,
            fullgraph=False  # More compatible
        )
        # Test the compiled model
        # print("✓ PyTorch compilation successful")
        model = compiled_model
        model_type = "compiled"
    except Exception as e:
        # print(f"⚠ PyTorch compilation failed: {e}")
        print("Continuing with standard optimizations...")

    # Apply CPU-specific optimizations
    try:
        # Enable CPU optimizations
        torch.backends.cudnn.enabled = False  # Ensure CPU mode
        if hasattr(torch.backends, 'mkldnn'):
            torch.backends.mkldnn.enabled = True
        # print("✓ CPU backend optimizations enabled")
    except Exception as e:
        print(f"⚠ Backend optimization exception: {e}")

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