import time
import torch
from translator.translator_util import setup_cpu_environment, load_optimized_model, create_optimized_generate_function


def _get_translation(model, tokenizer, inputs):
    """Comprehensive benchmarking"""
    # Create optimized generate function
    generate_func = create_optimized_generate_function(model)
    input_ids_len = inputs["input_ids"].shape[-1]

    # Warmup runs
    start_time = time.time()
    with torch.inference_mode():
        outputs = generate_func(inputs)
    generation_time = time.time() - start_time

    # Get sample output for analysis
    generated_tokens = outputs[:, input_ids_len:]
    response = tokenizer.decode(generated_tokens[0], skip_special_tokens=True)
    tokens_generated = len(generated_tokens[0])

    return tokens_generated, response, generation_time


def translate(message="Change the develop branch name to joojoo"):
    # Setup environment
    model_name = "Salesforce/xLAM-2-1b-fc-r"
    setup_cpu_environment()

    # Load optimized model
    tokenizer, model, model_type = load_optimized_model(model_name)
    messages = [
        {"role": "user", "content": message},
    ]
    tools = [
        {
            "name": "git_config_username",
            "description": "Configure Git global username",
            "parameters": {
                "type": "object",
                "properties": {
                    "username": {"type": "string", "description": "Git username"},
                },
                "required": ["username"]
            }
        },
        {
            "name": "git_config_email",
            "description": "Configure Git global email",
            "parameters": {
                "type": "object",
                "properties": {
                    "email": {"type": "string", "description": "Git email"}
                },
                "required": ["email"]
            }
        },
        {
            "name": "git_commit",
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
            "name": "git_checkout",
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
            "name": "git_create_branch",
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
            "name": "git_delete_branch",
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
            "name": "git_rename_branch",
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
            "name": "git_status",
            "description": "Show git status",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "git_reset_last_commit",
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
            "name": "git_add_remote",
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
            "name": "git_remove_remote",
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
            "name": "git_list_remotes",
            "description": "List all remotes",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        },
        {
            "name": "git_add",
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
            "name": "git_unstage",
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
            "name": "git_pull",
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
            "name": "git_push",
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
            "name": "git_init",
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
            "name": "git_clone",
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

    # Run translation
    tokens_generated, sample_response, generation_time = _get_translation(
        model, tokenizer, inputs
    )
    return sample_response
