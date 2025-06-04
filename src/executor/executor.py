import json
from click.testing import CliRunner
from commands.git import (git_config_username, git_config_email, git_commit,
                              git_checkout, git_create_branch, git_delete_branch,
                              git_rename_branch, git_status, git_reset_last_commit,
                              git_add_remote, git_remove_remote, git_list_remotes,
                              git_add, git_unstage, git_pull, git_push, git_init,
                              git_clone)


def execute_function(command: str):
    function_map = {
        "git_config_username": git_config_username,
        "git_config_email": git_config_email,
        "git_commit": git_commit,
        "git_checkout": git_checkout,
        "git_create_branch": git_create_branch,
        "git_delete_branch": git_delete_branch,
        "git_rename_branch": git_rename_branch,
        "git_status": git_status,
        "git_reset_last_commit": git_reset_last_commit,
        "git_add_remote": git_add_remote,
        "git_remove_remote": git_remove_remote,
        "git_list_remotes": git_list_remotes,
        "git_add": git_add,
        "git_unstage": git_unstage,
        "git_pull": git_pull,
        "git_push": git_push,
        "git_init": git_init,
        "git_clone": git_clone
    }

    data = json.loads(command)
    for cmd in data:
        func_name = cmd['name']
        args = cmd['arguments']
        func = function_map.get(func_name)
        runner = CliRunner()
        result = runner.invoke(func, list(args.values()))


def get_real_command(command: str) -> str:
    COMMAND_MAP = {
        "git_config_username": lambda username: ["git", "config", "--global", "user.name", username],
        "git_config_email": lambda email: ["git", "config", "--global", "user.email", email],
        "git_commit": lambda message: ["git", "commit", "-m", message],
        "git_checkout": lambda branch_name: ["git", "checkout", branch_name],
        "git_create_branch": lambda branch_name: ["git", "branch", branch_name],
        "git_delete_branch": lambda branch_name: ["git", "branch", "-d", branch_name],
        "git_rename_branch": lambda old_name, new_name: ["git", "branch", "-m", old_name, new_name],
        "git_status": lambda: ["git", "status"],
        "git_reset_last_commit": lambda mode="soft": ["git", "reset", f"--{mode}", "HEAD~1"],
        "git_add_remote": lambda name, url: ["git", "remote", "add", name, url],
        "git_remove_remote": lambda name: ["git", "remote", "remove", name],
        "git_list_remotes": lambda: ["git", "remote", "-v"],
        "git_add": lambda files: ["git", "add"] + files,
        "git_unstage": lambda files: ["git", "reset"] + files,
        "git_pull": lambda remote="origin", branch=None: ["git", "pull", remote] + ([branch] if branch else []),
        "git_push": lambda remote="origin", branch=None: ["git", "push", remote] + ([branch] if branch else []),
        "git_init": lambda directory=".": ["git", "init", directory],
        "git_clone": lambda url, directory=None: ["git", "clone", url] + ([directory] if directory else []),
    }
    data = json.loads(command)
    if not data:
        return ""
    cmd0 = data[0]
    func_name = cmd0['name']
    args = cmd0['arguments']
    func = COMMAND_MAP.get(func_name)
    if not func:
        return ""
    tokens = func(**args)
    quoted_tokens = []
    for token in tokens:
        token_str = str(token)
        if ' ' in token_str or '"' in token_str:
            token_escaped = token_str.replace('"', '\\"')
            quoted_tokens.append(f'"{token_escaped}"')
        else:
            quoted_tokens.append(token_str)
    return " ".join(quoted_tokens)
