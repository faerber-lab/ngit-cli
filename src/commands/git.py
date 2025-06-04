import os
import subprocess
import click


# Utility function for running and printing commands
def _run_git_command(command: list, capture_output=True):
    result = subprocess.run(command, capture_output=True, text=True)
    print(result)
    if capture_output:
        output = result.stdout.strip() or result.stderr.strip()
        click.echo(result.stdout)
        click.echo(result.stderr)
        return output
    return None


@click.group(invoke_without_command=True)
@click.pass_context
def git(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('I was invoked without subcommand')
    else:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand}")


@git.command("config_username")
@click.argument("username")
def git_config_username(username):
    """Configure Git global username"""
    _run_git_command(["git", "config", "--global", "user.name", username], capture_output=True)
    click.echo(f"✅ Git user configured as {username}")


@git.command("config_email")
@click.argument("email")
def git_config_email(email):
    """Configure Git global email"""
    _run_git_command(["git", "config", "--global", "user.email", email], capture_output=True)
    click.echo(f"✅ Git user email configured as {email}")


@git.command("commit")
@click.argument("message")
def git_commit(message):
    """Commit with a message"""
    click.echo("🔧 Committing changes...")
    _run_git_command(["git", "commit", "-m", message])


@git.command("checkout")
@click.argument("branch_name")
def git_checkout(branch_name):
    """Checkout to a branch"""
    click.echo(f"🔀 Switching to branch '{branch_name}'...")
    _run_git_command(["git", "checkout", branch_name])


@git.command("create_branch")
@click.argument("branch_name")
def git_create_branch(branch_name):
    """Create a new branch"""
    _run_git_command(["git", "branch", branch_name], capture_output=True)
    click.echo(f"🌿 Branch '{branch_name}' created")


@git.command("delete_branch")
@click.argument("branch_name")
def git_delete_branch(branch_name):
    """Delete a branch"""
    _run_git_command(["git", "branch", "-d", branch_name], capture_output=True)
    click.echo(f"❌ Branch '{branch_name}' deleted")


@git.command("rename_branch")
@click.argument("old_name")
@click.argument("new_name")
def git_rename_branch(old_name, new_name):
    """Rename a branch"""
    _run_git_command(["git", "branch", "-m", old_name, new_name], capture_output=True)
    click.echo(f"🔁 Branch renamed from '{old_name}' to '{new_name}'")


@git.command("status")
def git_status():
    """Show git status"""
    click.echo("📦 Git status:")
    _run_git_command(["git", "status"])


@git.command("reset_last_commit")
@click.option("--mode", default="soft", type=click.Choice(["soft", "mixed", "hard"]), help="Reset mode")
def git_reset_last_commit(mode):
    """Remove the last commit (soft/mixed/hard)"""
    _run_git_command(["git", "reset", f"--{mode}", "HEAD~1"], capture_output=True)
    click.echo(f"⚠️ Last commit removed with {mode} reset")


@git.command("add_remote")
@click.argument("name")
@click.argument("url")
def git_add_remote(name, url):
    """Add a new remote"""
    _run_git_command(["git", "remote", "add", name, url], capture_output=True)
    click.echo(f"🔗 Remote '{name}' added with URL {url}")


@git.command("remove_remote")
@click.argument("name")
def git_remove_remote(name):
    """Remove a remote"""
    _run_git_command(["git", "remote", "remove", name], capture_output=True)
    click.echo(f"🔌 Remote '{name}' removed")


@git.command("list_remotes")
def git_list_remotes():
    """List all remotes"""
    click.echo("🌐 Git remotes:")
    _run_git_command(["git", "remote", "-v"])


@git.command("add")
@click.argument("files", nargs=-1)
def git_add(files):
    """Add files to staging"""
    _run_git_command(["git", "add"] + list(files), capture_output=True)
    click.echo(f"📥 Staged: {', '.join(files)}")


@git.command("unstage")
@click.argument("files", nargs=-1)
def git_unstage(files):
    """Remove files from staging"""
    _run_git_command(["git", "reset"] + list(files), capture_output=True)
    click.echo(f"📤 Unstaged: {', '.join(files)}")


@git.command("pull")
@click.option("--remote", default="origin", help="Remote name")
@click.option("--branch", default="main", help="Branch name")
def git_pull(remote, branch):
    """Pull changes from remote"""
    click.echo(f"⬇️ Pulling from {remote}/{branch}...")
    _run_git_command(["git", "pull", remote, branch])


@git.command("push")
@click.option("--remote", default="origin", help="Remote name")
@click.option("--branch", default="main", help="Branch name")
def git_push(remote, branch):
    """Push changes to remote"""
    click.echo(f"⬆️ Pushing to {remote}/{branch}...")
    _run_git_command(["git", "push", remote, branch])


@git.command("init")
@click.argument("directory", default=".")
def git_init(directory):
    """Initialize a new git repo"""
    _run_git_command(["git", "init", directory], capture_output=True)
    click.echo(f"📁 Git repo initialized in {os.path.abspath(directory)}")


@git.command("clone")
@click.argument("url")
@click.argument("directory", required=False)
def git_clone(url, directory):
    """Clone a git repo"""
    args = ["git", "clone", url] + ([directory] if directory else [])
    _run_git_command(args)

