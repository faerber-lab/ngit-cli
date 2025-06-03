import os
import subprocess
import click


# Utility function for running and printing commands
def _run_git_command(command: list, capture_output=True):
    result = subprocess.run(command, capture_output=capture_output, text=True)
    if capture_output:
        output = result.stdout.strip() or result.stderr.strip()
        click.echo(output)
        return output
    return None


@click.group(invoke_without_command=True)
@click.pass_context
def git(ctx):
    if ctx.invoked_subcommand is None:
        click.echo('I was invoked without subcommand')
    else:
        click.echo(f"I am about to invoke {ctx.invoked_subcommand}")


@git.command()
@click.argument("username")
@click.argument("email")
def config_user(username, email):
    """Configure Git global username and email"""
    _run_git_command(["git", "config", "--global", "user.name", username], capture_output=False)
    _run_git_command(["git", "config", "--global", "user.email", email], capture_output=False)
    click.echo(f"✅ Git user configured as {username} <{email}>")


@git.command()
@click.argument("message")
def commit(message):
    """Commit with a message"""
    click.echo("🔧 Committing changes...")
    _run_git_command(["git", "commit", "-m", message])


@git.command()
@click.argument("branch_name")
def checkout(branch_name):
    """Checkout to a branch"""
    click.echo(f"🔀 Switching to branch '{branch_name}'...")
    _run_git_command(["git", "checkout", branch_name])


@git.command()
@click.argument("branch_name")
def create_branch(branch_name):
    """Create a new branch"""
    _run_git_command(["git", "branch", branch_name], capture_output=False)
    click.echo(f"🌿 Branch '{branch_name}' created")


@git.command()
@click.argument("branch_name")
def delete_branch(branch_name):
    """Delete a branch"""
    _run_git_command(["git", "branch", "-d", branch_name], capture_output=False)
    click.echo(f"❌ Branch '{branch_name}' deleted")


@git.command()
@click.argument("old_name")
@click.argument("new_name")
def rename_branch(old_name, new_name):
    """Rename a branch"""
    _run_git_command(["git", "branch", "-m", old_name, new_name], capture_output=False)
    click.echo(f"🔁 Branch renamed from '{old_name}' to '{new_name}'")


@git.command()
def status():
    """Show git status"""
    click.echo("📦 Git status:")
    _run_git_command(["git", "status"])


@git.command()
@click.option("--mode", default="soft", type=click.Choice(["soft", "mixed", "hard"]), help="Reset mode")
def reset_last_commit(mode):
    """Remove the last commit (soft/mixed/hard)"""
    _run_git_command(["git", "reset", f"--{mode}", "HEAD~1"], capture_output=False)
    click.echo(f"⚠️ Last commit removed with {mode} reset")


@git.command()
@click.argument("name")
@click.argument("url")
def add_remote(name, url):
    """Add a new remote"""
    _run_git_command(["git", "remote", "add", name, url], capture_output=False)
    click.echo(f"🔗 Remote '{name}' added with URL {url}")


@git.command()
@click.argument("name")
def remove_remote(name):
    """Remove a remote"""
    _run_git_command(["git", "remote", "remove", name], capture_output=False)
    click.echo(f"🔌 Remote '{name}' removed")


@git.command()
def list_remotes():
    """List all remotes"""
    click.echo("🌐 Git remotes:")
    _run_git_command(["git", "remote", "-v"])


@git.command()
@click.argument("files", nargs=-1)
def add(files):
    """Add files to staging"""
    _run_git_command(["git", "add"] + list(files), capture_output=False)
    click.echo(f"📥 Staged: {', '.join(files)}")


@git.command()
@click.argument("files", nargs=-1)
def unstage(files):
    """Remove files from staging"""
    _run_git_command(["git", "reset"] + list(files), capture_output=False)
    click.echo(f"📤 Unstaged: {', '.join(files)}")


@git.command()
@click.option("--remote", default="origin", help="Remote name")
@click.option("--branch", default="main", help="Branch name")
def pull(remote, branch):
    """Pull changes from remote"""
    click.echo(f"⬇️ Pulling from {remote}/{branch}...")
    _run_git_command(["git", "pull", remote, branch])


@git.command()
@click.option("--remote", default="origin", help="Remote name")
@click.option("--branch", default="main", help="Branch name")
def push(remote, branch):
    """Push changes to remote"""
    click.echo(f"⬆️ Pushing to {remote}/{branch}...")
    _run_git_command(["git", "push", remote, branch])


@git.command()
@click.argument("directory", default=".")
def init(directory):
    """Initialize a new git repo"""
    _run_git_command(["git", "init", directory], capture_output=False)
    click.echo(f"📁 Git repo initialized in {os.path.abspath(directory)}")


@git.command()
@click.argument("url")
@click.argument("directory", required=False)
def clone(url, directory):
    """Clone a git repo"""
    args = ["git", "clone", url] + ([directory] if directory else [])
    _run_git_command(args)

