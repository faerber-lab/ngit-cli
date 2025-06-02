import os
import subprocess
import click


# Utility function for running and printing commands
def run_git_command(command: list, capture_output=True):
    result = subprocess.run(command, capture_output=capture_output, text=True)
    if capture_output:
        output = result.stdout.strip() or result.stderr.strip()
        click.echo(output)
        return output
    return None


@click.group()
def cli():
    """🧠 Natural Git CLI with Click"""
    pass


@cli.command()
@click.argument("username")
@click.argument("email")
def config_user(username, email):
    """Configure Git global username and email"""
    run_git_command(["git", "config", "--global", "user.name", username], capture_output=False)
    run_git_command(["git", "config", "--global", "user.email", email], capture_output=False)
    click.echo(f"✅ Git user configured as {username} <{email}>")


@cli.command()
@click.argument("message")
def commit(message):
    """Commit with a message"""
    click.echo("🔧 Committing changes...")
    run_git_command(["git", "commit", "-m", message])


@cli.command()
@click.argument("branch_name")
def checkout(branch_name):
    """Checkout to a branch"""
    click.echo(f"🔀 Switching to branch '{branch_name}'...")
    run_git_command(["git", "checkout", branch_name])


@cli.command()
@click.argument("branch_name")
def create_branch(branch_name):
    """Create a new branch"""
    run_git_command(["git", "branch", branch_name], capture_output=False)
    click.echo(f"🌿 Branch '{branch_name}' created")


@cli.command()
@click.argument("branch_name")
def delete_branch(branch_name):
    """Delete a branch"""
    run_git_command(["git", "branch", "-d", branch_name], capture_output=False)
    click.echo(f"❌ Branch '{branch_name}' deleted")


@cli.command()
@click.argument("old_name")
@click.argument("new_name")
def rename_branch(old_name, new_name):
    """Rename a branch"""
    run_git_command(["git", "branch", "-m", old_name, new_name], capture_output=False)
    click.echo(f"🔁 Branch renamed from '{old_name}' to '{new_name}'")


@cli.command()
def status():
    """Show git status"""
    click.echo("📦 Git status:")
    run_git_command(["git", "status"])


@cli.command()
@click.option("--mode", default="soft", type=click.Choice(["soft", "mixed", "hard"]), help="Reset mode")
def reset_last_commit(mode):
    """Remove the last commit (soft/mixed/hard)"""
    run_git_command(["git", "reset", f"--{mode}", "HEAD~1"], capture_output=False)
    click.echo(f"⚠️ Last commit removed with {mode} reset")


@cli.command()
@click.argument("name")
@click.argument("url")
def add_remote(name, url):
    """Add a new remote"""
    run_git_command(["git", "remote", "add", name, url], capture_output=False)
    click.echo(f"🔗 Remote '{name}' added with URL {url}")


@cli.command()
@click.argument("name")
def remove_remote(name):
    """Remove a remote"""
    run_git_command(["git", "remote", "remove", name], capture_output=False)
    click.echo(f"🔌 Remote '{name}' removed")


@cli.command()
def list_remotes():
    """List all remotes"""
    click.echo("🌐 Git remotes:")
    run_git_command(["git", "remote", "-v"])


@cli.command()
@click.argument("files", nargs=-1)
def add(files):
    """Add files to staging"""
    run_git_command(["git", "add"] + list(files), capture_output=False)
    click.echo(f"📥 Staged: {', '.join(files)}")


@cli.command()
@click.argument("files", nargs=-1)
def unstage(files):
    """Remove files from staging"""
    run_git_command(["git", "reset"] + list(files), capture_output=False)
    click.echo(f"📤 Unstaged: {', '.join(files)}")


@cli.command()
@click.option("--remote", default="origin", help="Remote name")
@click.option("--branch", default="main", help="Branch name")
def pull(remote, branch):
    """Pull changes from remote"""
    click.echo(f"⬇️ Pulling from {remote}/{branch}...")
    run_git_command(["git", "pull", remote, branch])


@cli.command()
@click.option("--remote", default="origin", help="Remote name")
@click.option("--branch", default="main", help="Branch name")
def push(remote, branch):
    """Push changes to remote"""
    click.echo(f"⬆️ Pushing to {remote}/{branch}...")
    run_git_command(["git", "push", remote, branch])


@cli.command()
@click.argument("directory", default=".")
def init(directory):
    """Initialize a new git repo"""
    run_git_command(["git", "init", directory], capture_output=False)
    click.echo(f"📁 Git repo initialized in {os.path.abspath(directory)}")


@cli.command()
@click.argument("url")
@click.argument("directory", required=False)
def clone(url, directory):
    """Clone a git repo"""
    args = ["git", "clone", url] + ([directory] if directory else [])
    run_git_command(args)


if __name__ == "__main__":
    cli()
    # status()
