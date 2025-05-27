import click
import subprocess

# Define your command mappings here
COMMAND_MAP = {
    "init": "git init",
    "status": "git status",
    "do something": "echo You triggered a custom command",
}


@click.command()
@click.argument("command", nargs=-1, required=True)
def main(command):
    """NGit - Run mapped or raw shell commands using simple keywords."""
    user_input = ' '.join(command)

    # Map to real command if exists
    shell_command = COMMAND_MAP.get(user_input, user_input)

    click.echo(click.style(f"\n🔧 Executing: {shell_command}", fg="cyan"))

    try:
        result = subprocess.run(shell_command, shell=True, check=True, text=True,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        if result.stdout:
            click.secho(result.stdout.strip(), fg="green")
        if result.stderr:
            click.secho(result.stderr.strip(), fg="yellow")
    except subprocess.CalledProcessError as e:
        click.secho("❌ Command failed:", fg="red")
        click.secho(e.stderr.strip(), fg="red")
