import click
import subprocess
from util import prefill_input
from commands.git import git


# Define your command mappings here
COMMAND_MAP = {
    "init": "git init",
    "status": "git status",
    "do something": "echo You triggered a custom command",
}


@click.command()
@click.option('-e', 'execute', is_flag=True, help="Weather to execute the command (default: False)")
@click.argument("command", nargs=-1, required=True)
def ngit(execute, command):
    """NGit - A Natural Language Based git & gh Command Translator and Executor.
    by default, it will just translate
    use '-e' for execution
    """
    user_input = ' '.join(command)
    click.echo(f"Current task is: {user_input}")
    click.echo(f'Execute {execute} is passed')

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
        response = prefill_input(">>> ", shell_command)
        print(response)
    except subprocess.CalledProcessError as e:
        click.secho("❌ Command failed:", fg="red")
        click.secho(e.stderr.strip(), fg="red")


@click.group()
def mgit():
    pass


mgit.add_command(git)
