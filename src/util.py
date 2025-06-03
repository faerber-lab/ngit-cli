import click
from prompt_toolkit import PromptSession


def prefill_input(prompt: str, prefill: str) -> str:
    click.echo("Press 'Ctrl + C' to abort 'Enter' to proceed")
    session = PromptSession()
    return session.prompt(prompt, default=prefill)
