import click
from prompt_toolkit import PromptSession


def prefill_input(prompt: str, prefill: str) -> str:
    click.echo(f"ℹ️ Press {click.style('Ctrl + C', fg='green')} to abort {click.style('Enter', fg='green')} to proceed")
    session = PromptSession()
    return session.prompt(prompt, default=prefill)
