import click
from translator.offline_translator import translate
from executor import execute_function
from util import prefill_input
from commands.git import git
import warnings
from transformers.utils import logging
from yaspin import yaspin


# Suppress warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


@click.command()
@click.option('-e', 'execute', is_flag=True, help="Weather to execute the command (default: False)")
@click.argument("command", nargs=-1, required=True)
def ngit(execute, command):
    """NGit - A Natural Language Based git & gh Command Translator and Executor.
    by default, it will just translate
    use '-e' for execution
    """
    spinner = yaspin()
    spinner.start()

    user_input = ' '.join(command)
    generated_response = translate(user_input)

    spinner.stop()
    if execute:
        click.echo(f"\n🔧 Executing: {click.style(generated_response, fg='cyan')}")
        # Execute the generated response
        execute_function(generated_response)
    else:
        click.echo(f"\n🔧 Translation: {click.style(generated_response, fg='cyan')}")
        edited_response = prefill_input(">> ", generated_response)
        # Execute the edited response



    # TODO all debug statements
    # click.echo(f"Current task is: {user_input}")
    # click.echo(f"Generated response is: {generated_response}")
    # click.echo(f'Execute {execute} is passed')
    # click.echo(f'Edited response is: {edited_response}')


@click.group()
def mgit():
    pass


mgit.add_command(git)
