# Natural Git Command Line Tool

ngit is a simple natural language based git command line tool which enables users to use git in natural language. The package is built on top of `git` and `GitHub CLI` and is intended as a demonstration of edge device function calling AKA API calling using LLM.

## Features

- **This is still in development, No stable release is available yet.**
- You can run the ngit(Natural git) command like:
  - `ngit "Add all files to staging area"` >> this is just translate the command, and provide the command for edit
  - `ngit "Add all files to staging area" -e` >> this will execute the command without asking for edit
- You can run the mgit(Machine git) command like:
  - mgit git <command> <arguments> for example: `mgit git rename_branch develop development`
  - You can see all available commands by
    - `mgit --help`
    - `mgit git --help`
  - Available commands:
    - add                Add files to staging
    - add_remote         Add a new remote
    - checkout           Checkout to a branch
    - clone              Clone a git repo
    - commit             Commit with a message
    - config_email       Configure Git global email
    - config_username    Configure Git global username
    - create_branch      Create a new branch
    - delete_branch      Delete a branch
    - init               Initialize a new git repo
    - list_remotes       List all remotes
    - pull               Pull changes from remote
    - push               Push changes to remote
    - remove_remote      Remove a remote
    - rename_branch      Rename a branch
    - reset_last_commit  Remove the last commit (soft/mixed/hard)
    - status             Show git status
    - unstage            Remove files from staging


## Installation

You can install the package via **PyPI** or from **source**.

### Install from PyPI

pypi download url : https://pypi.org/project/ngit-cli/0.0.1/

```bash
pip install ngit-cli
```

for local installation of the package just run following from root directory
`pip install .`

### Install from Source (GitHub)

```bash
git clone https://github.com/faerber-lab/ngit-cli.git
cd ngit-cli
pip install .
```

## Usage

After installation, you can use `ngit-cli` using natural english to use git and GitHub from terminal

### Example:

```python
ngit "Change the develop branch name to development"
mgit git status
```

### Dependencies

1. [git](https://git-scm.com/downloads)
2. [GitHub CLI](https://cli.github.com/manual/gh)
3. torch
4. click
5. prompt_toolkit
6. yaspin

