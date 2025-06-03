import readline
import subprocess


# Mapping of natural language to actual git commands
command_map = {
    "git add all files to staging area": "git add .",
    "git commit with message": lambda x: f'git commit -m "{x}"',
}

def translate(input_str):
    input_str = input_str.strip().lower()
    # Example: simple static mapping
    if input_str in command_map:
        cmd = command_map[input_str]
        return cmd if isinstance(cmd, str) else cmd("default message")
    return input_str  # fallback

def main():
    while True:
        # Step 1: Take natural language input
        user_input = input(">>> ")

        # Step 2: Translate
        translated_cmd = translate(user_input)

        # Step 3: Confirm and replace input buffer
        print(f"Translated: {translated_cmd}")
        readline.replace_line("git add .", 0)
        readline.redisplay()

        confirm = input("Press Enter to run, or type 'n' to cancel: ")
        if confirm.lower() == 'n':
            continue

        # Step 4: Execute
        try:
            subprocess.run(translated_cmd, shell=True)
        except Exception as e:
            print("Error:", e)

if __name__ == "__main__":
    main()

# # def prefill_input(prompt, prefill):
# #     def hook():
# #         readline.insert_text(prefill)
# #         readline.redisplay()
# #     readline.set_pre_input_hook(hook)
# #     try:
# #         return input(prompt)
# #     finally:
# #         readline.set_pre_input_hook()