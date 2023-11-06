# Function to write a message to the screen and to the log file
def log(message, log_file_path):
    print(message)
    with open(log_file_path, 'a') as file:
        file.write(message + "\n")
