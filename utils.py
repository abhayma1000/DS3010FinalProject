import constants
import os
import time

import smtplib
from email.mime.text import MIMEText
import time
import nbformat

import json
from IPython.display import display, Javascript

start_time = time.time()

config = constants.Config('utils')

# Email configuration
SMTP_SERVER = "smtp.gmail.com"  # For Gmail, change if using Outlook or another service
SMTP_PORT = 465  # SSL port (or use 587 for TLS)
SENDER_EMAIL = "abhaymathur1000@gmail.com"  # Replace with your email
SENDER_PASSWORD = config.EMAIL_PASSWORD  # Replace with your email password
RECEIVER_EMAIL = "abhaymathur1000@gmail.com"  # Your email (or another recipient)

def send_email(subject, body=""):
    body += "\nAbhayUpdateEmail"

    """Sends an email notification"""
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        with smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT) as server:
            server.login(SENDER_EMAIL, SENDER_PASSWORD)
            server.sendmail(SENDER_EMAIL, RECEIVER_EMAIL, msg.as_string())
        print("Email sent successfully!")
    except Exception as e:
        print(f"Error sending email: {e}")

def get_time_from_start(start_time):
    elapsed_time = time.time() - start_time
    hours, rem = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{int(hours):02}:{int(minutes):02}:{seconds:05.2f}"

def get_next_filename(path):
    """Returns the next filename in the sequence"""
    num=0
    parts = path.split(".")


    new_path = parts[0] + f"_{num}." + parts[1]

    while os.path.exists(new_path):
        num += 1
        new_path = parts[0] + f"_{num}." + parts[1]
    
    return new_path

# def download_out_files():
#     # Download outputs of the jupyter notebook as a text file

def extract_outputs_to_json(notebook_path, output_file):
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    outputs = []
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if "text" in output:
                    outputs.append(output["text"])
                elif "data" in output and "text/plain" in output["data"]:
                    outputs.append(output["data"]["text/plain"])

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=4)

# Example usage
# extract_outputs_to_json("my_notebook.ipynb", get_next_filename("logs/output.json"))

def save_notebook():
    """Triggers a save action in Jupyter Notebook."""
    display(Javascript('IPython.notebook.save_checkpoint();'))

def transform_range(source_value, source_min, source_max, target_min, target_max):
    return (source_value - source_min) * (target_max - target_min) / (source_max - source_min) + target_min

def get_next_output_folder(outputs_folder_path):
    """Returns the next filename in the sequence"""
    num=0

    path = outputs_folder_path

    new_path = path + f"_{num}"

    while os.path.exists(new_path):
        num += 1
        new_path = path + f"_{num}"
    
    os.mkdir(new_path)
    
    return new_path, num


if __name__ == "__main__":
    print(get_next_filename("data/reviews.csv"))
    print(get_time_from_start(start_time))
    send_email("Test subject", "This is a test email from Python.")
