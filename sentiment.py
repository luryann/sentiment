import subprocess
import sys
import os
import logging
import platform
import requests
import zipfile
import shutil
from pathlib import Path
import asyncio
import aiohttp
from transformers import pipeline
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
import time
import csv
import urwid
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
from cryptography.fernet import Fernet

# Required packages
required_packages = [
    'selenium', 'pandas', 'scikit-learn', 'textblob', 'transformers', 'torch', 'aiohttp', 'urwid', 'cryptography'
]

# Install required packages
def install_packages(packages):
    for package in packages:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_packages(required_packages)

# Logger setup
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Encryption key setup
KEY_FILE = "secret.key"
TOKEN_FILE = "token.enc"

def generate_key():
    key = Fernet.generate_key()
    with open(KEY_FILE, 'wb') as key_file:
        key_file.write(key)

def load_key():
    return open(KEY_FILE, 'rb').read()

def encrypt_message(message):
    key = load_key()
    fernet = Fernet(key)
    encrypted_message = fernet.encrypt(message.encode())
    with open(TOKEN_FILE, 'wb') as token_file:
        token_file.write(encrypted_message)

def decrypt_message():
    key = load_key()
    fernet = Fernet(key)
    with open(TOKEN_FILE, 'rb') as token_file:
        encrypted_message = token_file.read()
    return fernet.decrypt(encrypted_message).decode()

def is_token_available():
    return os.path.exists(KEY_FILE) and os.path.exists(TOKEN_FILE)

# Verify if Chrome is installed
def is_chrome_installed():
    chrome_path = shutil.which("google-chrome") or shutil.which("chrome") or shutil.which("chromium")
    if chrome_path:
        logger.info("Chrome is installed.")
        return True
    else:
        logger.error("Chrome is not installed.")
        return False

if not is_chrome_installed():
    logger.error("Please install Google Chrome and re-run the script.")
    sys.exit(1)

# Check for chromedriver
def get_chromedriver():
    if platform.system() == "Windows":
        chromedriver_name = "chromedriver.exe"
    else:
        chromedriver_name = "chromedriver"
    chromedriver_path = Path(chromedriver_name)

    if not chromedriver_path.exists():
        logger.info("chromedriver not found, downloading...")
        url = f"https://chromedriver.storage.googleapis.com/91.0.4472.101/chromedriver_{platform.system().lower()}64.zip"
        response = requests.get(url, stream=True)
        with open("chromedriver.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        with zipfile.ZipFile("chromedriver.zip", "r") as zip_ref:
            zip_ref.extractall()
        os.remove("chromedriver.zip")
        logger.info("chromedriver downloaded and extracted.")
    else:
        logger.info("chromedriver found.")

get_chromedriver()

# Initialize models
model_name_sentiment = "cardiffnlp/twitter-roberta-base-sentiment-latest"
sentiment_analysis = pipeline('sentiment-analysis', model=model_name_sentiment, tokenizer=model_name_sentiment)

model_name_ner = "dbmdz/bert-large-cased-finetuned-conll03-english"
ner_analysis = pipeline('ner', model=model_name_ner, tokenizer=model_name_ner)

model_name_summarization = "facebook/bart-large-cnn"
summarization = pipeline('summarization', model=model_name_summarization, tokenizer=model_name_summarization)

# CSV file setup
data_file = 'discord_messages.csv'

# Ensure the CSV file is initialized
if not Path(data_file).exists():
    with open(data_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["content", "sentiment", "score", "summary", "entities"])

# Setup Chrome options
chrome_options = Options()
chrome_options.add_argument("--headless")  # Ensure GUI is off
chrome_options.add_argument("--no-sandbox")
chrome_options.add_argument("--disable-dev-shm-usage")

# Set path to chromedriver as per your configuration
webdriver_service = Service("./chromedriver")

# Function to log in to Discord
def login_to_discord(email, password):
    try:
        driver.get('https://discord.com/login')
        time.sleep(5)  # Let the page load

        email_input = driver.find_element(By.NAME, 'email')
        password_input = driver.find_element(By.NAME, 'password')

        email_input.send_keys(email)
        password_input.send_keys(password)
        password_input.send_keys(Keys.RETURN)

        time.sleep(5)  # Let the page load
        logger.info('Logged in to Discord successfully.')

    except Exception as e:
        logger.error(f"Error during login: {e}")
        driver.quit()
        sys.exit(1)

# Async function to monitor messages in a channel
async def monitor_channel(session, channel_url, duration=60):
    try:
        driver.get(channel_url)
        time.sleep(10)  # Let the page load

        start_time = time.time()

        while time.time() - start_time < duration:
            messages = driver.find_elements(By.CLASS_NAME, 'messageContent-2qWWxC')
            tasks = []

            for message in messages:
                content = message.text
                tasks.append(analyze_and_save_message(content))
                chat_viewer_text.set_text(chat_viewer_text.get_text()[0] + "\n" + content)

            await asyncio.gather(*tasks)
            time.sleep(5)  # Adjust the frequency of message fetching as needed

    except Exception as e:
        logger.error(f"Error during monitoring: {e}")

# Async function to analyze sentiment and save to CSV file
async def analyze_and_save_message(content):
    try:
        sentiment_result = sentiment_analysis(content)[0]
        sentiment_label = sentiment_result['label']
        sentiment_score = sentiment_result['score']

        summary_result = summarization(content, max_length=50, min_length=25, do_sample=False)[0]
        summary = summary_result['summary_text']

        entities = ner_analysis(content)
        entities_text = ", ".join([entity['word'] for entity in entities])

        with open(data_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([content, sentiment_label, sentiment_score, summary, entities_text])
        
        logger.debug(f"Message: {content}, Sentiment: {sentiment_label}, Score: {sentiment_score}, Summary: {summary}, Entities: {entities_text}")

    except Exception as e:
        logger.error(f"Error during analysis: {e}")

# Evaluate model performance
def evaluate_model():
    try:
        data = pd.read_csv(data_file)
        if not data.empty:
            X = data['content']
            y = data['sentiment']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            predictions = [sentiment_analysis(text)[0]['label'] for text in X_test]
            report = classification_report(y_test, predictions, output_dict=True)
            accuracy = report['accuracy']
            status_text.set_text(f"Model Accuracy: {accuracy:.2f}\nNetwork Status: Connected\n")
        else:
            status_text.set_text("No data available for model evaluation.\nNetwork Status: Connected\n")
    except Exception as e:
        logger.error(f"Error during model evaluation: {e}")
        status_text.set_text(f"Error during model evaluation: {e}\nNetwork Status: Connected\n")

# Initialize the web driver
driver = webdriver.Chrome(service=webdriver_service, options=chrome_options)

# UI Components
console_output = urwid.Text("Console Output\n")
ml_analytics_output = urwid.Text("Machine Learning Analytics\n")
status_text = urwid.Text("Status: Initializing...\n")
chat_viewer_text = urwid.Text("Chat Viewer\n")

console_tab = urwid.LineBox(console_output, title="Console")
ml_analytics_tab = urwid.LineBox(ml_analytics_output, title="ML Analytics")
status_tab = urwid.LineBox(status_text, title="Status")
chat_viewer_tab = urwid.LineBox(chat_viewer_text, title="Chat Viewer")
main_tab = urwid.LineBox(urwid.Text("Main Page\nSummary and Stats\n"), title="Main")

tabs = [main_tab, console_tab, ml_analytics_tab, status_tab, chat_viewer_tab]
current_tab = 0

def switch_tab(input):
    global current_tab
    if input == 'tab':
        current_tab = (current_tab + 1) % len(tabs)
    elif input == 'q':
        raise urwid.ExitMainLoop()

    display_area.original_widget = tabs[current_tab]

display_area = urwid.Padding(urwid.AttrMap(tabs[current_tab], None))
main_loop = urwid.MainLoop(display_area, unhandled_input=switch_tab)

# Async main execution
async def main():
    async with aiohttp.ClientSession() as session:
        login_to_discord(DISCORD_EMAIL, DISCORD_PASSWORD)
        evaluate_model()
        await monitor_channel(session, CHANNEL_URL, duration=300)  # Monitor for 5 minutes

def first_time_setup():
    text = urwid.Text("First Time Setup\nPlease enter your Discord token:")
    edit = urwid.Edit()
    button = urwid.Button("Save", on_press=save_token, user_data=edit)
    pile = urwid.Pile([text, edit, button])
    filler = urwid.Filler(pile)
    main_loop.widget = filler

def save_token(button, edit):
    token = edit.get_edit_text()
    encrypt_message(token)
    main_loop.widget = urwid.Text("Token saved. Please restart the application.")
    raise urwid.ExitMainLoop()

def start_app():
    if is_token_available():
        # Show progress bar
        progress = urwid.ProgressBar('pg normal', 'pg complete', 0, 1, 'left')
        text = urwid.Text("Decrypting token...")
        pile = urwid.Pile([progress, text])
        filler = urwid.Filler(pile)
        main_loop.widget = filler
        
        # Simulate decryption delay
        for i in range(10):
            time.sleep(0.1)
            progress.set_completion((i + 1) / 10.0)
            main_loop.draw_screen()
        
        # Decrypt token
        try:
            global DISCORD_TOKEN
            DISCORD_TOKEN = decrypt_message()
            text.set_text("Decryption successful.")
            main_loop.widget = urwid.Text("Welcome to the Discord Sentiment and Analysis Tool")
            main_loop.draw_screen()
            asyncio.ensure_future(main())
            main_loop.run()
        except Exception as e:
            logger.error(f"Error decrypting token: {e}")
            main_loop.widget = urwid.Text(f"Error decrypting token: {e}")
    else:
        first_time_setup()

try:
    start_app()
finally:
    driver.quit()
