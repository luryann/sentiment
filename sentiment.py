import os
import logging
import json
import time
import asyncio
import csv
from selenium import webdriver
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from transformers import pipeline
import urwid
import base64
import matplotlib.pyplot as plt
import io
import threading
from selenium.common.exceptions import NoSuchWindowException, WebDriverException

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Replace this with your Discord token
DISCORD_TOKEN = ' '

# Hugging Face models
sentiment_model = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment-latest")
ner_model = pipeline('ner', model="dbmdz/bert-large-cased-finetuned-conll03-english")
summarization_model = pipeline('summarization', model="facebook/bart-large-cnn")

messages = []
results = []

class SentimentMonitorUI:
    def __init__(self, messages, results):
        self.messages = messages
        self.results = results
        self.filtered_messages = messages
        self.logs = []
        self.sentiment_stats = {'POSITIVE': 0, 'NEGATIVE': 0, 'NEUTRAL': 0}
        self.keyword_counts = {}

        # Define main UI components
        self.main_list = urwid.SimpleFocusListWalker([urwid.Text("Sentiment Monitor - Main Tab")])
        self.console_list = urwid.SimpleFocusListWalker([urwid.Text("Sentiment Monitor - Console Output Tab")])
        self.analytics_list = urwid.SimpleFocusListWalker([urwid.Text("Sentiment Monitor - Machine Learning Analytics Tab")])
        self.status_list = urwid.SimpleFocusListWalker([urwid.Text("Sentiment Monitor - Status Tab")])
        self.chat_list = urwid.SimpleFocusListWalker([urwid.Text("Sentiment Monitor - Chat Viewer Tab")])
        self.logs_list = urwid.SimpleFocusListWalker([urwid.Text("Sentiment Monitor - Logs")])
        
        self.main_view = urwid.ListBox(self.main_list)
        self.console_view = urwid.ListBox(self.console_list)
        self.analytics_view = urwid.ListBox(self.analytics_list)
        self.status_view = urwid.ListBox(self.status_list)
        self.chat_view = urwid.ListBox(self.chat_list)
        self.logs_view = urwid.ListBox(self.logs_list)
        
        self.search_edit = urwid.Edit("Search: ")
        self.search_box = urwid.LineBox(self.search_edit, title="Search Box", title_align='left')
        self.search_view = urwid.Pile([self.search_box, self.chat_view])
        
        self.header = urwid.Text("Sentiment Monitor - Press TAB to switch tabs, Enter to search")
        self.footer = urwid.Text("Main | Console | Analytics | Status | Chat Viewer | Logs")
        
        self.layout = urwid.Frame(
            body=self.main_view,
            header=self.header,
            footer=self.footer
        )
        
        self.views = [self.main_view, self.console_view, self.analytics_view, self.status_view, self.search_view, self.logs_view]
        self.current_view = 0
        
        self.loop = urwid.MainLoop(self.layout, unhandled_input=self.handle_input)
    
    def handle_input(self, key):
        if key == 'tab':
            self.current_view = (self.current_view + 1) % len(self.views)
            self.layout.body = self.views[self.current_view]
        elif key == 'enter':
            self.filter_messages(self.search_edit.edit_text)
    
    def filter_messages(self, query):
        self.filtered_messages = [msg for msg in self.messages if query.lower() in msg['message'].lower() or query.lower() in msg['user'].lower()]
        self.update_chat_view()
    
    def update_chat_view(self):
        self.chat_list.clear()
        for message_data in self.filtered_messages:
            self.chat_list.append(urwid.Text(f"[{message_data['timestamp']}] {message_data['user']}: {message_data['message']}"))
        try:
            self.loop.draw_screen()
        except RuntimeError as e:
            logging.error(f"Error drawing screen: {e}")
    
    def refresh(self, loop=None, data=None):
        try:
            self.loop.draw_screen()
        except RuntimeError as e:
            logging.error(f"Error drawing screen: {e}")
        self.loop.set_alarm_in(1, self.refresh)  # Refresh every second
    
    def run(self):
        self.refresh()
        self.loop.run()
    
    def log_messages(self, new_messages):
        for message_data in new_messages:
            self.messages.append(message_data)
            self.filtered_messages.append(message_data)
            self.chat_list.append(urwid.Text(f"[{message_data['timestamp']}] {message_data['user']} - {message_data['message']}"))
        try:
            self.loop.draw_screen()
        except RuntimeError as e:
            logging.error(f"Error drawing screen: {e}")
    
    def log_result(self, result):
        sentiment = result['sentiment'][0]['label']
        entities = ', '.join([entity['word'] for entity in result['entities']])
        summary = result['summary'][0]['summary_text']
        
        self.analytics_list.append(urwid.Text(f"Timestamp: {result['timestamp']}"))
        self.analytics_list.append(urwid.Text(f"User: {result['user']}"))
        self.analytics_list.append(urwid.Text(f"Message: {result['message']}"))
        self.analytics_list.append(urwid.Text(f"Sentiment: {sentiment}"))
        self.analytics_list.append(urwid.Text(f"Entities: {entities}"))
        self.analytics_list.append(urwid.Text(f"Summary: {summary}"))
        self.analytics_list.append(urwid.Divider())
        try:
            self.loop.draw_screen()
        except RuntimeError as e:
            logging.error(f"Error drawing screen: {e}")
    
    def log_status(self, status):
        self.status_list.append(urwid.Text(status))
        self.logs.append(status)
        self.logs_list.append(urwid.Text(status))
        try:
            self.loop.draw_screen()
        except RuntimeError as e:
            logging.error(f"Error drawing screen: {e}")
    
    def update_analytics(self, sentiment_stats, keyword_counts):
        self.analytics_list.clear()
        self.analytics_list.append(urwid.Text("Sentiment Distribution"))
        pie_chart = generate_sentiment_pie_chart(sentiment_stats)
        pie_chart_image = urwid.Text(('banner', f"{pie_chart}"), align='center')
        self.analytics_list.append(pie_chart_image)
        self.analytics_list.append(urwid.Divider())
        
        self.analytics_list.append(urwid.Text("Keyword Trends"))
        bar_chart = generate_keyword_bar_chart(keyword_counts)
        bar_chart_image = urwid.Text(('banner', f"{bar_chart}"), align='center')
        self.analytics_list.append(bar_chart_image)
        self.analytics_list.append(urwid.Divider())
        try:
            self.loop.draw_screen()
        except RuntimeError as e:
            logging.error(f"Error drawing screen: {e}")

# Function to generate pie chart for sentiment distribution
def generate_sentiment_pie_chart(sentiment_stats):
    labels = sentiment_stats.keys()
    sizes = sentiment_stats.values()
    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

# Function to generate bar chart for keyword trends
def generate_keyword_bar_chart(keyword_counts):
    sorted_keywords = sorted(keyword_counts.items(), key=lambda item: item[1], reverse=True)
    labels, values = zip(*sorted_keywords)
    fig, ax = plt.subplots()
    ax.bar(labels[:10], values[:10])  # Display top 10 keywords
    ax.set_ylabel('Frequency')
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return base64.b64encode(buf.getvalue()).decode('utf-8')

def discord_login_with_token(driver, token):
    driver.get("https://discord.com/login")
    script = """
    (function() {{
        document.body.appendChild(document.createElement('iframe')).contentWindow.localStorage.token = `"{}"`;
    }})();
    """.format(token)
    driver.execute_script(script)
    driver.get("https://discord.com/channels/@me")
    time.sleep(5)  # Wait for the page to load

def monitor_messages(driver, messages, ui):
    while True:
        try:
            elements = driver.find_elements(By.CLASS_NAME, 'messageContent-2qWWxC')
        except NoSuchWindowException as e:
            logging.error(f"Window closed: {e}")
            break
        except WebDriverException as e:
            logging.error(f"WebDriver error: {e}")
            time.sleep(1)
            continue
        
        new_messages = []
        for element in elements:
            try:
                message = element.text
                timestamp = element.find_element(By.CLASS_NAME, 'timestampClassName').text  # Replace with actual timestamp class name
                user = element.find_element(By.CLASS_NAME, 'usernameClassName').text  # Replace with actual username class name
            except Exception as e:
                logging.error(f"Error reading message: {e}")
                continue

            message_data = {"message": message, "timestamp": timestamp, "user": user}
            messages.append(message_data)
            new_messages.append(message_data)

        if new_messages:
            ui.log_messages(new_messages)
        time.sleep(1)

def process_messages(messages, results, ui, sentiment_model, ner_model, summarization_model, sentiment_stats, keyword_counts):
    while True:
        new_results = []
        for message_data in messages:
            message = message_data['message']
            timestamp = message_data['timestamp']
            user = message_data['user']
            try:
                sentiment = sentiment_model(message)
                entities = ner_model(message)
                summary = summarization_model(message, max_length=50, min_length=25, do_sample=False)
            except Exception as e:
                logging.error(f"Error processing message: {e}")
                ui.log_status(f"Error processing message: {e}")
                continue
            result = {
                "message": message,
                "timestamp": timestamp,
                "user": user,
                "sentiment": sentiment,
                "entities": entities,
                "summary": summary
            }
            results.append(result)
            new_results.append(result)
            update_sentiment_stats(sentiment_stats, sentiment[0]['label'])
            update_keyword_counts(keyword_counts, entities)
        if new_results:
            ui.update_analytics(sentiment_stats, keyword_counts)
            time.sleep(1)

def update_sentiment_stats(sentiment_stats, sentiment):
    if sentiment in sentiment_stats:
        sentiment_stats[sentiment] += 1
    else:
        sentiment_stats[sentiment] = 1

def update_keyword_counts(keyword_counts, entities):
    for entity in entities:
        keyword = entity['word']
        if keyword in keyword_counts:
            keyword_counts[keyword] += 1
        else:
            keyword_counts[keyword] = 1

def save_results_to_csv(results, file_path='results.csv'):
    fieldnames = ['timestamp', 'user', 'message', 'sentiment', 'entities', 'summary']
    try:
        logging.info("Starting to save results to CSV.")
        with open(file_path, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            logging.info("Writing header to CSV.")
            writer.writeheader()
            for result in results:
                writer.writerow({
                    'timestamp': result['timestamp'],
                    'user': result['user'],
                    'message': result['message'],
                    'sentiment': result['sentiment'][0]['label'],
                    'entities': ', '.join([entity['word'] for entity in result['entities']]),
                    'summary': result['summary'][0]['summary_text']
                })
                logging.info(f"Written result for message from {result['user']} at {result['timestamp']}.")
        logging.info(f"Results successfully saved to {file_path}")
    except Exception as e:
        logging.error(f"Error saving results to CSV: {e}")

ui = SentimentMonitorUI(messages, results)

def main():
    # Setup WebDriver
    options = webdriver.ChromeOptions()
    service = ChromeService(executable_path='./chromedriver.exe')
    driver = webdriver.Chrome(service=service, options=options)

    # Login to Discord using the token
    discord_login_with_token(driver, DISCORD_TOKEN)

    # Start monitoring and processing messages
    monitor_thread = threading.Thread(target=monitor_messages, args=(driver, messages, ui))
    process_thread = threading.Thread(target=process_messages, args=(messages, results, ui, sentiment_model, ner_model, summarization_model, ui.sentiment_stats, ui.keyword_counts))
    
    monitor_thread.start()
    process_thread.start()
    
    try:
        ui.run()
    except KeyboardInterrupt:
        logging.info("Exiting program.")
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        ui.log_status(f"Unexpected error: {e}")
    
    # Save results to CSV
    logging.info("Saving results to CSV before exiting.")
    save_results_to_csv(results)

if __name__ == "__main__":
    main()
