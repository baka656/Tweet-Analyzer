import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from transformers import pipeline
import tkinter as tk
from tkinter import ttk
from tkcalendar import DateEntry

# Load dataset
try:
    data = pd.read_csv('data.csv')
    data['date'] = pd.to_datetime(data['date'], format='%d-%m-%Y %H:%M')
    data['topic'] = data['topic'].str.strip()  # remove leading/trailing whitespaces
except FileNotFoundError:
    print("Error: File 'data.csv' not found.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: Empty file 'data.csv'.")
    exit()
except pd.errors.ParserError:
    print("Error: Invalid data format in 'data.csv'.")
    exit()

# Create GUI using Tkinter
class SentimentAnalysisGUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Sentiment Analysis GUI")
        self.geometry("500x500")
        self.create_widgets()

    def create_widgets(self):
        # Date range selection
        date_label = tk.Label(self, text="Select a date range to analyze:")
        date_label.pack(pady=10)

        # Start date selection
        start_date_label = tk.Label(self, text="Start Date:")
        start_date_label.pack()
        self.start_date_var = tk.StringVar()
        self.start_date_entry = DateEntry(self, date_pattern='yyyy-mm-dd', textvariable=self.start_date_var)
        self.start_date_entry.pack(pady=5)

        # End date selection
        end_date_label = tk.Label(self, text="End Date:")
        end_date_label.pack()
        self.end_date_var = tk.StringVar()
        self.end_date_entry = DateEntry(self, date_pattern='yyyy-mm-dd', textvariable=self.end_date_var)
        self.end_date_entry.pack(pady=5)

        # Generate Topics button
        generate_topics_button = tk.Button(self, text="Generate Topics", command=self.generate_topics)
        generate_topics_button.pack(pady=10)

        # Topic selection
        self.topic_var = tk.StringVar()
        topic_label = tk.Label(self, text="Select a topic:")
        topic_label.pack()
        self.topic_combobox = ttk.Combobox(self, textvariable=self.topic_var, state="readonly")
        self.topic_combobox.pack(pady=5)

        # Submit button
        submit_button = tk.Button(self, text="Submit", command=self.submit)
        submit_button.pack(pady=10)

    def generate_topics(self):
        # Generate a list of topics based on the selected date range
        try:
            start_date = pd.to_datetime(self.start_date_var.get())
            end_date = pd.to_datetime(self.end_date_var.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid date format.")
            return

        selected_data = data.loc[(data['date'] >= start_date) & (data['date'] <= end_date)]
        random_topics = random.sample(list(selected_data['topic'].unique()), 5)
        self.topic_combobox['values'] = random_topics
        self.topic_combobox.current(0)

    def submit(self):
        # Get user input and perform sentiment analysis for the selected topic
        try:
            start_date = pd.to_datetime(self.start_date_var.get())
            end_date = pd.to_datetime(self.end_date_var.get())
        except ValueError:
            tk.messagebox.showerror("Error", "Invalid date format.")
            return

        selected_data = data.loc[(data['date'] >= start_date) & (data['date'] <= end_date)]
        topic = self.topic_var.get()

        # Select tweets based on the chosen topic
        selected_data = selected_data.loc[selected_data['topic'] == topic]

        # Perform sentiment analysis using VADER
        sid = SentimentIntensityAnalyzer()
        selected_data['sentiment'] = selected_data['text'].apply(lambda x: 'Positive' if sid.polarity_scores(x)['compound'] >= 0.05 else ('Negative' if sid.polarity_scores(x)['compound'] <= -0.05 else 'Neutral'))

        overall_sentiment = selected_data['sentiment'].value_counts(normalize=True) * 100
        summary_pipeline = pipeline("summarization")
        summary = summary_pipeline('\n'.join(selected_data['text']), max_length=100, min_length=20, do_sample=False)

        # Display sentiment analysis results
        tk.messagebox.showinfo("Sentiment Analysis Results", f"Overall sentiment for the topic '{topic}':\n{overall_sentiment}\n\nBART Summary:\n{summary[0]['summary_text']}")

        # Display sentiment of each tweet
        sentiment_window = tk.Toplevel(self)
        sentiment_window.title(f"Sentiment of each tweet for topic '{topic}'")
        sentiment_window.geometry("600x400")
        sentiment_text = tk.Text(sentiment_window, wrap=tk.WORD)
        sentiment_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(sentiment_window, command=sentiment_text.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        sentiment_text.config(yscrollcommand=scrollbar.set)
        for _, row in selected_data.iterrows():
            sentiment_text.insert(tk.END, f"Tweet: {row['text']}\nSentiment: {row['sentiment']}\n\n")
        sentiment_text.config(state=tk.DISABLED)

        # Plot sentiment trend
        selected_data['date'] = selected_data['date'].dt.to_period('D' if (end_date - start_date).days <= 7 else 'W')
        sentiment_trend = selected_data.groupby(['date', 'sentiment']).size().unstack(fill_value=0)
        ax = sentiment_trend.plot(kind='bar', stacked=True, figsize=(10, 5))
        plt.title(f"Sentiment trend for topic '{topic}'")
        plt.xlabel("Date")
        plt.ylabel("Number of tweets")
        plt.legend(title="Sentiment")

        # Customize x-axis tick labels
        x_tick_labels = [date.strftime('%Y-%m-%d') if isinstance(date, pd.Period) else date for date in sentiment_trend.index]
        ax.set_xticklabels(x_tick_labels, rotation=45, ha='right')

        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    gui = SentimentAnalysisGUI()
    gui.mainloop()
