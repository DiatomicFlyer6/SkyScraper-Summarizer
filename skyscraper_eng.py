import requests
from lxml import html
import tkinter as tk
from tkinter import messagebox, scrolledtext, ttk, simpledialog
import nltk
nltk.download('stopwords')
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
# Not strictly necessary, depends on summarization implementation
from nltk.cluster.util import cosine_distance
import numpy as np
# Not strictly necessary, depends on summarization implementation
import feedparser
import webbrowser

# Define dictionaries containing UI text for different languages
language_texts = {
    'english': {
        'enter_url': 'Enter the URL:',
        'enter_sentences': 'Enter the number of sentences for summarization:',
        'fetch_button': 'Fetch and Summarize Content',
        'fetch_news_button': 'Fetch News Headlines',
        'news_label': 'Latest News Headlines:',
        'success_message': 'Valuable content has been successfully summarized and saved.',
        'error_message': 'An error occurred: {}',
        'warning_message': 'No valuable content found based on the specified criteria.',
        'language_button': 'Choose Language',
    },
    'polish': {
        'enter_url': 'Enter the URL:',
        'enter_sentences': 'Enter the number of sentences for summarization:',
        'fetch_button': 'Fetch and Summarize Content',
        'fetch_news_button': 'Fetch News Headlines',
        'news_label': 'Latest News Headlines:',
        'success_message': 'Valuable content has been successfully summarized and saved.',
        'error_message': 'An error occurred: {}',
        'warning_message': 'No valuable content found based on the specified criteria.',
        'language_button': 'Choose Language',
    }
}


current_language = 'english'  # You can adjust this to your default language

def set_language():
    global current_language
    # Get the chosen language from the user
    language = simpledialog.askstring("Language", "Choose language (english/polish):")
    if language and language.lower() in ['english', 'polish']:
        current_language = language.lower()
        update_ui()

def update_ui():
    # Update UI texts according to the chosen language
    url_label.config(text=language_texts[current_language]['enter_url'])
    sentence_label.config(text=language_texts[current_language]['enter_sentences'])
    fetch_button.config(text=language_texts[current_language]['fetch_button'])
    fetch_news_button.config(text=language_texts[current_language]['fetch_news_button'])
    news_label.config(text=language_texts[current_language]['news_label'])
    success_message = language_texts[current_language]['success_message']
    error_message = language_texts[current_language]['error_message']
    warning_message = language_texts[current_language]['warning_message']
    language_button.config(text=language_texts[current_language]['language_button'])


def open_browser(url):
    webbrowser.open(url, new=2)

# Function to fetch and display news headlines using feedparser
def fetch_news_from_feed():
    """
    Function to fetch and display news headlines from an RSS feed using feedparser.

    It retrieves headlines and corresponding URLs from the specified RSS feed URL,
    displays them in the news text widget of the GUI, and adds hyperlink functionality.
    """

    try:
        # Specify the RSS feed URL
        rss_feed_url = 'https://rss.nytimes.com/services/xml/rss/nyt/HomePage.xml'  # Replace with your own RSS feed URL

        # Get and parse the RSS feed
        feed = feedparser.parse(rss_feed_url)

        # Extract headlines and URLs from the feed
        headlines_and_urls = [(entry['title'], entry['link']) for entry in feed.entries]

        # Display headlines in the news text widget with hyperlink functionality
        news_text.config(state=tk.NORMAL)  # Enable editing the text widget
        news_text.delete(1.0, tk.END)  # Clear previous content

        for i, (headline, url) in enumerate(headlines_and_urls, start=1):
            # Add hyperlink to the text widget
            news_text.insert(tk.END, f"{headline}\n", f'hyperlink_{i}')  # Use f-strings for dynamic tag names

            # Mark the hyperlink for click detection
            news_text.tag_add(f'hyperlink_{i}', news_text.index(tk.END) + f"-{len(headline)-1}c", tk.END)

            # Bind click event to the hyperlink with the open_browser function
            news_text.tag_bind(f'hyperlink_{i}', '<Button-1>', lambda event, url=url: open_browser(url))

        news_text.config(state=tk.DISABLED)  # Disable editing after adding links

    except Exception as e:
        # Handle any exceptions that occur during the process
        messagebox.showerror("Error", f"An error occurred while fetching news: {str(e)}")

def extract_and_summarize_content(url, max_sentences):
    """
    Function to extract valuable content, summarize it, and display/save the results.

    This function takes a URL and the maximum number of sentences for summarization as inputs.
    It attempts to fetch the content from the URL, extract valuable elements based on criteria,
    summarize the extracted content, and display/save the summarized content.

    Args:
        url (str): The URL of the web page to process.
        max_sentences (int): The maximum number of sentences for the summary.
    """

    try:
        # Get the content of the page using requests library
        response = requests.get(url)

        if response.status_code == 200:
            # Check for successful response (status code 200)
            # Use response.text instead of response.content for text data
            tree = html.fromstring(response.text)

            # Determine character encoding from response headers
            encoding = response.encoding
            if 'charset' in response.headers.get('content-type', ''):
                encoding = response.headers['content-type'].split('charset=')[-1]

            # Extract valuable content based on your criteria (paragraphs with text length > 50 characters)
            valuable_elements = tree.xpath('//p[string-length(text()) > 50]')

            if not valuable_elements:
                # Handle case where no valuable content is found based on criteria
                messagebox.showwarning("Warning", "No valuable content found based on the specified criteria.")
                return

            # Extract and display valuable content (combine text content from valuable elements)
            valuable_content = "\n".join(element.text_content() for element in valuable_elements)

            # Summarize the content to the user-specified number of sentences (implementation not shown)
            summarized_content = summarize_text(valuable_content, max_sentences)  # Replace with your summarization logic

            # Display the summarized content on the GUI text output widget
            text_output.delete(1.0, tk.END)  # Clear previous content
            text_output.insert(tk.END, summarized_content)

            # Save summarized content to a file with the appropriate encoding
            with open('summarized_content.txt', 'w', encoding=encoding) as file:
                file.write(summarized_content)

            messagebox.showinfo("Success", "Valuable content has been successfully summarized and saved.")
        else:
            # Handle error status codes from the request
            messagebox.showerror("Error", f"Failed to fetch the page. Status code: {response.status_code}")

    except Exception as e:
        # Handle any exceptions that occur during the process
        messagebox.showerror("Error", f"An error occurred: {str(e)}")


# Function to calculate similarity between two sentences
def sentence_similarity(sent1, sent2, stop_words):
    """
    This function calculates the cosine similarity between two sentences.

    It preprocesses the sentences by converting them to lowercase, removing non-alphanumeric characters,
    and removing stop words. Then, it creates TF (Term Frequency) vectors for both sentences
    and calculates the cosine similarity between these vectors.

    Args:
        sent1 (str): The first sentence.
        sent2 (str): The second sentence.
        stop_words (list): A list of stop words to be removed.

    Returns:
        float: The cosine similarity score between the two sentences (1 indicates most similar).
    """

    # Preprocess sentences: lowercase, remove non-alphanumeric characters, remove stop words
    words1 = [word.lower() for word in sent1.split() if word.isalnum() and word.lower() not in stop_words]
    words2 = [word.lower() for word in sent2.split() if word.isalnum() and word.lower() not in stop_words]

    # Find the unique words used in both sentences
    all_words = list(set(words1 + words2))

    # Create TF vectors for both sentences (number of occurrences of each unique word)
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for word in words1:
        vector1[all_words.index(word)] += 1
    for word in words2:
        vector2[all_words.index(word)] += 1

    # Calculate cosine similarity between the TF vectors
    return 1 - cosine_distance(vector1, vector2)  # 1 - distance for higher score for similar sentences


# Function to build a similarity matrix for all sentences
def build_similarity_matrix(sentences, stop_words):
    """
    This function builds a similarity matrix for all sentences in a list.

    It iterates through each pair of sentences (except for the same sentence)
    and calculates the cosine similarity using the `sentence_similarity` function.
    The similarity score is stored in the corresponding position of the matrix.

    Args:
        sentences (list): A list of sentences.
        stop_words (list): A list of stop words to be removed.

    Returns:
        numpy.ndarray: A numpy array representing the similarity matrix.
    """

    similarity_matrix = np.zeros((len(sentences), len(sentences)))  # Initialize a zero-filled matrix

    # Calculate similarity for each pair of sentences (excluding identical pairs)
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                similarity_matrix[i][j] = sentence_similarity(sentences[i], sentences[j], stop_words)

    return similarity_matrix

# Function to generate a summary of the content
def generate_summary(content, max_sentences):
    """
    This function generates a summary of the provided content based on sentence importance.

    It performs the following steps:
    1. Sentence tokenization: Splits the content into individual sentences.
    2. Stop word removal: Removes common words (stop words) from each sentence.
    3. Similarity matrix creation: Calculates the cosine similarity between all sentence pairs.
    4. Sentence ranking: Ranks sentences based on the sum of their similarity scores.
    5. Summary generation: Selects the top `max_sentences` sentences as the summary and joins them with newlines.

    Args:
        content (str): The content to be summarized.
        max_sentences (int): The maximum number of sentences for the summary.

    Returns:
        str: The generated summary of the content.
    """

    sentences = sent_tokenize(content)  # Split content into sentences

    # Remove stop words from sentences and convert to lowercase
    stop_words = set(stopwords.words("english"))
    sentences = [sentence.lower() for sentence in sentences if sentence.strip()]  # Remove empty sentences

    # Create a similarity matrix to represent sentence relationships
    similarity_matrix = build_similarity_matrix(sentences, stop_words)

    # Rank sentences based on the sum of their similarity scores with other sentences
    sentence_scores = np.sum(similarity_matrix, axis=1)
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[::-1][:max_sentences]]

    # Combine the top-ranked sentences as the summary
    return "\n".join(ranked_sentences)


# Function to summarize text with a default or optional number of sentences
def summarize_text(text, max_sentences=5):
    """
    This function acts as a wrapper for the `generate_summary` function.

    Args:
        text (str): The text to be summarized.
        max_sentences (int, optional): The maximum number of sentences for the summary. Defaults to 5.

    Returns:
        str: The generated summary of the text.
    """

    summarized_content = generate_summary(text, max_sentences)
    return summarized_content


# Function to fetch the URL, get the number of sentences for summarization,
# and call the function to extract, summarize, and save the content
def fetch_and_save_summarized_content():
    """
    This function retrieves user input (URL and number of sentences),
    and calls the `extract_and_summarize_content` function to process the content.
    """

    url = url_entry.get()
    max_sentences = int(sentence_entry.get())
    extract_and_summarize_content(url, max_sentences)


# Create the main application window
root = tk.Tk()
root.title("SkyScraper")

# Variable to store the current language selection
language_var = tk.StringVar(root)
language_var.set(current_language)  # Set the default language

# Button to open the language selection dialog
language_button = tk.Button(root, text="Choose Language", command=set_language)
language_button.place(relx=0.9, rely=0)  # Position the button in the top right corner

# Add labels for user input
url_label = tk.Label(root, text="Enter the URL:")
url_label.pack(pady=5)

# Entry field for URL input
url_entry = tk.Entry(root, width=40)
url_entry.pack(pady=5)

sentence_label = tk.Label(root, text="Enter the number of sentences for summarization:")
sentence_label.pack(pady=5)

# Entry field for number of sentences
sentence_entry = tk.Entry(root, width=10)
sentence_entry.pack(pady=5)

# Button to fetch and summarize content
fetch_button = tk.Button(root, text="Fetch and Summarize Content", command=fetch_and_save_summarized_content)
fetch_button.pack(pady=10)

# Scrollable text widget to display the summarized content
text_output = scrolledtext.ScrolledText(root, width=60, height=15, wrap=tk.WORD)
text_output.pack(pady=10)

# Button to fetch and display news headlines using the fetch_news_from_feed function
fetch_news_button = tk.Button(root, text="Fetch News Headlines", command=fetch_news_from_feed)
fetch_news_button.pack(pady=10)

# Label for the news section
news_label = tk.Label(root, text="Latest News:")
news_label.pack(pady=5)

# Scrollable text widget to display news headlines with hyperlink functionality
news_text = scrolledtext.ScrolledText(root, width=40, height=15, wrap=tk.WORD, state=tk.DISABLED)
news_text.pack(pady=10)

# Configure the news text widget to handle hyperlinks (blue foreground, underline)
news_text.tag_configure('hyperlink', foreground='blue', underline=True)

# Set the main window geometry (size)
root.geometry("1280x720")

# Run the main event loop for the GUI application
root.mainloop()
