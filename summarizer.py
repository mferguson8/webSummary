import requests
from newspaper import Article
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

def get_text_from_url(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        text = article.text
        return text
    except Exception as e:
        print(f"Error extracting article content: {e}")
        return ""

def summarize_text(text):
    model_name = 'facebook/bart-large-cnn'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    model = BartForConditionalGeneration.from_pretrained(model_name)
    # Tokenize and encode the text
    inputs = tokenizer.encode(text, return_tensors='pt', max_length=1024, truncation=True)
    # Generate a summary
    summary_ids = model.generate(inputs, num_beams=4, max_length=150, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary

def main():
    url = input("Enter the URL: ")
    print("Fetching and processing the webpage content...")
    text = get_text_from_url(url)
    if not text:
        print("Failed to extract text from the URL.")
        return
    print("Summarizing the content...")
    summary = summarize_text(text)
    print("\nSummary of the webpage:")
    print(summary)

if __name__ == '__main__':
    main()
