from flask import Flask, render_template, request
from newspaper import Article
from transformers import pipeline
from rake_nltk import Rake

# Initialize Flask app
app = Flask(__name__)

# Load models for summarization and question answering
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Function to fetch articles from URLs
def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        print(f"Error fetching article from {url}: {e}")
        return None

# Route for home page
@app.route("/", methods=["GET", "POST"])
def index():
    urls = []  # Store URLs to pass back to the template
    results = None
    articles = []  # Initialize articles here to avoid UnboundLocalError
    error_messages = []  # List to store error messages for invalid URLs

    if request.method == "POST":
        option = request.form.get("option")
        url_1 = request.form.get("url_1")
        url_2 = request.form.get("url_2")
        url_3 = request.form.get("url_3")
        question = request.form.get("question")

        # Collect URLs to keep them in context
        if url_1:
            urls.append(url_1)
        if url_2:
            urls.append(url_2)
        if url_3:
            urls.append(url_3)

        # Fetch articles from URLs
        articles = [fetch_article(url) for url in urls]

        # Filter out None values for valid articles and track invalid URLs
        valid_articles = [article for article in articles if article]
        invalid_urls = [url for article, url in zip(articles, urls) if article is None]

        # Add error messages for invalid URLs
        for invalid_url in invalid_urls:
            error_messages.append(f"URL '{invalid_url}' is not valid or cannot be fetched.")

        # Perform summarization
        if option == "Summarize Articles" and valid_articles:
            summaries = []
            for article in valid_articles:
                try:
                    summary = summarizer(article, max_length=100, min_length=30, do_sample=False)
                    summaries.append(summary[0]['summary_text'])
                except Exception as e:
                    summaries.append(f"Error summarizing article: {e}")
            results = summaries

        # Perform keyword extraction
        elif option == "Extract Keywords" and valid_articles:
            rake = Rake()
            keywords = []
            for article in valid_articles:
                try:
                    rake.extract_keywords_from_text(article)
                    keywords.append(rake.get_ranked_phrases_with_scores()[:10])
                except Exception as e:
                    keywords.append(f"Error extracting keywords: {e}")
            results = keywords

        # Perform question answering
        elif option == "Ask a Question" and valid_articles and question:
            combined_text = " ".join(valid_articles)
            try:
                answer = qa_pipeline(question=question, context=combined_text)
                results = [answer['answer']]
            except Exception as e:
                results = [f"Error finding answer: {e}"]

    return render_template("index.html", results=results, articles=articles, urls=urls, error_messages=error_messages)

if __name__ == "__main__":
    app.run(debug=True)

