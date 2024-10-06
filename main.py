import streamlit as st
from newspaper import Article
from transformers import pipeline
from rake_nltk import Rake

# Load models for summarization and question answering
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Apply custom CSS for better styling
st.markdown("""
    <style>
        .main {
            background-image: url('https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.freepik.com%2Ffree-photos-vectors%2Ftech-gradient-background%2F59&psig=AOvVaw3LtqCEYeWbVaVbU0qu0UDz&ust=1727939506901000&source=images&cd=vfe&opi=89978449&ved=0CBQQjRxqFwoTCLi3r8KS74gDFQAAAAAdAAAAABBJ'); 
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            padding: 20px;
            color: white; /* Make text white so it's visible on dark backgrounds */
        }
        .sidebar .sidebar-content {
            background-color: #00416a;
            color: white;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            padding: 10px 20px;
            font-size: 16px;
            border-radius: 10px;
        }
        .header-text {
            font-size: 32px;
            color: #4CAF50;
            margin-bottom: 20px;
        }
        .subheader-text {
            font-size: 24px;
            color: #ff6347;
            margin-bottom: 15px;
        }
        .article-container {
            background-color: white;
            color: #333333; /* Ensure text is visible */
            padding: 15px;
            margin-bottom: 15px;
            border-radius: 10px;
            box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.1);
        }
        .footer {
            font-size: 12px;
            text-align: center;
            color: gray;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar for user options
st.sidebar.title("🔎 Choose an Option")
option = st.sidebar.radio("Select an option", ('Summarize Articles', 'Extract Keywords', 'Ask a Question'))

# Sidebar input for URLs
st.sidebar.subheader("📄 Input Article URLs")
url_1 = st.sidebar.text_input("Article URL 1")
url_2 = st.sidebar.text_input("Article URL 2")
url_3 = st.sidebar.text_input("Article URL 3")



# Function to fetch articles from URLs
def fetch_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        st.error(f"❌ Error fetching article from {url}")
        return None

# Get articles
articles = []
if url_1:
    articles.append(fetch_article(url_1))
if url_2:
    articles.append(fetch_article(url_2))
if url_3:
    articles.append(fetch_article(url_3))

# Display warning if no articles
if not articles or len([a for a in articles if a]) == 0:
    st.warning("⚠️ Please input at least one valid URL to proceed.")

# Show title for the app
st.markdown('<div class="header-text">📰 Article Analyzer</div>', unsafe_allow_html=True)

# If user selects summarization
if option == "Summarize Articles" and any(articles):
    st.markdown('<div class="subheader-text">📝 Summarized Articles</div>', unsafe_allow_html=True)
    for i, article in enumerate(articles):
        if article:
            summary = summarizer(article, max_length=100, min_length=30, do_sample=False)
            st.markdown(f'<div class="article-container"><strong>Summary of Article {i+1}</strong></div>', unsafe_allow_html=True)
            st.write(summary[0]['summary_text'])

# If user selects keyword extraction
elif option == "Extract Keywords" and any(articles):
    st.markdown('<div class="subheader-text">🔑 Extracted Keywords</div>', unsafe_allow_html=True)
    rake = Rake()
    for i, article in enumerate(articles):
        if article:
            rake.extract_keywords_from_text(article)
            keywords = rake.get_ranked_phrases_with_scores()
            st.markdown(f'<div class="article-container"><strong>Keywords of Article {i+1}</strong></div>', unsafe_allow_html=True)
            # Display top 10 keywords as bullet points
            for score, keyword in keywords[:10]:
                st.write(f"- {keyword} (Score: {score})")

# If user selects question-answering
elif option == "Ask a Question" and any(articles):
    st.markdown('<div class="subheader-text">❓ Question Answering</div>', unsafe_allow_html=True)
    question = st.text_input("Type your question:")
    if question:
        # Combine all articles' texts
        combined_text = " ".join([a for a in articles if a])
        # Perform question-answering
        answer = qa_pipeline(question=question, context=combined_text)
        st.markdown(f'<div class="article-container"><strong>Answer to your Question:</strong></div>', unsafe_allow_html=True)
        st.write(answer['answer'])

