import os
import torch
import re
import streamlit as st
import requests  # For fetching Bible verses
from sentence_transformers import SentenceTransformer, util

# Suppress symlink warnings
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Load the Sentence Transformer model
@st.cache_resource
def load_model():
    return SentenceTransformer('all-mpnet-base-v2')

model = load_model()

# Function to load hymns from a text file
def load_hymns(file_path):
    hymns = {}
    hymn_dict = {}
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.readlines()
    except FileNotFoundError:
        st.error("Hymn file not found. Please check the file path.")
        return {}, {}

    hymn_number = None
    title = None
    lyrics = []

    for line in content:
        line = line.strip()
        
        if line.startswith("# GHS"):
            if hymn_number and title:
                hymn_key = f"GHS {hymn_number} - {title}"
                hymns[hymn_key] = "\n".join(lyrics).strip()
                hymn_dict[hymn_number] = hymn_key

            parts = line.split(" ")
            hymn_number = parts[2].strip() if len(parts) > 2 else None
            lyrics = []
            title = None

        elif line.startswith("Title:"):
            title = line.replace("Title:", "").strip()

        elif line:
            lyrics.append(line)

    if hymn_number and title:
        hymn_key = f"GHS {hymn_number} - {title}"
        hymns[hymn_key] = "\n".join(lyrics).strip()
        hymn_dict[hymn_number] = hymn_key

    return hymns, hymn_dict

# Function to compute embeddings for hymns
@st.cache_resource
def compute_embeddings(hymns):
    hymn_titles = list(hymns.keys())
    hymn_lyrics = list(hymns.values())
    hymn_embeddings = model.encode(hymn_lyrics, convert_to_tensor=True)
    return hymn_titles, hymn_lyrics, hymn_embeddings

# Improved Function to fetch Bible verse and handle mixed input
def fetch_bible_verse(reference):
    try:
        url = f"https://bible-api.com/{reference}"
        response = requests.get(url)
        data = response.json()

        # Return the summary or full text of the verse/chapter
        if "text" in data:
            return data["text"]
        elif "summary" in data:
            return data["summary"]
        else:
            return None  # Return None if no valid text or summary found
    except Exception as e:
        st.error(f"Error fetching Bible verse: {e}")
        return None

# Enhanced Function to process mixed inputs
def process_query(query):
    bible_pattern = r"([1-3]?\s?[A-Za-z]+\s\d+:\d+)"  # Matches references like "Galatians 5:17"
    bible_match = re.search(bible_pattern, query)

    if bible_match:
        bible_reference = bible_match.group(0)
        bible_text = fetch_bible_verse(bible_reference)
        if bible_text:
            query = query.replace(bible_reference, bible_text)  # Replace reference with text
        else:
            query = query.replace(bible_reference, "")  # Remove the reference if not found

    return query.strip()

# Function to find hymns using search and semantic matching
def find_best_hymns(query, hymn_titles, hymn_lyrics, hymn_embeddings):
    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, hymn_embeddings)[0]
    
    top_indices = torch.topk(similarities, k=3).indices.tolist()
    results = [(hymn_titles[i], hymn_lyrics[i], similarities[i].item()) for i in top_indices]
    
    return results

# Load hymns and compute embeddings
hymn_file = "hymns.txt"
hymns, hymn_dict = load_hymns(hymn_file)
if hymns:
    hymn_titles, hymn_lyrics, hymn_embeddings = compute_embeddings(hymns)

# Sidebar Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Search Hymns", "About & Contact"])

if page == "Search Hymns":
    st.title("📖 GHS Search App 🎵")
    st.write("Enter a topic, Bible verse, a hymn number (1-260), or a line from hymn lyrics to find related hymns.")

    query = st.text_input("🔍 Search Hymns, Enter Bible Verse, or Hymn Number:", "")

    if st.button("Find Hymns"):
        if query:
            query = process_query(query)  # Process the query to handle mixed inputs

            top_hymns = find_best_hymns(query, hymn_titles, hymn_lyrics, hymn_embeddings)
            st.subheader("🎶 Recommended Hymns:")
            for i, (title, lyrics, score) in enumerate(top_hymns, 1):
                st.markdown(f"## {i}. {title}")  

                with st.expander(f"📖 View Lyrics for {title}"):
                    formatted_lyrics = [
                        f"***{line.strip()}***" if line.strip().lower().startswith(("verse", "refrain")) else line.strip()
                        for line in lyrics.split("\n")
                    ]
                    formatted_text = "\n".join(formatted_lyrics).replace("\n", "  \n")
                    st.markdown(f"<div style='font-size:20px; line-height:1.8; white-space:pre-wrap;'>{formatted_text}</div>",
                                unsafe_allow_html=True)
                st.write("---")
        else:
            st.warning("Please enter a topic, Bible verse, hymn number, or a line from the lyrics before searching.")

elif page == "About & Contact":
    st.title("📌 About GHSSAv1")
    st.markdown(
        """
        **GHSSAv1** is a semantic search and retrieval system designed to recommend suitable hymns for users.  
        It leverages a **pretrained sentence transformer (all-mpnet-base-v2)** to search the Gospel Hymns and Songs (GHS) of the Deeper Life Christian Ministry.  

        Users can search using:
        - A Bible verse in a specific format (e.g., *John 3:16*)
        - The exact number of a hymn (e.g., *11* for GHS 11)
        - A line from a hymn  
        - Any topic related to the Christian faith  

        The tool aims to assist Christians in easily finding relevant hymns based on their search query, whether it’s a verse, hymn number, or lyrics.
        """
    )
    st.title("📞 Contact")
    st.markdown(
        """
        For queries, please reach out to:  
        **Teak-Tech Engineering | Temitope Dada**  
        📧 [topeemmanueldada@yahoo.co.uk](mailto:topeemmanueldada@yahoo.co.uk)  
        """
    )
