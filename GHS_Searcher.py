import os
import torch
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
    hymn_dict = {}  # Dictionary to map numbers to hymn titles
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
        
        # Detect hymn number
        if line.startswith("# GHS"):
            if hymn_number and title:
                hymn_key = f"GHS {hymn_number} - {title}"
                hymns[hymn_key] = "\n".join(lyrics).strip()
                hymn_dict[hymn_number] = hymn_key  # Store hymn number mapping

            parts = line.split(" ")
            if len(parts) > 2:
                hymn_number = parts[2].strip()
            else:
                hymn_number = None  # Reset if incorrect format

            lyrics = []  # Reset lyrics for new hymn
            title = None  # Reset title

        # Detect hymn title
        elif line.startswith("Title:"):
            title = line.replace("Title:", "").strip()

        # Collect hymn lyrics
        elif line:
            lyrics.append(line)

    # Save last hymn in file
    if hymn_number and title:
        hymn_key = f"GHS {hymn_number} - {title}"
        hymns[hymn_key] = "\n".join(lyrics).strip()
        hymn_dict[hymn_number] = hymn_key
    else:
        st.warning("Skipping last hymn due to missing number/title.")

    return hymns, hymn_dict

# Function to compute embeddings for hymns
@st.cache_resource
def compute_embeddings(hymns):
    hymn_titles = list(hymns.keys())
    hymn_lyrics = list(hymns.values())
    
    hymn_embeddings = model.encode(hymn_lyrics, convert_to_tensor=True)
    return hymn_titles, hymn_lyrics, hymn_embeddings

# Function to find hymns using search and semantic matching
def find_best_hymns(query, hymn_titles, hymn_lyrics, hymn_embeddings):
    matching_hymns = [
        (title, lyrics, 1.0)
        for title, lyrics in zip(hymn_titles, hymn_lyrics)
        if query.lower() in lyrics.lower()
    ]
    
    if matching_hymns:
        return matching_hymns[:3]  # Return top 3 exact matches

    query_embedding = model.encode(query, convert_to_tensor=True)
    similarities = util.pytorch_cos_sim(query_embedding, hymn_embeddings)[0]
    
    top_indices = torch.topk(similarities, k=3).indices.tolist()
    results = [(hymn_titles[i], hymn_lyrics[i], similarities[i].item()) for i in top_indices]
    
    return results

# Function to fetch Bible verse from API (including handling chapter-only input)
def fetch_bible_verse(reference):
    try:
        # Check if only the chapter is provided (e.g., "Deuteronomy 28")
        if ":" not in reference:
            # Assume it is a chapter and get a summary or full chapter
            url = f"https://bible-api.com/{reference}+summary"
        else:
            url = f"https://bible-api.com/{reference}"

        response = requests.get(url)
        data = response.json()

        # Return the summary or full text of the verse/chapter
        if "text" in data:
            return data["text"]
        elif "summary" in data:
            return data["summary"]
        else:
            st.error(f"No text or summary found for reference: {reference}")
            return None
    except Exception as e:
        st.error(f"Error fetching Bible verse: {e}")
        return None

# Load hymns and compute embeddings
hymn_file = "hymns.txt"  # Ensure this file exists
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
            hymns_found = False

            # Check if it's a hymn number input
            if query.isdigit() and query in hymn_dict:
                hymn_key = hymn_dict[query]
                st.subheader(f"🎶 {hymn_key}")
                st.write("📖 **Hymn Lyrics:**")

                formatted_lyrics = [f"***{line.strip()}***" if line.strip().lower().startswith(("verse", "refrain")) else line.strip() for line in hymns[hymn_key].split("\n")]
                formatted_text = "\n".join(formatted_lyrics).replace("\n", "  \n")

                st.markdown(
                    f"<div style='font-size:20px; line-height:1.8; white-space:pre-wrap;'>{formatted_text}</div>",
                    unsafe_allow_html=True
                )
                hymns_found = True

            # Check if it's a Bible verse or chapter reference
            elif any(char.isdigit() for char in query) and ":" in query:
                # Fetch Bible verse without displaying it
                bible_text = fetch_bible_verse(query)
                if bible_text:
                    top_hymns = find_best_hymns(bible_text, hymn_titles, hymn_lyrics, hymn_embeddings)
                    st.subheader("🎶 Recommended Hymns:")
                    for i, (title, lyrics, score) in enumerate(top_hymns, 1):
                        st.markdown(f"## {i}. {title}")  

                        with st.expander(f"📖 View Lyrics for {title}"):

                            formatted_lyrics = [f"***{line.strip()}***" if line.strip().lower().startswith(("verse", "refrain")) else line.strip() for line in lyrics.split("\n")]
                            formatted_text = "\n".join(formatted_lyrics).replace("\n", "  \n")

                            st.markdown(
                                f"<div style='font-size:20px; line-height:1.8; white-space:pre-wrap;'>{formatted_text}</div>",
                                unsafe_allow_html=True
                            )

                        st.write("---")  
                    hymns_found = True

            # Regular text or mixed input (e.g., text and Bible)
            if not hymns_found:
                top_hymns = find_best_hymns(query, hymn_titles, hymn_lyrics, hymn_embeddings)
                st.subheader("🎶 Recommended Hymns:")
                for i, (title, lyrics, score) in enumerate(top_hymns, 1):
                    st.markdown(f"## {i}. {title}")  

                    with st.expander(f"📖 View Lyrics for {title}"):

                        formatted_lyrics = [f"***{line.strip()}***" if line.strip().lower().startswith(("verse", "refrain")) else line.strip() for line in lyrics.split("\n")]
                        formatted_text = "\n".join(formatted_lyrics).replace("\n", "  \n")

                        st.markdown(
                            f"<div style='font-size:20px; line-height:1.8; white-space:pre-wrap;'>{formatted_text}</div>",
                            unsafe_allow_html=True
                        )

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
