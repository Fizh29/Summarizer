import nltk
# nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.cluster.util import cosine_distance
import numpy as np
import networkx as nx
import streamlit as st

# Fungsi membaca artikel
def read_article(file_name):
    file = open(file_name, "r")
    filedata = file.readlines()
    article = filedata[0].split(". ")
    sentences = []
    for sentence in article:
        sentences.append(sentence.replace("[^a-zA-z]", " ").split(" "))
    sentences.pop()
    return sentences

# Fungsi menghitung kesamaan antar kalimat
def sentence_similarity(sent1, sent2, stopwords=None):
    if stopwords is None:
        stopwords = []
    sent1 = [w.lower() for w in sent1]
    sent2 = [w.lower() for w in sent2]
    all_words = list(set(sent1 + sent2))
    
    vector1 = [0] * len(all_words)
    vector2 = [0] * len(all_words)
    for w in sent1:
        if w in stopwords:
            continue
        vector1[all_words.index(w)] += 1
    for w in sent2:
        if w in stopwords:
            continue
        vector2[all_words.index(w)] += 1
    return 1 - cosine_distance(vector1, vector2)

# Fungsi membuat matriks kesamaan
def gen_sim_matrix(sentences, stop_words):
    similarity_matrix = np.zeros((len(sentences), len(sentences)))
    for idx1 in range(len(sentences)):
        for idx2 in range(len(sentences)):
            if idx1 == idx2:
                continue
            similarity_matrix[idx1][idx2] = sentence_similarity(sentences[idx1], sentences[idx2], stop_words)
    return similarity_matrix

# Fungsi untuk menghasilkan ringkasan
def generate_summary(text, top_n=5):
    stop_words = stopwords.words('english')
    summarize_text = []
    
    # Baca artikel
    sentences = text.split('. ')
    processed_sentences = []
    for sentence in sentences:
        processed_sentences.append(sentence.replace("[^a-zA-Z]", " ").split(" "))
    
    if len(processed_sentences) == 0:
        return "Error: No sentences found in the text."
    
    # Buat similarity matrix
    sentence_similarity_matrix = gen_sim_matrix(processed_sentences, stop_words)
    sentence_similarity_graph = nx.from_numpy_array(sentence_similarity_matrix)
    scores = nx.pagerank(sentence_similarity_graph)
    
    # Urutkan kalimat berdasarkan skor
    ranked_sentence = sorted(((scores[i], s) for i, s in enumerate(processed_sentences)), reverse=True)
    
    # Sesuaikan jumlah kalimat yang dirangkum
    top_n = min(top_n, len(ranked_sentence))
    
    for i in range(top_n):
        summarize_text.append(" ".join(ranked_sentence[i][1]))
    
    return ". ".join(summarize_text)

# Antarmuka Streamlit
st.title("Automatic Text Summarizer")
st.write("Upload a text file or paste your text below to generate a summary.")

# Input teks
uploaded_file = st.file_uploader("Upload a text file", type=["txt"])
input_text = st.text_area("Or paste your text here:", "")

# Input jumlah kalimat dalam ringkasan
top_n = st.slider("Number of sentences in the summary:", 1, 10, 3)

if st.button("Generate Summary"):
    if uploaded_file is not None:
        content = uploaded_file.read().decode("utf-8")
        summary = generate_summary(content, top_n)
        st.subheader("Summary:")
        st.write(summary)
    elif input_text:
        summary = generate_summary(input_text, top_n)
        st.subheader("Summary:")
        st.write(summary)
    else:
        st.error("Please upload a file or paste text to summarize.")
