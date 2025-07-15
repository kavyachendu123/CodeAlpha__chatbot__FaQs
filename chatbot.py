# ✅ Import required libraries
import nltk
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ✅ Download tokenizer
nltk.download('punkt')

# ✅ Define FAQ data
faq_data = {
    "What is AI?": "AI stands for Artificial Intelligence. It enables machines to mimic human intelligence.",
    "What is Machine Learning?": "Machine learning is a branch of AI that allows systems to learn from data.",
    "What is Python?": "Python is a high-level, interpreted programming language widely used in AI and data science.",
    "What is Java?": "Java is a high-level, object-oriented programming language used to build applications.",
    "What is a chatbot?": "A chatbot is a software that simulates conversation with users using AI.",
    "What is deep learning?": "Deep learning is a type of machine learning using neural networks with many layers.",
    "What is NLP?": "NLP stands for Natural Language Processing. It helps machines understand human language.",
}

# ✅ Prepare the data
questions = list(faq_data.keys())
answers = list(faq_data.values())

vectorizer = TfidfVectorizer()
question_vectors = vectorizer.fit_transform(questions)

# ✅ Function to get best response
def get_response(user_input):
    user_vector = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vector, question_vectors)
    index = np.argmax(similarity)
    score = np.max(similarity)
    
    if score > 0.3:
        return answers[index]
    else:
        return "🤖 Sorry, I don't understand that question."

# ✅ Runtime interaction loop
print("🤖 Hello! I am your FAQ Chatbot. Type 'exit' to end.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Bot: Goodbye! 👋")
        break
    response = get_response(user_input)
    print("Bot:", response)