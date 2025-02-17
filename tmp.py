import spacy
 
# Load the spaCy English model
nlp = spacy.load('en_core_web_sm')
 
def lemmatize(text):
    # Process the text using spaCy
    doc = nlp(text)
     
    # Extract lemmatized tokens
    lemmatized_tokens = [token.lemma_ for token in doc]
     
    # Join the lemmatized tokens into a sentence
    lemmatized_text = ' '.join(lemmatized_tokens)

    return lemmatized_text

# Define a sample text
the_text = "The quick brown foxes are jumping over the lazy dogs."
 
# Print the original and lemmatized text
print("Original Text:", the_text)
print("Lemmatized Text:", lemmatize(the_text))
