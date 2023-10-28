import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download NLTK resources (run this once)
nltk.download('stopwords')
nltk.download('punkt')

# Load the dataset
df = pd.read_parquet('test.parquet')  # Replace 'test.parquet' with your file path

# Lowercasing the text
if 'Consumer complaint narrative' in df.columns:
    df['Consumer complaint narrative'] = df['Consumer complaint narrative'].str.lower()
else:
    print("Column 'Consumer complaint narrative' not found in the DataFrame.")

# Take a sample of the dataset for preprocessing (adjust the sample size as needed)
sample_size = 10  # Define the number of rows you want to sample
sample_df = df.sample(n=sample_size, random_state=1)  # Randomly sample 'sample_size' rows

# Text cleaning and preprocessing
def preprocess_text(text):
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Remove numbers
    text = re.sub(r'\d+', '', text)

    # Tokenization using NLTK
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]

    # Join tokens back into a single string
    preprocessed_text = ' '.join(filtered_tokens)
    return preprocessed_text

# Apply preprocessing to the sampled text column
if 'Consumer complaint narrative' in sample_df.columns:
    sample_df['Consumer complaint narrative'] = sample_df['Consumer complaint narrative'].apply(preprocess_text)
    print(sample_df['Consumer complaint narrative'].head())  # Display the preprocessed text for the sampled rows
else:
    print("Column 'Consumer complaint narrative' not found in the DataFrame.")