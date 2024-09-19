
# Hybrid Keyword Extraction Project

This project explores and implements various keyword extraction methods in an attempt to create a hybrid solution with improved accuracy. Multiple extraction techniques such as TF-IDF, YAKE, and KeyBERT are combined to test their effectiveness on a dataset of StackOverflow posts.

## Project Overview

The goal of this project is to extract keywords from text data using different methods and combine their strengths to form a hybrid keyword extraction method. This approach aims to achieve better accuracy in identifying the most relevant keywords from a given text.

### Methods Used:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**: A statistical measure that evaluates the importance of a word in a document relative to a corpus.
2. **YAKE (Yet Another Keyword Extractor)**: An unsupervised keyword extraction algorithm that ranks keywords based on their statistical properties.
3. **KeyBERT**: A keyword extraction method based on BERT (Bidirectional Encoder Representations from Transformers), a transformer model that provides context-aware embeddings.
4. **Summa**: A library used for extracting keywords using TextRank, a graph-based ranking algorithm.
5. **Custom Preprocessing**: Involves stemming, tokenization, and stopword removal using the NLTK library to clean the text data before applying extraction methods.

## Dataset

The dataset used in this project is a sample of StackOverflow posts from the **Train.csv** file. For performance reasons, a random subset of the data was sampled, as well as the first 10,000 records.

- **Train.csv**: The original dataset containing StackOverflow posts with titles and bodies.
- **Sampled Dataset**: A small random sample from the original data (`0.00006` fraction of the dataset) and the first 10,000 rows of the dataset for faster experimentation.

## Preprocessing Pipeline

The text preprocessing step cleans the data to ensure efficient keyword extraction:
1. **Lowercasing**: Converts all text to lowercase to maintain uniformity.
2. **Regex Cleaning**: Removes special characters and digits.
3. **Tokenization**: Splits the text into individual tokens (words).
4. **Stemming**: Reduces words to their root form using the Porter Stemmer.
5. **Stopword Removal**: Removes common stopwords and unnecessary tokens like "www" and "http".

## Keyword Extraction Workflow

1. **Preprocess Data**: Using the `process` function, titles and bodies of posts are cleaned and tokenized.
2. **Token Extraction**: A regular expression is used to extract content between HTML tags like `<p>` and `<li>`.
3. **TF-IDF Calculation**: For each document in the sampled dataset, the term frequencies (`tf`) and inverse document frequencies (`idf`) are computed.
4. **Keyword Extraction**: YAKE and KeyBERT are used to extract keywords from the processed text.

## Code Structure

- **Imports**: Libraries for natural language processing (NLTK), data manipulation (Pandas, NumPy), and keyword extraction (YAKE, KeyBERT, Summa).
- **Data Sampling**: The `Train.csv` file is read into a Pandas DataFrame, and a small fraction of the data is sampled for faster experimentation.
- **Preprocessing**: Text data is cleaned and tokenized using the `process` function, which performs stemming and stopword removal.
- **TF-IDF Calculation**: Term frequency and inverse document frequency are calculated manually.
- **Keyword Extraction**: Keywords are extracted using multiple methods, including YAKE, KeyBERT, and TF-IDF.

## Usage

1. Clone the repository and ensure the necessary libraries are installed.
2. Place the **Train.csv** dataset in the working directory.
3. Run the Python script to process the data and extract keywords using multiple methods.

### Requirements

Install the following Python libraries before running the code:

```bash
pip install nltk pandas numpy scikit-learn summa yake keybert
```

Additionally, download the required NLTK resources:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

## Future Work

- Combine the results of different keyword extraction methods to develop a hybrid model that boosts accuracy.
- Test the model on additional datasets.
- Optimize performance for larger datasets.

---

## License

This project is open-source under the MIT License.

