import os    # for parsing documents
import sys   # for getting size of objects
from nltk.tokenize import word_tokenize  # for tokenization
from nltk.stem import PorterStemmer  # for stemming
import concurrent.futures   #for parallel processing
import string 


# this function display student's information
def display_student_info():
    equal_signs = "=" * 20
    text = "CSC790-IR Homework 01"
    output_line1 = equal_signs  + text  + equal_signs
    print(output_line1)
    print("First Name: Jeniya")
    print("Last Name : Sultana")
    output_line2 = "=" * (40 + len(text))
    print(output_line2)


# this function parse files in a directory
def list_files_in_folder(folder_path):
    return [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]

# this function loads stopwords from the given custom stopwords file
def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as file:
        return set(file.read().splitlines())
    

# this function loads stopwords from the given custom stopwords file
def load_special_characters(special_char_file):
    with open(special_char_file, 'r', encoding='utf-8') as file:
        punctuation_chars = file.read().strip()
    return punctuation_chars

    

# this function extract the numeric part from the file name (assuming it's always at the end)
def extract_doc_id(file_path):
    return int(''.join(filter(str.isdigit, file_path)))

# this function reads each file in the directory
def read_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()
    

def tokenize(text):
    tokens = word_tokenize(text)
    return tokens


def lowercase(tokens):
    return [token.lower() for token in tokens]

def stemming(tokens):
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

def remove_punctuation(tokens, special_chars):
     # Create a translation table to remove punctuations
    translation_table = str.maketrans("", "", special_chars)
    
    # Remove punctuations from each token
    return [token.translate(translation_table) for token in tokens]



# this function performs tokenization and stemming for each of the file
def tokenize_and_stem(text):
    tokens = word_tokenize(text)
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

# this function removes stopwords after stemming
def remove_stopwords(tokens, custom_stopwords):
    return [token for token in tokens if token not in custom_stopwords]

# this function calculates term frequency for each of the file
def calculate_term_frequency(tokens):
    term_frequency = {}
    for token in tokens:
        term_frequency[token] = term_frequency.get(token, 0) + 1
    return term_frequency


# this function combines the tokenization, stemming, stop words removal and term frequency calculation process for each of the file
def process_text_file(file_path, custom_stopwords, special_chars):
    text = read_text(file_path)
    tokens = tokenize(text)
    lower_case_tokens = lowercase(tokens)
    punctuations_removed = remove_punctuation(lower_case_tokens, special_chars)
    filtered_tokens = remove_stopwords(punctuations_removed, custom_stopwords)
    stemmed_tokens = stemming(filtered_tokens)
    term_frequency = calculate_term_frequency(stemmed_tokens)
    return term_frequency

# this function combines term frequency calculation and document id extraction process together
def process_file(file_path, custom_stopwords, special_chars):
    doc_id = extract_doc_id(file_path)
    term_frequency = process_text_file(file_path, custom_stopwords, special_chars)
    return doc_id, term_frequency


# this function iterates through the files using parallelisation and performs text processing steps
def process_text_dataset(file_paths, custom_stopwords, special_chars):
    dataset = {} # this dataset contains term frequency for each of the file. This will be used for furthur processing

    with concurrent.futures.ThreadPoolExecutor() as executor:
        for file_path in file_paths:
            future = executor.submit(process_file, file_path, custom_stopwords, special_chars)
            try:
                doc_id, term_frequency = future.result()
                dataset[doc_id] = term_frequency
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    return dataset

# this function merges the frequencies of terms to be calculated over entire documents
def merge_and_calculate_total_frequency(doc_term_frequencies):
    merged_term_frequency = {}

    for term_frequency in doc_term_frequencies.values():
        for term, frequency in term_frequency.items():
            if term not in merged_term_frequency:
                merged_term_frequency[term] = 0
            merged_term_frequency[term] += frequency

    return merged_term_frequency


def build_inverted_index(doc_term_frequencies):
    inverted_index = {}

    for doc_id, term_frequency in doc_term_frequencies.items():
        for term, frequency in term_frequency.items():
            if term not in inverted_index:
                inverted_index[term] = {'total_frequency': 0, 'documents': set()}  # Use a set to store unique document IDs
            inverted_index[term]['total_frequency'] += frequency
            inverted_index[term]['documents'].add(doc_id)

    # Sort the inverted index by total frequency in descending order
    inverted_index = {term: {'total_frequency': data['total_frequency'], 
                             'documents': sorted(data['documents'])} 
                             for term, data in sorted(inverted_index.items(), 
                                                      key=lambda x: x[1]['total_frequency'], reverse=True)}
    return inverted_index

# this function saves the inverted index with filename "inverted_index.txt"
def save_inverted_index_to_file(inverted_index, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for term, data in inverted_index.items():
            total_frequency = data['total_frequency']
            documents = data['documents']
            file.write(f"{term}: {total_frequency}, {documents}\n")

# this function loads the inverted index from directory
def load_inverted_index_from_file(filename):
    inverted_index = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(": ")
            term = parts[0]
            data_str = ': '.join(parts[1:])
            total_frequency, documents_str = map(str.strip, data_str.split(", ", 1))

            total_frequency = int(total_frequency)
            documents = [int(doc_id) for doc_id in documents_str[1:-1].split(',')] if documents_str != '{}' else []

            inverted_index[term] = {'total_frequency': total_frequency, 'documents': documents}

    return inverted_index

# this function converts file size from bytes to megabytes
def get_size_in_megabytes(obj):
    return sys.getsizeof(obj) / (1024 * 1024)

# this function saves the term frequency for each of the document with the name "tf_data.txt"
# call to this function is commented in main() assuming it is not required for this assignmnet
# might require for furthur processing
def save_tf_to_file(tf_data, filename):
    with open(filename, 'w', encoding='utf-8') as file:
        for term, doc_freq in tf_data.items():
            file.write(f"{term}: {doc_freq}\n")

# load the saved tf_data.txt 
# call to this function is commented in main() assuming it is not required for this assignmnet
# might require for furthur processing 
def load_tf_from_file(filename):
    term_frequency = {}

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split(": ")
            doc_id = int(parts[0])
            term_freq_str = ': '.join(parts[1:])  # Correctly reconstruct term_freq part
            term_freq = eval(term_freq_str) if term_freq_str else {}
            term_frequency[doc_id] = term_freq

    return term_frequency


def main():
    display_student_info()
    # when extracting zip file it might result either in "Documents/" or "Documents/Documents"
    # please change the relative file path if needed
    # loads all files
    folder_path = "Documents/Documents/"
    stopwords_file = "stopwords/stopwords.txt"
    special_chars_file = "special_chars-queries/special-chars.txt"
    file_paths = list_files_in_folder(folder_path)
    custom_stopwords = load_stopwords(stopwords_file)
    special_chars = load_special_characters(special_chars_file)


    # generates term frequency for each of the document
    term_frequency_dataset = process_text_dataset(file_paths, custom_stopwords, special_chars)

    # build inverted index such that it contains the list of documents for each term in a sorted manner (document that contain
    # most of that terms are at the beginning of the list)
    inverted_index = build_inverted_index(term_frequency_dataset)
    # save and load inverted index
    inverted_index_filename = 'inverted_index.txt'
    save_inverted_index_to_file(inverted_index, inverted_index_filename)
    loaded_inverted_index = load_inverted_index_from_file(inverted_index_filename)

    # merge the frequency of the terms to get the most frequent terms for entire documents
    merged_term_frequency = merge_and_calculate_total_frequency(term_frequency_dataset)
    n = int(input("Enter n (to display top n frequent terms):"))
    top_terms = sorted(merged_term_frequency.items(), key=lambda x: x[1], reverse=True)[:n]

    print("Top ",n, " Most Frequent Terms:")
    for term, frequency in top_terms:
        print(f"{term}")
        
    # size of inverted index in bytes and megabytes    
    size_in_bytes = sys.getsizeof(inverted_index)
    size_in_megabytes = get_size_in_megabytes(inverted_index)

    print(f"Size of inverted index: {size_in_bytes} bytes or {size_in_megabytes:.2f} MB")

    # # Save the TF data to a file with name "tf_data.txt", it contains term frequency for each of the document. 
    # # Displaying this might not be required for this assignmnet.     
    # tf_filename = 'tf_data.txt'
    # save_tf_to_file(term_frequency_dataset, tf_filename)

    # # Load the TF data from the file with name "tf_data.txt", it contains term frequency for each of the document. 
    # # Displaying this might not be required for this assignmnet.  
    # loaded_tf_data = load_tf_from_file(tf_filename)


    #########################Query####################################


if __name__ == "__main__":
    main()