import os    # for parsing documents
from nltk.tokenize import word_tokenize  # for tokenization
from nltk.stem import PorterStemmer  # for stemming 
import numpy as np
from decimal import Decimal # to truncate similarity_score to two decimal point
import multiprocessing #for parallel processing


# this function display student's information
def display_student_info():
    equal_signs = "=" * 20
    text = "CSC790-IR Homework 03"
    output_line1 = equal_signs  + text  + equal_signs
    print(output_line1)
    print("First Name: Jeniya")
    print("Last Name : Sultana")
    output_line2 = "=" * (40 + len(text))
    print(output_line2)


# this function parse files in a directory
def list_files_in_folder(folder_path):
    num_of_documents = len(os.listdir(folder_path))
    return num_of_documents, [os.path.join(folder_path, filename) for filename in os.listdir(folder_path)]


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
    

# this function tokenize text for each document
def tokenize(text):
    tokens = word_tokenize(text)
    return tokens


# this function converts texts into lowercase
def lowercase(tokens):
    return [token.lower() for token in tokens]

# this function generates the stem for each token
def stemming(tokens):
    ps = PorterStemmer()
    return [ps.stem(token) for token in tokens]

# this function remove punctuation from token list
def remove_punctuation(tokens, special_chars):
     # create a translation table to remove punctuations
    translation_table = str.maketrans("", "", special_chars)
    
    # remove punctuations from each token
    return [token.translate(translation_table) for token in tokens]


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
    dataset = {}  

    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_file, [(file_path, custom_stopwords, special_chars) for file_path in file_paths])

    for result in results:
        try:
            doc_id, term_frequency = result
            dataset[doc_id] = term_frequency
        except Exception as e:
            print(f"Error processing {result[0]}: {e}")

    return dataset

# this function find the set of unique terms in the collection
def finding_unique_terms(term_frequency_dataset):
    unique_terms = set()
    for frequencies in term_frequency_dataset.values():
        unique_terms.update(frequencies.keys())
    return unique_terms


# this function generate tf vector for a specific document
def process_tf_for_each_document(doc_number, frequencies, unique_terms):
    vector = np.zeros(len(unique_terms))
    for term, freq in frequencies.items():
        index = list(unique_terms).index(term)
        vector[index] = freq
    return (doc_number, vector)

# this function concurrently generate tf vector for all documents
def term_frequency_vector_generate(term_frequency_dataset, unique_terms):
    with multiprocessing.Pool() as pool:
        results = pool.starmap(process_tf_for_each_document, [(doc_number, frequencies, unique_terms) 
                                                              for doc_number, frequencies in term_frequency_dataset.items()])
    return results

# this function calculates the cosine similarity score between two documents
def cosine_similarity(vector1, vector2):
    dot_product = np.dot(vector1, vector2)
    norm_vector1 = np.linalg.norm(vector1)
    norm_vector2 = np.linalg.norm(vector2)
    similarity = dot_product / (norm_vector1 * norm_vector2)
    return similarity



# this function calculates similarity for a range of indices
def calculate_similarity_range(vectors, doc_numbers, start, end, result_queue):
    similarity_scores = {}
    for i in range(start, end):
        for j in range(i + 1, len(vectors)):
            doc_pair = (doc_numbers[i], doc_numbers[j])
            similarity_scores[doc_pair] = cosine_similarity(vectors[i], vectors[j])
    result_queue.put(similarity_scores)

# this function calculates cosine similarity in parallel
def calculate_cosine_similarity(vectors):
    # extract vectors
    doc_numbers, vectors = zip(*vectors)
    vectors = np.array(vectors)

    # split the work across available CPU cores
    num_cores = multiprocessing.cpu_count()
    chunk_size = len(vectors) // num_cores
    result_queue = multiprocessing.Queue()

    # create and start processes
    processes = []
    for i in range(num_cores):
        start = i * chunk_size
        end = (i + 1) * chunk_size if i < num_cores - 1 else len(vectors)
        process = multiprocessing.Process(target=calculate_similarity_range, args=(vectors, doc_numbers, start, end, result_queue))
        process.start()
        processes.append(process)

    # wait for all processes to finish and collect results
    similarity_scores = {}
    for _ in range(num_cores):
        similarity_scores.update(result_queue.get())

    # join all processes
    for process in processes:
        process.join()


    return similarity_scores



# print top k pairs of with similarity score
def print_top_k_painrs(similarity_scores, k, text):
    # create a set to keep track of pairs already included
    included_pairs = set()

    # sort the similarity scores dictionary by values
    sorted_similarity_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)

    # print the top k pairs
    print(f"Using {text}:")
    for doc_pair, similarity_score in sorted_similarity_scores:
        # Check if the pair or its reverse is already included
        if doc_pair not in included_pairs and (doc_pair[1], doc_pair[0]) not in included_pairs:
            similarity_score_truncated = Decimal(similarity_score).quantize(Decimal('0.00'), rounding='ROUND_DOWN')
            print(f"file {doc_pair[1]}, file {doc_pair[0]} with similarity of {similarity_score_truncated}")

            included_pairs.add(doc_pair)
            k -= 1
        if k == 0:
            break
    print("\n\n")

# this function generates document frequency for each of the term
def calculate_document_frequency(term_frequency_dataset):
    document_frequency = {}
    
    # initialize document frequency counts for each term to 0
    for frequencies in term_frequency_dataset.values():
        for term in frequencies.keys():
            document_frequency[term] = 0
    
    # count the number of documents containing each term
    for frequencies in term_frequency_dataset.values():
        for term in frequencies.keys():
            document_frequency[term] += 1
    
    return document_frequency

# this function generates idf for each of the document
def idf(num_of_documents, document_frequency):
    # Compute inverse document frequency (IDF)
    idf_vector = np.log10((num_of_documents) / (np.array(list(document_frequency.values()))))

    return idf_vector


# this function calculates tf-idf for each of the document
def calculate_tf_idf(num_of_documents, tf_vectors, document_frequency):

    # calculate idf
    idf_vector = idf(num_of_documents, document_frequency)

    # calculate tf-idf
    tf_idf_tuples = []
    for doc_number, tf_vector in tf_vectors:
        tf_idf_vector = tf_vector * idf_vector
        tf_idf_tuples.append((doc_number, tf_idf_vector.tolist()))

    return tf_idf_tuples

# this function calculates wf form tf
def sublinear_tf_scaling(tf):
    return 1 + np.log10(tf) if tf > 0 else 0


def calculate_tf_idf_sublinear(num_of_documents, term_frequency_vectors, document_frequency):
    # calculate idf
    idf_vector = idf(num_of_documents, document_frequency)

    # calculate wf-idf
    wf_idf_tuples = []
    for doc_number, tf_vector in term_frequency_vectors:
        wf_vector = [sublinear_tf_scaling(tf) for tf in tf_vector]
        wf_idf_vector = np.array(wf_vector) * idf_vector
        wf_idf_tuples.append((doc_number, wf_idf_vector.tolist()))

    return wf_idf_tuples


# this function merges the frequencies of terms to be calculated over entire documents
def merge_and_calculate_total_frequency(doc_term_frequencies):
    merged_term_frequency = {}

    for term_frequency in doc_term_frequencies.values():
        for term, frequency in term_frequency.items():
            if term not in merged_term_frequency:
                merged_term_frequency[term] = 0
            merged_term_frequency[term] += frequency

    return merged_term_frequency

def main():
    display_student_info()
    # when extracting zip file it might result either in "Documents/" or "Documents/Documents"
    # please change the relative file path if needed
    # loads all files
    folder_path = "Documents/Documents/"
    stopwords_file = "stopwords/stopwords.txt"
    special_chars_file = "special_chars-queries/special-chars.txt"
    num_of_documents, file_paths = list_files_in_folder(folder_path)
    custom_stopwords = load_stopwords(stopwords_file)
    special_chars = load_special_characters(special_chars_file)

    # generates term frequency for each of the document
    term_frequency_dataset = process_text_dataset(file_paths, custom_stopwords, special_chars)
    unique_terms = finding_unique_terms(term_frequency_dataset)

    # # addiotional printing
    # print(len(unique_terms))
    # # merge the frequency of the terms to get the most frequent terms for entire documents
    # merged_term_frequency = merge_and_calculate_total_frequency(term_frequency_dataset)
    # n = int(input("Enter n (to display top n frequent terms):"))
    # top_terms = sorted(merged_term_frequency.items(), key=lambda x: x[1], reverse=True)[:n]
    # print("Top ",n, " Most Frequent Terms:")
    # for term, frequency in top_terms:
    #     print(f"{term}")

    k = int(input("Enter value of k (top pairs):"))
    print(f"The top {k} closest documents are:\n")

    # generates tf vector for each of the document
    term_frequency_vector = term_frequency_vector_generate(term_frequency_dataset, unique_terms)    
    similarity_scores = calculate_cosine_similarity(term_frequency_vector)
    print_top_k_painrs(similarity_scores, k, "tf")

    # generates tf-idf for each of the document
    document_frequencies = calculate_document_frequency(term_frequency_dataset)
    tf_idf_dataset = calculate_tf_idf(num_of_documents, term_frequency_vector, document_frequencies)
    similarity_scores = calculate_cosine_similarity(tf_idf_dataset)
    print_top_k_painrs(similarity_scores, k, "tf-idf")

    # generates wf-idf for each of the document
    sublinear_tf_idf = calculate_tf_idf_sublinear(num_of_documents, term_frequency_vector, document_frequencies)
    similarity_scores = calculate_cosine_similarity(sublinear_tf_idf)
    print_top_k_painrs(similarity_scores, k, "wf-idf")





if __name__ == "__main__":
    main()