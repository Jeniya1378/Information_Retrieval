import os    # for parsing documents
from nltk.tokenize import word_tokenize  # for tokenization
from nltk.stem import PorterStemmer  # for stemming 
import numpy as np
import multiprocessing #for parallel processing
import math # for logarithm


# this function display student's information
def display_student_info():
    equal_signs = "=" * 20
    text = "CSC790-IR Homework 04"
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

# this function generates the contigency table for each of the term
def build_contingency_table(term, term_frequency_dataset, relevance_labels, query_term_document_frequency):
    # initialization
    term_present_relevant = 0
    term_present_nonrelevant = 0
    term_absent_relevant = 0
    term_absent_nonrelevant = 0

    # calculating s, S_s
    for doc_id, terms_dict in term_frequency_dataset.items():
        terms = terms_dict.keys()
        if term in terms:
            rel_label = relevance_labels.get(doc_id, 0)
            if rel_label == 1:
                term_present_relevant += 1
            else:
                term_present_nonrelevant += 1
        else:
            rel_label = relevance_labels.get(doc_id, 0)
            if rel_label == 1:
                term_absent_relevant += 1
            else:
                term_absent_nonrelevant += 1

    s = term_present_relevant
    S_s = term_absent_relevant
    dft = query_term_document_frequency[term]

    return s, S_s, dft


# this function genrated contingency values for each term in the query
def build_contingency_table_for_query_terms(query_term_frequency_dataset, term_frequency_dataset, relevance_label, query_term_document_frequency, num_of_documents):
    contingency_value = {}
    # building contingency tables for each term in the query
    for term in query_term_frequency_dataset:
        s, S_s, dft = build_contingency_table(term, term_frequency_dataset, relevance_label, query_term_document_frequency)
        ct = calculate_ct(s, S_s, dft, num_of_documents)
        contingency_value[term] = {'s':s, 'S_s':S_s, 'dft':dft, 'ct': ct}
    return contingency_value


# this function reads the relevance value from the file
def read_relevance_file(file_path):
    relevance_labels = {}
    with open(file_path, "r") as file:
        for line in file:
            doc_name, rel_label = line.strip().split(',')
            doc_id = extract_doc_id(doc_name)
            relevance_labels[doc_id] = int(rel_label)
    return relevance_labels


# this function calculates ct for each t
def calculate_ct(s, S_s, dft, N):
    ct = math.log(((s + 0.5) / (S_s + 0.5)) / ((dft - s + 0.5) / (N - dft - S_s + 0.5)))
    return ct

# this function calculates RSVd value for each document
def calculate_rsvd(document_terms, query_terms, contingency_value):
    rsvd = 0
    for term in query_terms:
        if term in document_terms:
            rsvd += contingency_value[term]['ct']
    return rsvd

# this function generates rsvd value for each document
def build_rsvd_dict(term_frequency_dataset, query_term_frequency_dataset, contingency_value, relevance_labels):
    rsvd_dict = {}
    for doc_id, terms_dict in term_frequency_dataset.items():
        terms = list(terms_dict.keys())  
        rsvd = calculate_rsvd(terms, query_term_frequency_dataset.keys(), contingency_value)
        rel_label = relevance_labels.get(doc_id, 0)
        rsvd_dict[doc_id] = {'rsvd':rsvd, 'relevance':rel_label}

    return rsvd_dict

# this function gets the dft value for query terms
def find_query_document_frequency(document_frequency, query_term_frequency):
    query_term_document_frequency = {}

    for term in query_term_frequency:
        if term in document_frequency:
            query_term_document_frequency[term] = document_frequency[term]
        else:
            query_term_document_frequency = 0
    return query_term_document_frequency

# this function prints the first k rsvd values
def print_rsvd_values(rsvd_dict, k):
    # sorting based on RSVd scores
    sorted_rsvd = dict(sorted(rsvd_dict.items(), key=lambda item: item[1]['rsvd'], reverse=True))
    first_k_elements = list(sorted_rsvd.items())[:k]


    for count, value in enumerate(first_k_elements):
        print(f"RSV {{file_{value[0]}}} = {round(value[1]['rsvd'], 2)} label = {value[1]['relevance']}")


def main():
    display_student_info()
    # when extracting zip file it might result either in "Documents/" or "Documents/Documents"
    # please change the relative file path if needed
    # loads all files
    folder_path = "Documents/"
    stopwords_file = "stopwords/stopwords.txt"
    special_chars_file = "special_chars/special-chars.txt"
    num_of_documents, file_paths = list_files_in_folder(folder_path)
    custom_stopwords = load_stopwords(stopwords_file)
    special_chars = load_special_characters(special_chars_file)

    # generates term frequency for each of the document
    term_frequency_dataset = process_text_dataset(file_paths, custom_stopwords, special_chars)

    # generate dft
    document_frequency = calculate_document_frequency(term_frequency_dataset)


    query_path = "query.txt"
    query_term_frequency_dataset = process_text_file(query_path, custom_stopwords, special_chars)
    # get dft for query terms
    query_term_document_frequency = find_query_document_frequency(document_frequency, query_term_frequency_dataset)


    relevance_file_path = "file_label.txt"
    relevance_label = read_relevance_file(relevance_file_path)
    
    contingency_value = build_contingency_table_for_query_terms(query_term_frequency_dataset, term_frequency_dataset, relevance_label, query_term_document_frequency, num_of_documents)
    rsvd_dict = build_rsvd_dict(term_frequency_dataset, query_term_frequency_dataset, contingency_value, relevance_label)

    k = 10 # print first 10 rsvd values
    print_rsvd_values(rsvd_dict, k)



if __name__ == "__main__":
    main()