from HW01_sultana_modified import load_special_characters, load_stopwords, load_inverted_index_from_file    #for loading related items for HW01
from HW01_sultana_modified import tokenize, lowercase, remove_punctuation, remove_stopwords, stemming  #for performing preprocesing
from HW01_sultana_modified import build_inverted_index, list_files_in_folder, process_text_dataset #to build inverted index if HW01 is not executed prior
import concurrent.futures   #for parallel processing



# this function display student's information
def display_student_info():
    equal_signs = "=" * 20
    text = "CSC790-IR Homework 02"
    output_line1 = equal_signs  + text  + equal_signs
    print(output_line1)
    print("First Name: Jeniya")
    print("Last Name : Sultana")
    output_line2 = "=" * (40 + len(text))
    print(output_line2)

# this function loads queries from file
def load_queries(query_file):
    with open(query_file, 'r', encoding = 'utf-8') as file:
        return set(file.read().splitlines())
    

# this function performs preprocessing of queries to match with preprocessed tokens of the inverted index
def query_pre_processing(query, custom_stopwords, special_chars):
    tokens = tokenize(query)
    lower_case_tokens = lowercase(tokens)
    punctuations_removed = remove_punctuation(lower_case_tokens, special_chars)
    filtered_tokens = remove_stopwords(punctuations_removed, custom_stopwords)
    stemmed_tokens = stemming(filtered_tokens)
    return stemmed_tokens

# this function generates sets of doc id for each of the token in the query
def generate_query_sets(inverted_index, query_tokens):
    query_sets = {}
    for token in query_tokens:
        if token in inverted_index:
            query_sets[token] = set(inverted_index[token]['documents'])
    return query_sets


# this function prints query token with AND/OR operators
def print_query_token_ops(binary, query_tokens):
    equal_signs = "=" * 20
    text1 = "Results for: "
    output_line2 = equal_signs  + text1 + query_tokens[0] 
    for j, op in enumerate(binary):
        token = query_tokens[j+1]
        if op == '0':  # AND operation
            output_line2 = output_line2 + " AND " + token
        elif op == '1':  # OR operation
            output_line2 = output_line2 + " OR " + token

    output_line2 = output_line2 + equal_signs
    print(output_line2)
    return len(output_line2)


# this function prints a line at the end of each result
def print_equal_line(length):
    output_line3 = "=" * length
    print(output_line3)


# this function perform set operation for the tokens of queries associated with doc id
def query_token_set_op(query_sets):
    num_of_op = len(query_sets) - 1   #num of AND/OR operations
    results = []  # List to store results
    
    # Generate binary numbers of appropriate length (representing AND/OR combinations)
    for i in range(2**num_of_op):
        binary = bin(i)[2:].zfill(num_of_op)  # generate binary number
        token = list(query_sets.keys())[0]   # first token with set of doc id
        result_set = query_sets[token]   #initialize with the first set
        tokens_used = [token]  # List to store tokens used in this combination
        
        for j, op in enumerate(binary):
            token = list(query_sets.keys())[j+1]  #iterate through the tokens of the query
            tokens_used.append(token)
            
            if op == '0':  # AND operation
                result_set = result_set.intersection(query_sets[token])
            elif op == '1':  # OR operation
                result_set = result_set.union(query_sets[token])
        
        # Append the tuple to the results list
        results.append((binary, tokens_used, result_set))
    
    return results


        
# this function prints the query with its number
def print_query_num(num, query):
    equal_signs = "=" * 20
    text1 = "User Query "
    text2 = ": "
    output_line1 = equal_signs  + text1  + str(num) + text2 + query + equal_signs
    print(output_line1)


# this function process each query for processing
def process_each_query(query, custom_stopwords, special_chars, inverted_index):  
    preprocessed_query = query_pre_processing(query, custom_stopwords, special_chars) #performs pre-processing
    query_sets = generate_query_sets(inverted_index, preprocessed_query)  #for each token of the query generate sets of docs
    results = query_token_set_op(query_sets) # perform AND/OR operation to obtain result
    return results


# this function prints result in the desired format
def print_result(query_num, query, results):
    print_query_num(query_num + 1, query)  # Print the query

    for result in results:
        binary, tokens_used, result_set = result #unpack result
        output_line3 = print_query_token_ops(binary, tokens_used)  #print token of query with AND/OR operators
        [print(f"file {num}") for num in sorted(result_set)]  #print result
        # print(result)   #uncomment this and comment out previous line of code if you only want the doc id
        print_equal_line(output_line3)


# this function process queries in parallel then print out the results
def process_queries(queries, custom_stopwords, special_chars, inverted_index):
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries)) as executor:
        # Submit each query for processing
        futures = [executor.submit(process_each_query, query, custom_stopwords, special_chars, inverted_index) for i, query in enumerate(queries)]

        # Print the results for each query in the order they were submitted
        for query_num, query in enumerate(queries):
            future = futures[query_num]
            print_result(query_num, query, future.result())


# this function builds inverted index if HW01 is not run first
def build_index(folder_path, custom_stopwords, special_chars):
    file_paths = list_files_in_folder(folder_path)
    # generates term frequency for each of the document
    term_frequency_dataset = process_text_dataset(file_paths, custom_stopwords, special_chars)
    # build inverted index such that it contains the list of documents for each term in a sorted manner (document that contain
    # most of that terms are at the beginning of the list)
    inverted_index = build_inverted_index(term_frequency_dataset)
    return inverted_index
    

def main():
    display_student_info()
    # paths for file loading
    stopwords_file = "stopwords/stopwords.txt"
    special_chars_file = "special_chars-queries/special-chars.txt"
    query_file = "special_chars-queries/queries.txt"

    # load stopwords and special characters 
    custom_stopwords = load_stopwords(stopwords_file)
    special_chars = load_special_characters(special_chars_file)

    # if you run don't run HW01 then need to build the inverted index. Please uncomment call to function build_index()
    folder_path = "Documents/Documents/"
    # inverted_index = build_index(folder_path, custom_stopwords, special_chars)

    # if run HW01 prior to HW02, then it will save the generated inverted index named "inverted_index.txt"
    inverted_index_filename = 'inverted_index.txt'
    inverted_index = load_inverted_index_from_file(inverted_index_filename)

    # load queries and pass to function for processing
    queries = load_queries(query_file)
    process_queries(queries, custom_stopwords, special_chars, inverted_index) 
 
    

if __name__ == "__main__":
    main()