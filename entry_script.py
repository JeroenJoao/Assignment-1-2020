import csv
import sys
import math
from nltk.stem import PorterStemmer
import numpy as np
from numpy import dot
from numpy.linalg import norm




def write_output_file(trace_link):
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('output/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)
        for key in trace_link:
            str_add = ""
            highlevel_list = trace_link.get(key)
            for el in highlevel_list:
                str_add += el + ","

            writer.writerow([key, str_add[:-1]])


# input: filepath of csv file
# returns 2D array containing an array of lists of tokenized words
def tokenize(filepath):
    tokens = []
    tokenIndexes = []
    with open(filepath, "r") as inputfile:
        csv_reader = csv.reader(inputfile, delimiter=',')
        for row in csv_reader:
            tokenIndexes.append(row[0])
            tokenized_row = row[1].split()
            tokens.append(tokenized_row)
    return tokens, tokenIndexes


# input: list of tokens
# returns list of tokens without stop-words
def stop_words_removal(tokens):
    tokens_without_stopword = []
    stopwords = ["a", "about", "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along", "already", "also","although","always","am","among", "amongst", "amoungst", "amount",  "an", "and", "another", "any","anyhow","anyone","anything","anyway", "anywhere", "are", "around", "as",  "at", "back","be","became", "because","become","becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside", "besides", "between", "beyond", "bill", "both", "bottom","but", "by", "call", "can", "cannot", "cant", "co", "con", "could", "couldnt", "cry", "de", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg", "eight", "either", "eleven","else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone", "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for", "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had", "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers", "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest", "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many", "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must", "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none", "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto", "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own","part", "per", "perhaps", "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she", "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something", "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their", "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon", "these", "they", "thick", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru", "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until", "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever", "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while", "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet", "you", "your", "yours", "yourself", "yourselves", "the"]
    for i in range(0, len(tokens)):
        if tokens[i] not in stopwords:
            tokens_without_stopword.append(tokens[i])
    return tokens_without_stopword


# input: path to csv files
# outputs preprocessed 2D array
def preprocess(path):
    preprocessed = []
    tokenized_input, index_list = tokenize(path)
    ps = PorterStemmer()

    for list in tokenized_input:
        low_tokens_without_stopwords = stop_words_removal(list)
        temp = []
        for word in low_tokens_without_stopwords:
            temp.append(ps.stem(word))
        preprocessed.append(temp)

    return preprocessed[1:], index_list[1:]  # slice the first element which is the type


# Input: 2 2D arrays of vectors.
# Output: Similarity matrix
def similarity_matrix(H, L):
    similarity_matrix_ret = []

    for vector_h in H:
        row = []
        for vector_l in L:
            row.append(cosine_similarity(vector_h, vector_l))
        similarity_matrix_ret.append(row)

    return similarity_matrix_ret


# Input: two vectors of the same size as list
# Returns: cosine similarity of the two vectors
def cosine_similarity(a, b):
    return dot(a, b)/(norm(a)*norm(b))


# Input: preprocessed high and lowlevel requirements
# Output: List of all tokens used
def master_vocabulary(highlevel, lowlevel):
    master_vocabulary_list = []

    for list in highlevel:
        for word in list:
            if word not in master_vocabulary_list:
                master_vocabulary_list.append(word)

    for list in lowlevel:
        for word in list:
            if word not in master_vocabulary_list:
                master_vocabulary_list.append(word)

    return master_vocabulary_list


# Input: list of tokens and list master vocabulary, n and d_list
# output: Vector representation of token list
def vector_representation(tokenlist, master_vocabulary, n, d_list):
    vector = []

    for i in range(0, len(master_vocabulary)):
        if master_vocabulary[i] not in tokenlist:
            vector.append(0)
        else:
            tf = tokenlist.count(master_vocabulary[i])
            idf = math.log2(n/d_list[i])
            vector.append(tf * idf)

    return vector

# returns list of vectors from requirements
def vector_list(requirements, master_vocabulary, n, d_list):
    vector_list = []
    for requirement in requirements:
        vector_list.append(vector_representation(requirement, master_vocabulary, n, d_list))
    return vector_list


# Input: highlevel and lowlevel requirements list and the master vocabulary(list) with an index
# Returns the number of requirements containing the ith word of the master vocabulary
def d(highlevel, lowlevel, master_vocabulary, i):
    word_from_vocabulary = master_vocabulary[i]
    requirements_counter = 0

    for list in highlevel:
        for word in list:
            if word == word_from_vocabulary:
                requirements_counter += 1
                break

    for list in lowlevel:
        for word in list:
            if word == word_from_vocabulary:
                requirements_counter += 1
                break

    return requirements_counter


# Input: highlevel, lowlevel, master_vocabulary
# Output: array where index i had the number of requirements containing the ith word of the master vocabulary
def create_d_array(highlevel, lowlevel, master_vocabulary):
    d_list = []

    for i in range(0, len(master_vocabulary)):
        d_list.append(d(highlevel, lowlevel, master_vocabulary, i))

    return d_list


# input: highlevel and lowlevel requirements
# returns the total number of requirements
def total_requirements(highlevel, lowlevel):
    return len(highlevel) + len(lowlevel)

# returns tracelink based on minimum similarity score
def tracelink_generation(sim_matrix, high_index_list, low_index_list, min_similarity):
    trace_link = {}

    for i in range(0, len(sim_matrix)):
        for j in range(0, len(sim_matrix[i])):
            if sim_matrix[i][j] > min_similarity:
                low_id = trace_link.get(high_index_list[i])
                if low_id is None:
                    trace_link[high_index_list[i]] = [low_index_list[j]]
                else:
                    trace_link[high_index_list[i]].append(low_index_list[j])

    return trace_link

def highest_similarity_tracelink(sim_matrix, high_index_list, low_index_list):
    trace_link = {}

    for i in range(0, len(sim_matrix)):
        max_similarity = np.max(sim_matrix[i])
        for j in range(0, len(sim_matrix[i])):
            if sim_matrix[i][j] >= max_similarity*0.67:
                low_id = trace_link.get(high_index_list[i])
                if low_id is None:
                    trace_link[high_index_list[i]] = [low_index_list[j]]
                else:
                    trace_link[high_index_list[i]].append(low_index_list[j])

    return trace_link


if __name__ == "__main__":
    '''
    Entry point for the script
    '''
    if len(sys.argv) < 2:
        print("Please provide an argument to indicate which matcher should be used")
        exit(1)

    match_type = 0

    try:
        match_type = int(sys.argv[1])
    except ValueError as e:
        print("Match type provided is not a valid number")
        exit(1)

    # preprocess the input
    high_preprocessed, high_index_list = preprocess("input/high.csv")
    low_preprocessed, low_index_list = preprocess("input/low.csv")

    # get information on input and create the vectors
    master_vocabulary = master_vocabulary(high_preprocessed, low_preprocessed)
    n = total_requirements(high_preprocessed, low_preprocessed)
    d_list = create_d_array(high_preprocessed, low_preprocessed, master_vocabulary)

    vectors_low = vector_list(low_preprocessed, master_vocabulary, n, d_list)
    vectors_high = vector_list(high_preprocessed, master_vocabulary, n, d_list) 

    # create similarity matrix
    sim_matrix = similarity_matrix(vectors_high, vectors_low)
    trace = tracelink_generation(sim_matrix, high_index_list, low_index_list, 0.25)

    # branch on program input (0, 1, 2 or 3)
    if match_type == 0: 
        # Similarity of at least 0
        trace = tracelink_generation(sim_matrix, high_index_list, low_index_list, 0.0)
        write_output_file(trace)
    if match_type == 1:
         # Similarity of at least 0.25
        trace = tracelink_generation(sim_matrix, high_index_list, low_index_list, 0.25)
        write_output_file(trace)
    if match_type == 2:
        # Similarity of at least .67 of the most similar low level requirement.
        trace = highest_similarity_tracelink(sim_matrix, high_index_list, low_index_list)
        write_output_file(trace)
    if match_type == 3:
        # custom technique
        print(3)