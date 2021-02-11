import csv
import sys
import math
from nltk.stem import PorterStemmer
import numpy as np




def write_output_file():
    '''
    Writes a dummy output file using the python csv writer, update this 
    to accept as parameter the found trace links. 
    '''
    with open('dataset-1/links.csv', 'w') as csvfile:

        writer = csv.writer(csvfile, delimiter=",", quotechar="\"", quoting=csv.QUOTE_MINIMAL)


        fieldnames = ["id", "links"]

        writer.writerow(fieldnames)

        writer.writerow(["UC1", "L1, L34, L5"]) 
        writer.writerow(["UC2", "L5, L4"])


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


# Input: 2 2D arrays of preprocessed data.
# Output: Similarity matrix
def similarityMatrix(H, L):
    num=np.dot(H,L.T)
    p1=np.sqrt(np.sum(H**2,axis=1))[:,np.newaxis]
    p2=np.sqrt(np.sum(L**2,axis=1))[np.newaxis,:]
    return num/(p1*p2)



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
def vectorRepresentation(tokenlist, master_vocabulary, n, d_list):
    vector = []

    for i in range(0, len(master_vocabulary)):
        if master_vocabulary[i] not in tokenlist:
            vector.append(0)
        else:
            tf = tokenlist.count(master_vocabulary[i])
            idf = math.log2(n/d_list[i])
            vector.append(tf * idf)

    return vector


# Input: highlevel and lowlevel requirements list and the master vocabulary(list) with an index
# Returns the number of requirements containing the ith word of the master vocabulary
def d(highlevel, lowlevel, master_vocabulary, i):
    word_from_vocabulary = master_vocabulary[i]
    requirements_counter = 0

    for list in highlevel:
        for word in list:
            if word == word_from_vocabulary:
                requirements_counter += 1

    for list in lowlevel:
        for word in list:
            if word == word_from_vocabulary:
                requirements_counter += 1

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

    print(f"Hello world, running with matchtype {match_type}!")

    # Read input low-level requirements and count them (ignore header line).
    with open("dataset-1/low.csv", "r") as inputfile:
        print(f"There are {len(inputfile.readlines()) - 1} low-level requirements")

    '''
    This is where you should implement the trace level logic as discussed in the 
    assignment on Canvas. Please ensure that you take care to deliver clean,
    modular, and well-commented code.
    '''

    high_preprocessed, high_index_list = preprocess("dataset-1/high.csv")
    low_preprocessed, low_index_list = preprocess("dataset-1/low.csv")

    master_vocabulary = master_vocabulary(high_preprocessed, low_preprocessed)
    n = total_requirements(high_preprocessed, low_preprocessed)
    d_list = create_d_array(high_preprocessed, low_preprocessed, master_vocabulary)

    vector = vectorRepresentation(low_preprocessed[0], master_vocabulary, n, d_list)


    # create similarity matrix
    # similarityMatrix(H, L) where H is vector representation of high, L of low

    # branch on program input (0, 1, 2 or 3)
    if match_type == 0: 
        # Similarity of at least 0
        print(0)
    if match_type == 1:
        # Similarity of at least .25
        print(1)    
    if match_type == 2:
        # Similarity of at least .67 of the most similar low level requirement.
        print(2)
    if match_type == 3:
        # custom technique
        print(3)
    
    # output current links to file (add parameter a list of created links)
    write_output_file()