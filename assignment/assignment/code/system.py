"""Classification system.

Solution the COM2004/3004 assignment.


version: v1.0
"""

from typing import List

import numpy as np
import scipy.linalg
import scipy.stats
from scipy.stats import mode
from utils import utils
from utils.utils import Puzzle

# The required maximum number of dimensions for the feature vectors.
N_DIMENSIONS = 20


def load_puzzle_feature_vectors(image_dir: str, puzzles: List[Puzzle]) -> np.ndarray:
    """Extract raw feature vectors for each puzzle from images in the image_dir.

    OPTIONAL: ONLY REWRITE THIS FUNCTION IF YOU WANT TO REPLACE THE DEFAULT IMPLEMENTATION

    The raw feature vectors are just the pixel values of the images stored
    as vectors row by row. The code does a little bit of work to center the
    image region on the character and crop it to remove some of the background.

    You are free to replace this function with your own implementation but
    the implementation being called from utils.py should work fine. Look at
    the code in utils.py if you are interested to see how it works. Note, this
    will return feature vectors with more than 20 dimensions so you will
    still need to implement a suitable feature reduction method.

    Args:
        image_dir (str): Name of the directory where the puzzle images are stored.
        puzzle (dict): Puzzle metadata providing name and size of each puzzle.

    Returns:
        np.ndarray: The raw data matrix, i.e. rows of feature vectors.

    """
    return utils.load_puzzle_feature_vectors(image_dir, puzzles)


def reduce_dimensions(data: np.ndarray, model: dict) -> np.ndarray:
    """Reduce the dimensionality of a set of feature vectors down to N_DIMENSIONS.

    The function implements a basic PCA algorithm to get the 20 principal components.
    The old (non-reduced) training data is used to generate the covariance and the
    eigenvectors and then the data that has been inputted is centered and is "dot 
    product"-ed with the eigenvectors to produce our reduced dataset. Besides the
    first line, this code has been taken from the lab classes.

    Args:
        data (np.ndarray): The feature vectors to reduce.
        model (dict): A dictionary storing the model data that may be needed.

    Returns:
        np.ndarray: The reduced feature vectors.
    """
    # Take old training data, this is needed for when this function is called on the test data as we
    # do not have access to the training data then
    train_data = model["fvectors_train"]

    covx = np.cov(train_data, rowvar=0)
    N = covx.shape[0]
    w, v = scipy.linalg.eigh(covx, eigvals=(N - N_DIMENSIONS, N - 1))
    v = np.fliplr(v)
    pcadata = np.dot((data - np.mean(train_data)), v)

    return pcadata


def process_training_data(fvectors_train: np.ndarray, labels_train: np.ndarray) -> dict:
    """Process the labeled training data and return model parameters stored in a dictionary.

    Since we are not going for a non-parametric approach, all we need to store is the training
    labels, the full training data, and then the reduced training data.

    Args:
        fvectors_train (np.ndarray): training data feature vectors stored as rows.
        labels_train (np.ndarray): the labels corresponding to the feature vectors.

    Returns:
        dict: a dictionary storing the model data.
    """

    model = {}
    model["labels_train"] = labels_train.tolist()
    model["fvectors_train"] = fvectors_train.tolist()
    fvectors_train_reduced = reduce_dimensions(fvectors_train, model)
    model["fvectors_train_reduced"] = fvectors_train_reduced.tolist()
    return model


def classify_squares(fvectors_test: np.ndarray, model: dict) -> List[str]:
    """Nearest neighbour implementation of classify squares

    This uses the nearest neighbour algorithm to classift the squares. One function is a 1-NN algorithm
    that uses cosine distance as its distance measure, which produces an overall good score for both 
    high and low quality data. The second function is an implementation of the k-nearest neighbour 
    function using the euclidean distance as its distance measure. I have left it at k=3 which produces
    high quality results better than 1NN but worse results for low quality data than 1NN. Increasing k
    will improve the low quality data but worsen the high quality data

    Args:
        fvectors_train (np.ndarray): feature vectors that are to be classified, stored as rows.
        model (dict): a dictionary storing all the model parameters needed by your classifier.

    Returns:
        List[str]: A list of classifier labels, i.e. one label per input feature vector.
    """
    training_data = np.array(model["fvectors_train_reduced"])
    training_labels = np.array(model["labels_train"])

    return nearest_neighbour(training_data, training_labels, fvectors_test)

    # Because 1NN produces the better "overall" result I have kept that and commented the kNN implementation out
    # Feel free to try it and change k, or change p in Minkowski function.
    #return k_nearest_neighbour(3, training_data, training_labels, fvectors_test)



def find_words(labels: np.ndarray, words: List[str], model: dict) -> List[tuple]:
    """Implementation of find_words.

    This function first generates a dictionary of the words and their coordinates, as well
    as a score dictionary in order to make a "best guess" if a few letters are incorrect.
    The function searches horizontally, vertically and finally diagonally - each using a 
    recursive method that compares each letter to build up the "score". We will take the coords
    of our best guess if the first and last letters are correct and up to 3 letters are incorrect.
    If we cannot make a guess then we just return -1, -1 for end row and col. 

    Args:
        labels (np.ndarray): 2-D array storing the character in each
            square of the wordsearch puzzle.
        words (list[str]): A list of words to find in the wordsearch puzzle.
        model (dict): The model parameters learned during training.

    Returns:
        list[tuple]: A list of four-element tuples indicating the word positions.
    """

    output_dict = {}
    score_dict = {} # Dictionary for keeping track of how correct we are with the guesses
    for word in words:
        output_dict[word] = (0,0,0,0)
        score_dict[word] = 0

    # Rows
    output_dict = finding_words_row(labels, words, output_dict, score_dict)

    # Columns
    output_dict = finding_words_column(labels, words, output_dict, score_dict)

    # Diagonally
    output_dict = finding_words_diagonally(labels, words, output_dict, score_dict)

    # Create list of positions from dict
    output_list = []
    for word in words:
        output_list.append(output_dict[word])

    return output_list

# HELPER FUNCTIONS

def minkowski(point1, point2, p):
    # The Minkowski distance formula can be seen as a generalisation of both the Manhattan and Euclidean distance functions.
    # For Euclidean distance use p = 2
    # For Manhattan distance use p = 1
    # Data doesn't really vary that much if different p values are used but Euclidean distance gives the best result.
    return np.sum((abs(point1-point2)**p))**(1/p)

def cosine(test_data, training_data):
    # This distance function has been taken from the lab classes.

    x = np.dot(test_data, training_data.transpose())
    test_mod = np.sqrt(np.sum(test_data * test_data, axis=1))
    train_mod = np.sqrt(np.sum(training_data * training_data, axis=1))
    return x / np.outer(test_mod, train_mod.transpose())

def nearest_neighbour(training_data, training_labels, test_data):
    # This method has been taken from the lab classes. 

    dist = cosine(test_data, training_data)
    nearest = np.argmax(dist, axis=1)
    return training_labels[nearest]

def k_nearest_neighbour(k, training_data, training_labels, test_data):
    # KNN implementation
    # By default this implementation uses k = 3 in the classify_squares function because 3 has performed the best
    # For 3NN results see the report

    test_labels = []

    for data_point in test_data:
        dist_list = []
        for j in range(0,len(training_data)):
            dist = minkowski(np.array(training_data[j,:]), data_point, 2)
            dist_list.append(dist)
        dist_list = np.array(dist_list)

        k_nearest = np.argsort(dist_list)[:k]

        labels = training_labels[k_nearest]

        modelabels = mode(labels)
        modelabels = modelabels.mode[0]
        test_labels.append(modelabels)

    return test_labels
    
def finding_words_row(matrix, words_list, dictionary, score_dict):
    for word in words_list:
        for i in range(0,matrix.shape[0]):
            for j in range(0, len(matrix[i])):
                start_row = i
                start_col = j
                end_row, end_col, score = go_horizontal_start(i, j, word, matrix)

                if(end_row != -1 and score > score_dict[word]):
                    dictionary[word] = (start_row, start_col, end_row, end_col)
                    score_dict[word] = score
                else:
                    continue
    return dictionary

def go_horizontal_start(i, j, word, matrix):
    end_row = -1
    end_col = -1
    score = 0

    if not (j + len(word) > matrix.shape[1]):
        end_row, end_col, score = go_right(matrix[i,j].lower(), i, j, word, 0, matrix, 0)
    if not (j - len(word) < -1) and end_col == -1:
        end_row, end_col, score = go_left(matrix[i,j].lower(), i, j, word, 0, matrix, 0)

    return end_row, end_col, score

def go_right(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_right(matrix[i, j+1].lower(), i, j+1, word, count+1, matrix, num_correct)
    else:
        return go_right(matrix[i, j+1].lower(), i, j+1, word, count+1, matrix, num_correct)

def go_left(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_left(matrix[i, j-1].lower(), i, j-1, word, count+1, matrix, num_correct)
    else:
        return go_left(matrix[i, j-1].lower(), i, j-1, word, count+1, matrix, num_correct)   

def finding_words_column(matrix, words_list, dictionary, score_dict):
    for word in words_list:
        for i in range(0,matrix.shape[0]):
            for j in range(0, len(matrix[i])):
                start_row = i
                start_col = j
                end_row, end_col, score = go_vertical_start(i, j, word, matrix)

                if(end_row != -1 and score > score_dict[word]):
                    dictionary[word] = (start_row, start_col, end_row, end_col)
                    score_dict[word] = score
                else:
                    continue
    return dictionary

def go_vertical_start(i, j, word, matrix):
    end_row = -1
    end_col = -1
    score = 0

    if not (i + len(word) > matrix.shape[0]):
        end_row, end_col, score = go_down(matrix[i,j].lower(), i, j, word, 0, matrix, 0)
    if not (i - len(word) < -1) and end_col == -1:
        end_row, end_col, score = go_up(matrix[i,j].lower(), i, j, word, 0, matrix, 0)

    return end_row, end_col, score

def go_down(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_down(matrix[i+1, j].lower(), i+1, j, word, count+1, matrix, num_correct)
    else:
        return go_down(matrix[i+1, j].lower(), i+1, j, word, count+1, matrix, num_correct)

def go_up(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_up(matrix[i-1, j].lower(), i-1, j, word, count+1, matrix, num_correct)
    else:
        return go_up(matrix[i-1, j].lower(), i-1, j, word, count+1, matrix, num_correct)     

def finding_words_diagonally(matrix, words_list, dictionary, score_dict):
    for word in words_list:
        for i in range(0, matrix.shape[0]):
            for j in range(0, len(matrix[i])):
                start_row = i
                start_col = j
                end_row, end_col, score = go_diagonal_start(i, j, word, matrix)

                if(end_row != -1 and score > score_dict[word]):
                    dictionary[word] = (start_row, start_col, end_row, end_col)
                    score_dict[word] = score
                else:
                    continue
    return dictionary

def go_diagonal_start(i, j, word, matrix):
    end_row = -1
    end_col = -1
    score = 0

    if not (i + len(word) > matrix.shape[0] or j - len(word) < -1):
        # try search down left
        end_row, end_col, score = go_down_left(matrix[i, j].lower(), i, j, word, 0, matrix, 0)
    if not (i + len(word) > matrix.shape[0] or j + len(word) > matrix.shape[1]) and end_row == -1:
        #if no coords found try search down right
        end_row, end_col, score = go_down_right(matrix[i,j].lower(), i, j, word, 0, matrix, 0)
    if not (i - len(word) < -1 or j - len(word) < -1) and end_row == -1:
        #if still no coords found try search up left
        end_row, end_col, score = go_up_left(matrix[i,j].lower(), i, j, word, 0, matrix, 0)
    if not (i - len(word) < -1 or j + len(word) > matrix.shape[1]) and end_row == -1:
        # search up right
        end_row, end_col, score = go_up_right(matrix[i,j].lower(), i, j, word, 0, matrix, 0)
    
    return end_row, end_col, score

def go_down_left(letter, i, j, word, count, matrix, num_correct):

    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_down_left(matrix[i+1, j-1].lower(), i+1, j-1, word, count+1, matrix, num_correct)
    else:
        return go_down_left(matrix[i+1, j-1].lower(), i+1, j-1, word, count+1, matrix, num_correct)

def go_down_right(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_down_right(matrix[i+1, j+1].lower(), i+1, j+1, word, count+1, matrix, num_correct)
    else:
        return go_down_right(matrix[i+1, j+1].lower(), i+1, j+1, word, count+1, matrix, num_correct)

def go_up_left(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_up_left(matrix[i-1, j-1].lower(), i-1, j-1, word, count+1, matrix, num_correct)
    else:
        return go_up_left(matrix[i-1, j-1].lower(), i-1, j-1, word, count+1, matrix, num_correct)

def go_up_right(letter, i, j, word, count, matrix, num_correct):
    if count == len(word)-1:
        if(letter == word[count] and num_correct >= len(word)-3):
            return i, j, num_correct+1
        else:
            return -1, -1, num_correct
    elif letter == word[count]:
        if(count == 0 and letter != word[count]):
            return -1, -1, num_correct
        num_correct = num_correct + 1
        return go_up_right(matrix[i-1, j+1].lower(), i-1, j+1, word, count+1, matrix, num_correct)
    else:
        return go_up_right(matrix[i-1, j+1].lower(), i-1, j+1, word, count+1, matrix, num_correct)