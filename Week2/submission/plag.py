import heapq
import numpy as np
import re

# Preprocess the input text: tokenize, normalize (lowercase, remove punctuation)
def preprocess_input(text):
    # Remove punctuation and split into sentences
    text = re.sub(r'[^\w\s]', '', text)
    sentences = text.lower().split('.')
    return [s.strip() for s in sentences if s.strip()]

# Calculate Levenshtein distance (edit distance) between two strings
def calculate_edit_distance(str1, str2):
    length1, length2 = len(str1), len(str2)
    distance_matrix = np.zeros((length1 + 1, length2 + 1), dtype=int)

    # Initialize distance matrix
    for i in range(length1 + 1):
        distance_matrix[i][0] = i
    for j in range(length2 + 1):
        distance_matrix[0][j] = j

    # Populate the distance matrix
    for i in range(1, length1 + 1):
        for j in range(1, length2 + 1):
            if str1[i - 1] == str2[j - 1]:
                distance_matrix[i][j] = distance_matrix[i - 1][j - 1]  # No cost if characters match
            else:
                distance_matrix[i][j] = min(distance_matrix[i - 1][j] + 1,  # Deletion
                                             distance_matrix[i][j - 1] + 1,  # Insertion
                                             distance_matrix[i - 1][j - 1] + 1)  # Substitution
    return distance_matrix[length1][length2]

# Heuristic function to estimate the remaining edit distance between sentences
def estimate_remaining_cost(pos1, pos2, sentences1, sentences2):
    remaining_sentences1 = sentences1[pos1:]
    remaining_sentences2 = sentences2[pos2:]
    heuristic_cost = 0
    # Use minimum possible edit distances between remaining sentences as the heuristic
    for i in range(min(len(remaining_sentences1), len(remaining_sentences2))):
        heuristic_cost += calculate_edit_distance(remaining_sentences1[i], remaining_sentences2[i])
    return heuristic_cost

# A* search algorithm to align sentences with minimal edit distance
def a_star_sentence_alignment(sentences1, sentences2):
    # Min-heap for A* search
    open_set = []
    heapq.heappush(open_set, (0, 0, 0, []))  # (total_cost, index1, index2, alignment_path)

    # Set to track visited states
    visited_states = set()

    # Explore states until there are none left
    while open_set:
        total_cost, index1, index2, alignment_path = heapq.heappop(open_set)

        # Check if the end of both documents is reached
        if index1 == len(sentences1) and index2 == len(sentences2):
            return alignment_path

        # Skip already explored state
        if (index1, index2) in visited_states:
            continue
        visited_states.add((index1, index2))

        # Align current sentences from both documents
        if index1 < len(sentences1) and index2 < len(sentences2):
            cost = calculate_edit_distance(sentences1[index1], sentences2[index2])
            new_path = alignment_path + [(sentences1[index1], sentences2[index2], cost)]
            g_cost = total_cost + cost
            h_cost = estimate_remaining_cost(index1 + 1, index2 + 1, sentences1, sentences2)
            heapq.heappush(open_set, (g_cost + h_cost, index1 + 1, index2 + 1, new_path))

        # Skip current sentence in sentences1
        if index1 < len(sentences1):
            g_cost = total_cost + calculate_edit_distance(sentences1[index1], "")
            new_path = alignment_path + [(sentences1[index1], "", g_cost)]
            heapq.heappush(open_set, (g_cost, index1 + 1, index2, new_path))

        # Skip current sentence in sentences2
        if index2 < len(sentences2):
            g_cost = total_cost + calculate_edit_distance("", sentences2[index2])
            new_path = alignment_path + [("", sentences2[index2], g_cost)]
            heapq.heappush(open_set, (g_cost, index1, index2 + 1, new_path))

    return None

# Function to identify potential plagiarism based on low edit distances
def identify_plagiarism(alignment_results, threshold=5):
    detected_cases = []
    for sent1, sent2, cost in alignment_results:
        if cost <= threshold and sent1 and sent2:
            detected_cases.append((sent1, sent2, cost))
    return detected_cases

# Main execution block
if __name__ == "__main__":
    # Sample documents
    document1 = """This is the first sentence. Here is another sentence. Plagiarism detection is important."""
    document2 = """This is the first sentence. This is a different sentence. Plagiarism detection is vital."""

    # Preprocess the documents
    sentences1 = preprocess_input(document1)
    sentences2 = preprocess_input(document2)

    # Perform sentence alignment using A* search
    alignment_results = a_star_sentence_alignment(sentences1, sentences2)

    # Print aligned sentences along with their edit distances
    print("Alignment Results:")
    for sent1, sent2, cost in alignment_results:
        print(f"Doc1: {sent1}\nDoc2: {sent2}\nEdit Distance: {cost}\n")

    # Detect plagiarism based on the specified edit distance threshold
    plagiarism_cases = identify_plagiarism(alignment_results, threshold=5)
    print("Potential Plagiarism Cases:")
    for sent1, sent2, cost in plagiarism_cases:
        print(f"Doc1: {sent1}\nDoc2: {sent2}\nEdit Distance: {cost}\n")
