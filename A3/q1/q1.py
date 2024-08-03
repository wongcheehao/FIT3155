# Written By : Wong Chee Hao
# Student ID : 32734751
# Date modified: 26/05/2024
import sys

##### Since the question ask to write a program that can generate all possible strings of length N from the alphabet A,
##### I will use a recursive approach to generate all possible permutations of the string of length N.
##### And calculate the stats based on the generated permutations.

def has_repeated_cyclic_rotation(s):
    """ Check if the string has a repeated cyclic rotation """
    doubled_s = (s + s)[1:-1]  # Concatenate the string with itself, excluding first and last characters
    return s in doubled_s

def permutation(alphabet, current_string, string_length, results):
    # Check if the current string has reached the desired length
    if len(current_string) == string_length:
        if has_repeated_cyclic_rotation(current_string):
            results['rotations'] += 1  # Increment count of cyclic rotations
        results['total'] += 1  # Increment total count of permutations

        # print(current_string)  # Print the permutation
        return

    # Recursively generate permutations by adding one more character
    for char in alphabet:
        permutation(alphabet, current_string + char, string_length, results)

def generate_permutations(alphabet_size, string_length):
    # Generate the alphabet based on the given size
    alphabet = [chr(ord('a') + i) for i in range(alphabet_size)]
    
    # Dictionary to store results: total permutations and those with cyclic rotations
    results = {'total': 0, 'rotations': 0}
    
    # Start the recursive permutation generation
    permutation(alphabet, "", string_length, results)
    
    return results

def main():
    if len(sys.argv) != 3:
        print("Usage: python q1.py <alphabet size> <string length>")
        return

    alphabet_size = int(sys.argv[1])
    string_length = int(sys.argv[2])
    
    results = generate_permutations(alphabet_size, string_length)
    one_distinct = alphabet_size
    more_than_one_distinct = results['total'] - one_distinct   
    exactly_n_distinct = results['total'] - results['rotations']
    is_multiple = more_than_one_distinct % string_length == 0

    print(more_than_one_distinct, exactly_n_distinct, one_distinct, is_multiple)

if __name__ == '__main__':
    main()