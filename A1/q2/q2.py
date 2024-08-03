# Written By : Wong Chee Hao
# Student ID : 32734751
# Date modified: 25/03/2024
import sys
from bitarray import bitarray

##### Global variables
# Text and patterns are printable characters whose ASCII
# values are in the range [33, 126] (both inclusive).
ascii_start = 33
ascii_end = 126
ascii_range = ascii_end - ascii_start + 1

def delta(pat):
    """
    This function compute the delta vector for each character in the ascii_range. 
    (compared to the pattern)
    
    table[j-ascii_start][i] = 0, if character_j == pat[m-i-1]
    table[j-ascii_start][i] = 1, otherwise

    """
    m = len(pat)

    # Initialize the delta vector table
    delta_vector = [bitarray('1' * m) for _ in range(ascii_range)]

    # Compute the delta vector for each character in the ascii_range
    for j in range(ascii_start, ascii_end+1):
        for i in range(m):
            if pat[m-i-1] == chr(j):
                delta_vector[j-ascii_start][i] = 0

    return delta_vector

def Z_algo(s: str):
    """
    Implements Gusfield's Z-Algorithm to compute the z-values of a given string

    Args:
        string (str): The string to compute z-values for

    Returns:
        List[int]: A list of length len(string), with each index i corresponding to
                the z_i-value of the input string. First index is always None
    """
    n = len(s)

    # Handle the case when the string is empty
    if n == 0: 
        return [] 

    # Initialize the Z array 
    Z = [0] * n

    # Z[0] is always None
    Z[0] = None  

    # Initialize the left and right boundaries of the Z-box
    l, r = 0, 0  

    for k in range(1, n):
        # Case 1, k is outside the Z-box
        if k > r:  
            n_matches = 0
            while k + n_matches < n and s[n_matches] == s[k + n_matches]:
                n_matches += 1

            # Update the Z value
            Z[k] = n_matches

            if n_matches > 0:
                l = k
                r = k + n_matches - 1

        else:  # Case 2
            k_prime = k - l # k' = k - l, Python starts from index 0
            remaining = r - k + 1

            if Z[k_prime] < remaining:  # Case 2a
                Z[k] = Z[k_prime]

            elif Z[k_prime] == remaining:  # Case 2b
                n_matches = remaining # the matches we already know from Z-box

                # Explicitly check for more matches
                while k + n_matches < n and s[n_matches] == s[k + n_matches]:
                    n_matches += 1

                # Update the Z value = n_matches 
                Z[k] = n_matches 

                if n_matches > remaining:  # Update the l and r of Z-box if we found more matches outside from the Z-box
                    l = k
                    r = k + n_matches - 1

            else:  # Case 2c
                Z[k] = remaining

    return Z

def Z_bitvector(text, pat):
    """
    This function computes the bitvector(text, pat) using Z-algorithm.
    
    How it works:
    1. Concatenate the pattern and text together
    2. Run Z-algo on the concatenated string
    3. If the z-value is equal to m-i, set the bitvector to 0

    Why it works:
    1. The Z-algo will return the length of the longest substring starting 
    from the i-th index that matches its prefix
    2. If the z_pat_concatenate_text_array[m + i] is equal to m-i,
    => text[i .. m] == pat[0 .. m-i]

    Example:
    text = "aba"
    pat = "abc"
    z_pat_concatenate_text_array = Z_algo("abcaba") = [None, 0, 1, 2, 0, 1]

    (z_pat_concatenate_text_array[3] == 2) != 3-0 
    (z_pat_concatenate_text_array[4] == 0) != 3-1 
    (z_pat_concatenate_text_array[5] == 1) == 3-2 
    
    Hence,
    Z_bitvector(text, pat) = [1,1,0]
    """

    m = len(pat)

    # Initialize the bitvector to be returned
    bitvector = bitarray('1') * m
    
    # Run Z-algo on the the concat(pattern,text)
    z_pat_concatenate_text_array = Z_algo(pat + text)

    # Set set the bitvector to 0 if the z-value is equal to m-i
    for i in range(0,m):
        if z_pat_concatenate_text_array[i+m] == m - i:
            bitvector[i] = 0

    return bitvector

def bit_vector_pattern_matching(text, pattern):
    """
    This function implements the Bitwise Pattern Matching algorithm.

    Returns:
        List[int]: A list of index with each txt[index i ... index i + m - 1] == pattern
    """
    m = len(pattern)
    n = len(text)
    positions = [] # Initialize the reported matched pattern indexes reported
   
    ##### STEP 1
    # Precompute the delta_vector_table
    delta_vector_table = delta(pattern)
    
    #### STEP 2
    # Calculate the first bit vector bitvector_m-1 using Z_bitvector (m-1 because this is 0-indexed)
    bitvector = Z_bitvector(pattern, text[:m]) 
    
    # Append the index to positions if the pattern match the prefix of the text.
    if bitvector[0] == 0:   
        positions.append(1) # 1 because the returned positions are 1-indexed
    
    #### STEP 3
    # For the rest of the text, calculate the bitvector using the formula
    '''bitvector_j = (bitvector_j-1 ≪ 1) | Delta_j '''
    
    # Calculate the bitvector for the rest of the text (bitvector_j ... bitvector_n)
    for j in range(m, n): 

        # Get the character position in the delta_vector_table
        char_pos = ord(text[j]) - ascii_start
        
        ## Apply the formula
        # bitvector_j = (bitvector_j−1 ≪ 1) | Delta_j
        bitvector = (bitvector << 1) | delta_vector_table[char_pos] # Compute for the next bitvector "b_i"
        
        # Append the index to positions if the pattern match text[j − i + 1 ... j]]
        if bitvector[0] == 0:
            positions.append((j - m + 1) + 1) # +1 because the returned positions are 1-indexed
    
    return positions

def q2():
    if len(sys.argv) != 3:
        print("Usage: python q1.py <text filename> <pattern filename>")
        sys.exit(1)

    text_filename, pattern_filename = sys.argv[1], sys.argv[2]
    text = read_file(text_filename)
    pattern = read_file(pattern_filename)

    text = text[0].strip()

    for pat in pattern:
        positions = bit_vector_pattern_matching(text, pat)

        with open("output_q2.txt", "w") as f:
            for position in positions:
                f.write(f"{position}\n")

# this function reads a file and return its content
def read_file(file_path: str) -> str:
    f = open(file_path, 'r')
    line = f.readlines()
    f.close()
    return line

if __name__ == '__main__':
    #retrieve the file paths from the commandline arguments
    _, filename1, filename2 = sys.argv
    print("Number of arguments passed : ", len(sys.argv))

    # since we know the program takes two arguments
    print("First argument : ", filename1)
    print("Second argument : ", filename2)

    # print the files content
    file1content = read_file(filename1)
    print("\nContent of first file : ", file1content)
    file2content = read_file(filename2)
    print("\nContent of second file : ", file2content)

    q2()