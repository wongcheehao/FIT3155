import sys

##### Global variables
# Text and patterns are printable characters whose ASCII
# values are in the range [33, 126] (both inclusive).
ascii_start = 33
ascii_end = 126
ascii_range = ascii_end - ascii_start + 1

def Z_algo(s: str):
    """
    Implements Gusfield's Z-Algorithm to compute the z-values of a given string,
    z-values are the length of the longest substring starting from index i 
    that matches the prefix of the string.

    Args:
        string (str): The string to compute z-values for

    Returns:
        List[int]: A list of length len(string), with each index i corresponding to
                the z_i-value of the input string. First index is always None
    """
    
    n = len(s)
    if n == 0: 
        return []  # Handle the case when the string is empty
    Z = [0] * n
    Z[0] = None  # Z[0] is always the length of the string 

    l, r = 0, 0  # Initialize the left and right boundaries of the Z-box

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

def rightToLeft_Z_algo(s: str):
    """
    Implements Gusfield's Z-Algorithm in reverse similar to Z_algo
    By running the Z-Algorithm from right to left, we are able to compute the z-values,
    where the z-value is the length of the longest substring ending at position i of pat 
    that matches its suffix
    """
    
    n = len(s)
    if n == 0: 
        return []  # Handle the case when the string is empty
    Z = [0] * n
    Z[-1] = None  # Z[-1] is always None

    l, r = n - 1, n - 1  # Initialize the rightmost Z-box

    for k in range(n - 2, -1, -1):
        # Case 1, k is outside the Z-box
        if k < r:  
            n_matches = n-1
            counter = 0

            while k - counter >= 0 and s[n_matches] == s[k - counter]:
                n_matches -= 1
                counter += 1
                if n_matches == 0 or k - counter < 0:
                    break

            # Update the Z value
            Z[k] = counter

            # if n_matches > 0:
            l = k
            r = k - counter + 1

        else:  # Case 2
            k_prime = n - 1 - (l - k) # k' = k - l, Python starts from index 0
            remaining = k - r + 1

            if Z[k_prime] < remaining:  # Case 2a
                Z[k] = Z[k_prime]

            elif Z[k_prime] == remaining:  # Case 2b
                n_matches = n - 1 - remaining # the matches we already know from Z-box
                counter = remaining

                if k - counter >= 0:
                    while s[k-counter] == s[n_matches]:
                        n_matches -= 1
                        counter += 1
                        if n_matches == 0 or k - counter < 0:
                            break

                # Update the Z value = n_matches 
                Z[k] = counter 

                # if n_matches > remaining:  # Update the l and r of Z-box if we found more matches outside from the Z-box
                l = k
                r = k + counter - 1

            else:  # Case 2c
                Z[k] = remaining

    return Z

def good_suffix(pat: str):
    """
    This function calculates the good suffix values defined by the following formula:
        j := m - Zˆ{suffix}_p + 1
        goodsuffix(j) := p
    """
    m = len(pat)

    # Handle the case when the pat is empty
    if m == 0:
        return []
    
    # initialize the good suffix array to 0
    gs = [0] * (m + 1)

    for i in range(m - 1):
        j = m - rightToLeft_Z_algo(pat)[i] + 1 # using rightToLeft_Z_algo to find the Z^suffix_p
        gs[j-1] = i+1 # Shift by one for 0-based indexing
    
    return gs

def matched_prefix(pat: str):
    """
        This function computes the longest suffix that matches the prefix of the pattern.
    """
    m = len(pat)

    if m == 0:
        return []
    
    MP = [0] * (m+1)
    
    # Get the z array to know of the longest matched prefix
    z = Z_algo(pat)

    # Start from the end of the pattern, because we want to know the longest matched prefix
    for i in range(m - 1, 0, -1):

        # if Z value + the index != m, meaning the substring is not a suffix of the pattern
        if z[i] + i != m:

            MP[i] = MP[i+1] # the machted prefix is the same as the right of it

        # if Z value + the index == m, meaning the substring is a suffix of the pattern
        else: 
            MP[i] = z[i] # the matched prefix is the z value

    return MP

def reversedBM_extended_bad_character_preprocessing(pattern):
    """
    Preprocesses the pattern for the extended bad character rule used in the reversed
    Boyer-Moore string matching, by creating a 2D array representing the leftmost
    position of each character in the pattern at each index.

    :param pattern: The pattern string for which preprocessing is to be done.
    :return: A 2D list where each list represents the leftmost
    position for each character at each position.
    """
    m = len(pattern)

    # Initialize the Extended Bad Character table
    R = [[-1 for _ in range(m)] for _ in range(ascii_range)]

    # for each index in the pattern, update the table[char_pos][i]
    for i in range(m):
        char = pattern[i]
        char_pos = ord(char) - ascii_start

        # Update the table from the current position to the left, if not already updated
        # This works because we find the leftmost position of each character in the pattern
        for j in range(i, -1, -1):
            if R[char_pos][j] >= 0:
                break  # Stop if already updated for this position
            else:
                R[char_pos][j] = i # Update the leftmost position of the character if not already updated

    return R

def bad_character_rule(m, k, mismatched_character, extended_bad_character_array):
    """
    Given the index j of the mismatched character and the mismatched character itself, 
    calculate the shift distance using the bad character rule.
    """
    bc_shift = 0
    ascii_start = 33
    char_pos = ord(mismatched_character) - ascii_start

    # The character occur in the pattern
    if extended_bad_character_array[char_pos][k] != -1:
        
        # R(x) - k
        bc_shift = extended_bad_character_array[char_pos][k] - k

    # The mismatched character does not exist in the pattern
    else:
        bc_shift = m - k #shift the whole pattern pass the mismatch

    return bc_shift

def good_suffix_rule(m, k, matched_prefix_array, good_suffix_array):

    """
    Given the index k of the mismatched character, compute the shift distance using the good suffix rule.
    """
    gs_shift = 0   
    
    # Case 1: GS[k-1] > 0
    # We want the good suffix value of (k-1), which is good_suffix_array[k]
    if good_suffix_array[k] > 0: 
        gs_shift = m - good_suffix_array[k]
        stop = m - good_suffix_array[k] # Galil’s optimisation, stop at the good suffix(prefix) value
        resume = stop + k - 1 # Galil’s optimisation, resume at the end of the good suffix(prefix) value
    # Case 2: GS[k-1] == 0
    elif good_suffix_array[k] == 0:
        gs_shift = m - matched_prefix_array[k]
        stop = m - matched_prefix_array[k] # Galil’s optimisation, stop at the good suffix(prefix) value
        resume = m - 1# Galil’s optimisation, resume at the end of the good suffix(prefix) value
    
    return gs_shift, stop , resume

def RightToLeft_Boyer_Moore(pattern, text):
    """

                            RightToLeft_Boyer_Moore
                        /                           |
                    mismatch at k                  matched all
                    /              |                        |
                    BC                  GS                  m - Matched Prefix[-2]
                /      |           /          |
            R(x) - k    m-k  GS[k-1] == 0   GS[k-1] > 0
                                /               |
                          m - MP[k]          m - GS[k]
   
    """

    ##### STEP - 1 RightToLeft_Boyer_Moore preprocessing

    ## STEP - 1.1 Extended bad character preprocessing
    """
        R(x) =  Position of the leftmost occurrence of x in the pattern 
            if x occurs in pattern,
            
            -1  otherwise
    
    Since left-to-right scanning, we should save the leftmost occurence of x 
    instead of the rightmost, this make sure it is safe to shift the pattern to the left 
    by , R(x) - k places. 

    If the character x does not occur in the pattern, it is safe to shift the pattern to the left 
    by , m - k places. 
    
    """
    reversed_pattern = pattern[::-1]
    extended_bad_character_array = reversedBM_extended_bad_character_preprocessing(pattern)

    ## STEP - 1.2 Good suffix(prefix) preprocessing
    """
    In left-to-right scanning BM, we want to find the leftmost occurrence of the substring
    that matches the prefix. 

    EXAMPLE:
                                                        |
                                                        |
                                                    MISMATCH
                                       [Substring A][Character X] [Substring B][Character Y] [Substring C][Character Z]
                                        
                =====> SAFE SHIFT TO MATCH THE PREFIX <=====
    
            [Substring A][Character X] [Substring B][Character Y] [Substring C][Character Z] 


    To achieve this, we can use the concept introduced in the lecture.

    EXAMPLE:
            ORIGINAL PATTERN:
                                |
                                |
                            MISMATCH
            [Substring A][Character X] [Substring B][Character Y] [Substring C][Character Z]

            REVERSED PATTERN: 
                                                                        |
                                                                        |
                                                                    MISMATCH
            [Character Z][Substring C]  [Character Y][Substring B] [Character X][Substring A]
            
                            =====> SAFE SHIFT TO MATCH THE SUFFIX <=====

                                        [Character Z][Substring C] [Character Y][Substring B] [Character X][Substring A]  


    This is the exactly the concept of good suffix introduced in the lecture.

    Hence, we can use the good suffix preprocessing on reversed pattern to find 
    the leftmost occurrence of the substring that matches the prefix.

    BASED ON THIS CONCEPT, HERE IS THE STEPS TO IMPLEMENT GOOD SUFFIX (PREFIX) PREPROCESSING:
    1. Reverse the pattern
    2. Compute the good suffix array (using reverse GUESFILED'S Z-ALGORITHM)
    3. Reverse the good suffix array
    """
    good_suffix_array = good_suffix(reversed_pattern)
    good_suffix_array = good_suffix_array[::-1]

    ## STEP - 1.3 Matched prefix(suffix) preprocessing
    """
    In left-to-right scanning BM, we want to find the largest prefix of pat[1.. k -1] that is identical 
    to the suffix of pat[k ..m]. 

    Similar to the Step - 1.2 Good suffix(prefix) preprocessing, we can use the reversed pattern to do the pre-processing.

    HERE IS THE STEPS TO IMPLEMENT GOOD SUFFIX (PREFIX) PREPROCESSING:
    1. Use the reversed pattern to calculate matchedprefix(k + 1) which denotes the length of the largest 
    suffix of pat[k + 1..m] that is identical to the prefix of pat[1..m - k].
    2. Reverse the matchedprefix array
    """
    matched_prefix_array = matched_prefix(reversed_pattern)
    matched_prefix_array = matched_prefix_array[::-1]

    ##### STEP - 2 RightToLeft_Boyer_Moore search phase
    ## STEP - 2.1 Compare the current alignment in a left-to-right scan, 
    ## applying Galil’s optimisation to terminate the scan prematurely if appropriate.
    
    ## Galil's optimisation
    ## 1 -> stop   
    ## resume -> M
    
    positions = []

    m, n = len(pattern), len(text)

    start = n - m
    stop = -1 
    resume = -1
    bc_shift = 1
    gs_shift = 1
    while start >= 0: 
        k = 0
        while k < m:
            if k == stop: # Galil’s optimisation, stop scanning
                k = resume + 1# Continue scanning from the resume position
                continue 
            if pattern[k] != text[start + k]: # if there is a mismatch
                break
            k += 1

        # Full match
        if k == m: 
            # append to the return positions
            positions.append(start + 1)

            # when there is at least 2 characters in the pattern
            if len(matched_prefix_array) >= 3:
                # we want to find the largest suffix that match the rest of the pattern 
                shift = m - matched_prefix_array[-2] 
                stop = m - matched_prefix_array[-2] # Galil’s optimisation, stop at the matched_prefix_array(prefix) value
                resume = m - 1# Galil’s optimisation, resume at the end of the matched_prefix_array(prefix) value
            
            else: # there is only one character in the pattern
                # suffix impossible to match the rest of the pattern because there is only one character
                shift = 1
                stop = -1 
                resume = -1

        # There is a mismatch
        if k < m:
            bc_shift = bad_character_rule(m, k, text[start + k], extended_bad_character_array)
            gs_shift, gs_stop, gs_resume = good_suffix_rule(m, k, matched_prefix_array, good_suffix_array) 

            if gs_shift > bc_shift:
                shift = gs_shift
                stop = gs_stop
                resume = gs_resume
            else:
                shift = bc_shift
                stop = -1 
                resume = -1
            shift = max(shift,1)

        start -= shift

    print(positions)
    return positions

def q1():
    if len(sys.argv) != 3:
        print("Usage: python q1.py <text filename> <pattern filename>")
        sys.exit(1)

    text_filename, pattern_filename = sys.argv[1], sys.argv[2]
    text = read_file(text_filename)
    pattern = read_file(pattern_filename)

    text = text[0].strip()
    for pat in pattern:
        positions = RightToLeft_Boyer_Moore(pat,text)

        with open("output_q1.txt", "w") as f:
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
    file1content = read_file(filename1)
    print("\nContent of first file : ", file1content)
    file2content = read_file(filename2)
    print("\nContent of second file : ", file2content)
    q1()