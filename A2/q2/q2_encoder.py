# Written By : Wong Chee Hao
# Student ID : 32734751
# Date modified: 05/05/2024
import sys
from bitarray import bitarray
import heapq

##### Global variables
ASCII_START = 36
ASCII_END = 126
ASCII_RANGE = ASCII_END - ASCII_START + 1


                                ##### FROM Q1 #####

class Node:
    """
    Edge start and edge end is implemented here since the edge only carry these two values.

    Children is initialised as an array based on the ascii range of printable characters.
    """
    def __init__(self, start, end, is_leaf=False, j = None, suffix_link = None):
        
        # To check if a node is a leaf node
        self.is_leaf = is_leaf

        # Each leaf node carry the index of the extension that created it
        self.j = j

        # Each node's children is implemented with an array
        # WHEN DOING IN-ORDER TRAVERSAL, SHOULD TRAVERSE THE TERMINAL CHARACTER FIRST 
        self.children = [None] * ASCII_RANGE

        #  Space efficient representation of edge labels
        self.start = start

        # Space efficient representation of edge labels
        self.end = end

        # Suffix link
        self.suffix_link = suffix_link

class PointerEnd:
    """
    Since Python does not support pointers, glodbalEnd is implemented as a class to keep 
    track of endIndex for TRICK 1: RAPID LEAF EXTENSION
    """
    def __init__(self): 
        self.index = -1

    @property
    def get_index(self):
        return self.index

class Ukkonnen:
    """
    Ukkonnen class is used to build the suffix tree using Ukkonnen algorithm.

    The class has the following attributes:
    - text: The text to be inserted into the suffix tree.
    - root_node: The root node of the suffix tree.
    - active_node: The active node during the Ukkonnen algorithm.
    - active_length: The active length during the Ukkonnen algorithm.
    """
    def __init__(self, text=""):

        # Text to be inserted
        self.text = text

        # root_node of the suffix tree  
        self.root_node = Node(start = None, end = None, suffix_link=None)

        # Set the active node as the root_node
        self.active_node = self.root_node

        # Set the active length as 0
        self.active_length = 0

        # Set the suffix_link of root_node as itself
        self.root_node.suffix_link = self.root_node

        # Run Ukkonnen algorithm to build the suffix tree
        self.run_ukkonnen()

    def ukkonnen_traverse(self, end_index):
        """
        Traverse the suffix tree using a method that optimizes navigation by skipping unnecessary nodes.

        This function navigates the tree based on the active length and node, adjusting them as needed
        until it reaches the target node or the traversal criteria are no longer met (i.e., when the active node is a leaf or the active length is zero).

        Args:
        end_index (int): The current end index in the text being processed.

        Returns:
        Node: The last node reached during the traversal.
        """
        def navigate(node, length):
            """
            Auxiliary function to handle the traversal logic recursively.

            It checks the condition of the node (whether it's a leaf or the remaining length is zero),
            and moves through the tree by following the edges that match the current criteria of the traversal.

            Args:
            node (Node): The current node being navigated.
            length (int): The remaining length of the edge to navigate through.

            Returns:
            Node: Returns the node after the last edge that was fully navigated.
            """
            # If the node is a leaf or the length is zero, return the current node
            if node.is_leaf or length == 0:
                return node

            # Update active node for use in following recursive steps
            self.active_node = node

            # Update active length to the current length
            self.active_length = length

            # Fetch the edge that starts with the character at the calculated index
            next_node = node.children[ord(self.text[end_index - length]) - ASCII_START]

            # If the edge does not exist, return the current node
            if next_node is None:
                return node

            ## Calculate the length of the edge to determine if traversal should continue down this path
            # If the edge ends at a leaf, the length is the global end index
            if next_node.is_leaf:
                edge_span = next_node.end.get_index - next_node.start
            else:
                # Otherwise, calculate the length based on the edge's start and end indices
                edge_span = next_node.end - next_node.start

            # If the edge span is greater than or equal to the remaining length, return the current node
            if edge_span >= length:
                return node

            # Recursively navigate down the tree, adjusting the current length by the edge span
            return navigate(next_node, length - edge_span)

        # Initial call to the recursive navigate function
        return navigate(self.active_node, self.active_length)

    def update_suffix_link(self, non_leaf_node_last_extension, new_non_leaf_node):
        """
        Update the suffix link from the previous non-leaf node to the current non-leaf node.

        Args:
        non_leaf_node_last_extension (Node): The non-leaf node from the previous extension.
        new_non_leaf_node (Node): The current non-leaf node being linked to.

        Returns:
        Node: Returns the current non-leaf node to be used as the previous node in the next extension.
        """
        # If there is no previous non-leaf node, return the current non-leaf node
        if non_leaf_node_last_extension is None:
            return new_non_leaf_node
        
        # Otherwise, link the previous non-leaf node to the current non-leaf node
        else:
            # Link the previously created non-leaf node in tto the new non-leaf node
            non_leaf_node_last_extension.suffix_link = new_non_leaf_node

        # Return the current non-leaf node for use in the next extension
        return new_non_leaf_node
    
    def run_ukkonnen(self):
        """
        This function run the ukkonnen algorithm to build the suffix tree
        """
        # End pointer for TRICK 1: RAPID LEAF EXTENSION                      
        end_pointer = PointerEnd() 

        # Initialize phase and extension indices
        # Each new leaf node j, no need reset every phase since the TRICK 1: RAPID LEAF EXTENSION handles it
        i, j = 0, 0

        # For each phase
        for i in range(len(self.text) + 1):
            
            # Keep a variable of last non_leaf_node created in the same phase, 
            # so that we can link it to the next non_leaf_node created in the next extension
            non_leaf_node_last_extension = None   

            # Rule 1 extensions: Adjust the label of the edge to that leaf to account for the added character str[i+1]
            # IMPLEMENTATION: At the start of every phsae, Implicit extension 
            # This is covered by TRICK 1: RAPID LEAF EXTENSION
            end_pointer.index = end_pointer.get_index + 1      
            
            # For each suffix in the current phase
            # Check Explicit Extensions (Rule 2) / Suffix already exist in the tree (Rule 3)
            while j < i:

                # Reset active length if at root_node
                if self.active_node == self.root_node:
                    self.active_length = i - j

                # Traverse to find the extension point
                self.ukkonnen_traverse(i)   
                
                # Active edge
                active_edge = self.active_node.children[ord(self.text[i - self.active_length]) - ASCII_START]                     
                
                ### EXPLICIT EXTENSIONS START ###
                # Rule 2 extensions (case 1): The path end at a non-leaf node, ADD EDGE
                if active_edge is None:
                    
                    # Create a new leaf node
                    # Start is the constant i - self.active_length, end is the pointer end
                    # Set j as the payload of the leaf node
                    new_leaf_node = Node(i - self.active_length, end_pointer, is_leaf = True, j=j)     
                    
                    # Link active node to new leaf node 
                    self.active_node.children[ord(self.text[i - self.active_length]) - ASCII_START] = new_leaf_node

                # Rule 2 extensions(case 2): The path end at existing path, SPLIT EDGE
                elif self.text[i-1] != self.text[active_edge.start + self.active_length-1]:

                    # Create a new non-leaf node
                    # Whenever creating new non-leaf node, add a suffix link to root_node node
                    new_non_leaf_node = Node(active_edge.start, active_edge.start + self.active_length - 1, suffix_link = self.root_node) # Create new internal edge and node                                     
                    
                    ## Update Suffix Link ##
                    # Each time extend non-leaf node in the same phase, link last non-leaf node from previous extension to the current non-leaf node
                    non_leaf_node_last_extension = self.update_suffix_link(non_leaf_node_last_extension, new_non_leaf_node)
                    
                    # Link active node to new non leaf node
                    self.active_node.children[ord(self.text[i - self.active_length]) - ASCII_START] = new_non_leaf_node
                    
                    # Update original leaf node
                    active_edge.start = active_edge.start + self.active_length - 1 
                    
                    # Create a new leaf node
                    # Start is the constant i-1, end is the global end
                    # Set j as the payload of the leaf node
                    new_leaf_node = Node(i - 1, end_pointer, is_leaf=True, j=j)    
                    
                    # Link new non leaf node to original leaf node
                    new_non_leaf_node.children[ord(self.text[active_edge.start]) - ASCII_START] = active_edge
                    
                    # Link new non leaf node to new leaf node
                    new_non_leaf_node.children[ord(self.text[end_pointer.get_index-1]) - ASCII_START] = new_leaf_node
                

                # Rule 3 extension: The path already exist the tree, no further action is needed.
                else:

                    # SHOWSTOPPER RULE: If any extension j in phase i + 1 is performed using rule 3, 
                    # then immediately terminate the phase and begin next phase

                    # This rule break here so j won't be updated (correspond to last_j only be updated when rule 2 in the lecture notes)
                    # So next EXPLICIT extensions still begin from j
                    break                                                 
                
                # Speed traversal via suffix link                                                  
                self.active_node = self.active_node.suffix_link    
                
                # Update j, which is the beginning of the next EXPLICIT extensions if no existing suffix in the tree
                j = j + 1   

            self.active_length =  self.active_length + 1  
    
    def inorder_traversal(self, current_node, suffix_array = []):
        """
        Inorder traversal of the suffix tree 
        """

        if current_node is not None:
            # Traverse to each child node
            for child_node in current_node.children:
                # If the child node is not None, traversal to it
                if child_node is not None:
                    # Traverse to the child node
                    self.inorder_traversal(child_node, suffix_array)
            
            # If it's a leaf node, append the suffix index
            if current_node.is_leaf:
                # Append the suffix index to the suffix array
                suffix_array.append(current_node.j)

        return suffix_array
    
    def get_suffix_array(self):
        """
        get_suffix_array function returns the suffix array of the text by doing an inorder traversal of the suffix tree.
        
        This is lexographically sorted array of the suffixes of the text since the suffix tree is built using Ukkonnen algorithm.
        """
        suffix_array = []

        # Do inorder traversal from root_node to get suffix array 
        self.inorder_traversal(self.root_node, suffix_array)

        return suffix_array

######################## Q2 START ########################

def bwt_from_suffix_array(text, suffix_array):
    """
    This function takes a text and its suffix array and returns the BWT of the text.
    """
    bwt = ''
    for i in suffix_array:
        if i == 0:
            bwt += "$"
        else:
            bwt += text[i - 1]
    return bwt


def huffman_preprocessing(text):
    """
    Function to preprocess the text and calculate the frequency of each character in the text.

    Returns:
    - frequency_table: A list containing the frequency of each character in the text.
    - unique_character: A list of unique characters in the text.
    """
    # Initialize the frequency table
    frequency_table = [0] * ASCII_RANGE

    # Initialize the unique characters in the text
    unique_character = []

    # Calculate the frequency of each character in the text
    for char in text:
        index = ord(char) - ASCII_START
        if frequency_table[index] == 0:
            unique_character.append(char)
        frequency_table[index] += 1

    return frequency_table, unique_character

def huffman_heap_prepare(huffman_frequency_table, unique_character):
    """
    Function to prepare the Huffman heap by creating a heap array based on the frequency of characters.
    """
    # creating the heap array
    huffman_heap = []

    # loop for the number of distinct chars in string
    for index in range(len(unique_character)):

        # get the distinct char
        char = unique_character[index]

        # get frequency of the char
        frequency = huffman_frequency_table[ord(char) - ASCII_START]

        # heap table will contain the following (length of char + frequency of char, frequency of char, char)for each distinct char
        huffman_heap.append((frequency + 1, char, frequency))

    # heapify the char according to the key which is length of char + frequency of char
    heapq.heapify(huffman_heap)

    return huffman_heap

def huffman_heap_merge(huffman_heap, unique_character):
    """
    Function to merge the elements in the Huffman heap to build the Huffman codebook.
    """
    codeword_table = [None] * ASCII_RANGE

    # if the unique_character < 2 then we will only have one element in the heap
    if len(unique_character) < 2:
        codeword_table[ord(unique_character[0])] = bitarray()
        codeword_table[ord(unique_character[0])].append(0)

    # this will loop until we merge all the strings into one
    while len(huffman_heap) > 1:

        # pop the smallest element in the heap according to key
        first = heapq.heappop(huffman_heap)

        # pop the smallest element in the heap according to key
        second = heapq.heappop(huffman_heap)

        # accumulate the frequency+length of both strings getting combined
        total_sum = first[0] + second[0]

        # append both strings together
        new_str = first[1] + second[1]

        # push this new element in the heap
        heapq.heappush(huffman_heap, (total_sum, new_str, total_sum - len(new_str)))

        # the first pop string chars codeword will have 0 appended to it
        for char in first[1]:
            if codeword_table[ord(char) - ASCII_START] is None:

                # if the char is not in the codeword_table then we will create a new bitarray for it
                codeword_table[ord(char) - ASCII_START] = bitarray()
            # append 0 to the codeword_table
            codeword_table[ord(char) - ASCII_START].append(0)

        # the second pop string chars codeword will have 1 appended to it
        for char in second[1]:
            if codeword_table[ord(char) - ASCII_START] is None:
                # if the char is not in the codeword_table then we will create a new bitarray for it
                codeword_table[ord(char) - ASCII_START] = bitarray()
            # append 1 to the codeword_table
            codeword_table[ord(char) - ASCII_START].append(1)

    # return the codeword_table
    return codeword_table   

def huffman_build(string):
    """
    Function to build the Huffman codebook for the given string.
    """ 
    # Preprocess the string to get the frequency of each character and the unique characters
    huffman_frequency_table, unique_character = huffman_preprocessing(string)

    # Prepare the Huffman heap
    huffman_heap = huffman_heap_prepare(huffman_frequency_table, unique_character)

    # Merge the elements in the Huffman heap to build the Huffman codebook
    codeword_table = huffman_heap_merge(huffman_heap, unique_character)

    # Reverse the codewords since they are built in reverse order
    for char in unique_character:
        codeword_table[ord(char) - ASCII_START].reverse()

    # Return the unique characters and the Huffman codebook
    return unique_character, codeword_table

def huffman_encode(char, codebook):
    """ Returns the Huffman code for a character from the provided codebook. """
    return bitarray(codebook[ord(char) - ASCII_START])

def create_bitarray_from_int(number):
    """
    Transforms an integer into a bitarray representation. The bits are ordered
    from least significant to most significant. 
    """
    # Initialize an empty bitarray
    bits = bitarray()
    
    # Convert integer to binary
    while number > 0:
        remainder = number % 2
        number //= 2
        bits.append(remainder)

    # Return the result bitarray
    return bits

def elias_encoding(value):
    """
    Encodes a given integer using Elias gamma coding. This involves expressing
    the length of the binary form of the number as a binary number itself and
    then concatenating these lengths repeatedly.
    """
    # Start with converting the number to a bitarray
    main_bits = create_bitarray_from_int(value)
    
    # Calculate the length to pre-pend to the main bit array
    length = len(main_bits) - 1
    
    # Generate Elias encoding by appending bit lengths until the length is zero
    while length > 0:
        temp_bits = create_bitarray_from_int(length)  # Encode length with modified last bit
        temp_bits[-1] = 0
        main_bits += temp_bits                   # Concatenate to the main bit array
        length = len(temp_bits) - 1              # Update the length for the next iteration
    main_bits.reverse() 
    
    return main_bits

def encode_run_length(input_string):
    """
    Function to encode a string using run-length encoding.
    """
    # Initialize the list to store encoded tuples of characters and their counts.
    if not input_string:
        return []  # Return an empty list for empty input strings.

    encoded_data = []
    previous_char = input_string[0]  # Start with the first character.
    count = 1  # Initialize count for the first character.

    # Iterate over the string starting from the second character.
    for current_char in input_string[1:]:
        # Check if the current character matches the previous one.
        if current_char == previous_char:
            count += 1  # Increment count if the same character continues.
        else:
            # If a different character is found, append the previous character and its count.
            encoded_data.append((previous_char, count))
            previous_char = current_char  # Update the previous character.
            count = 1  # Reset count for the new character.

    # Append the last counted character and its count after exiting the loop.
    encoded_data.append((previous_char, count))

    return encoded_data

def encode_ascii(char):
    """
    Encodes a character to its binary ASCII representation.
    
    Parameters:
        char (str): A single character to be converted into its ASCII binary form.

    Returns:
        bitarray: A bitarray object containing the binary representation of the ASCII value of `char`.
    
    Example:
        encode_ascii('A') -> bitarray('1000001')
    """
    # Get the ASCII value of the character.
    ascii_value = ord(char)

    # Convert the ASCII value to a binary string, omitting the '0b' prefix.
    binary_string = bin(ascii_value)[2:]  # strip '0b'

    # Ensure the binary string is exactly 7 bits long by padding with zeros if necessary.
    # This is needed because ASCII values less than 64 (like space, which is 32) would
    # otherwise be represented with fewer than 7 bits.
    padded_binary_string = binary_string.zfill(7)

    # Convert the padded binary string to a bitarray and return.
    return bitarray(padded_binary_string)

def write_encoded_data_to_file(encoded_data, file_path):
    bits = bitarray(endian='big')  # Create an empty bitarray

    # Append bits for each part of the encoding schema
    # Elias Gamma Length
    bits.extend(encoded_data['elias_gamma_length'])

    # Number of Unique Characters
    bits.extend(encoded_data['elias_gamma_unique_chars'])

    # Encoding for each character
    for char, details in encoded_data['chars'].items():
        # ASCII code as bitarray
        # bits.frombytes(int(details['ascii'], 2).to_bytes(1, byteorder='big'))
        bits.extend(details['ascii'])
        
        # Elias Code for Huffman Code Length
        bits.extend(details['elias_codelen'])
        
        # Huffman Code
        bits.extend(details['huffman_code'])

    # Encoded BWT Data
    bits.extend(encoded_data['encoded_bwt'])

    # If the length of bits is not a multiple of 8, pad it
    if len(bits) % 8:
        bits.extend('0' * (8 - len(bits) % 8))

    # Write the bitarray to a binary file
    with open(file_path, 'wb') as file:
        bits.tofile(file)

def encode_data(input_data):
    """
    Function to encode the BWT data using Elias Gamma, Huffman, and Run-Length Encoding.
    """
    # Initialize the encoded data dictionary
    encoded_data = {
        'elias_gamma_length': None,
        'elias_gamma_unique_chars': None,
        'chars': {},
        'encoded_bwt': bitarray()
    }

    # Encode the length of the BWT using Elias Gamma
    bwt_length = len(input_data)
    encoded_data['elias_gamma_length'] = elias_encoding(bwt_length)

    # Build the Huffman codebook for the input data
    nUniqChars, huffman_codebook = huffman_build(input_data)
    # Encode the number of unique characters
    encoded_data['elias_gamma_unique_chars'] = elias_encoding(len(nUniqChars))

    # Encode each unique character's ASCII and Huffman coding information
    for char in nUniqChars: 

        # Encode the ASCII representation of the character
        ascii_encoded = encode_ascii(char)

        # Encode the length of the Huffman code using Elias Gamma
        elias_coded_huffman_length = elias_encoding(len(huffman_codebook[ord(char) - ASCII_START]))
        
        # Get the Huffman code for the character
        huffman_coded = huffman_codebook[ord(char) - ASCII_START]
        
        # Store the encoded data for the character
        encoded_data['chars'][char] = {
            'ascii': ascii_encoded,
            'elias_codelen': elias_coded_huffman_length,
            'huffman_code': huffman_coded
        }

    # Run-length encode the BWT
    run_length_encoded = encode_run_length(input_data)

    # Encode each tuple using Huffman for character and Elias for run length
    for char, run_len in run_length_encoded:
        encoded_data['encoded_bwt'].extend(huffman_encode(char, huffman_codebook))
        encoded_data['encoded_bwt'].extend(elias_encoding(run_len))

    return encoded_data

def q2():
    if len(sys.argv) != 2:
        print("Usage: python q1.py <text filename>")
        sys.exit(1)

    stringFileName= sys.argv[1]
    string = read_file(stringFileName)

    # Run the Ukkonnen algorithm to build the suffix tree
    ukkonnen_tree = Ukkonnen(string[0])

    # Get the suffix array from the suffix tree
    suffix_array = ukkonnen_tree.get_suffix_array()

    # Get the BWT from the suffix array
    bwt = bwt_from_suffix_array(string[0], suffix_array)

    # Encode the BWT data
    encoded_data = encode_data(bwt)

    write_encoded_data_to_file(encoded_data, "q2_encoder_output.bin")

# this function reads a file and return its content
def read_file(file_path: str) -> str:
    f = open(file_path, 'r')
    line = f.readlines()
    f.close()
    return line

if __name__ == '__main__':
    #retrieve the file paths from the commandline arguments
    _, filename1 = sys.argv
    print("Number of arguments passed : ", len(sys.argv))
    # since we know the program takes two arguments
    print("First argument : ", filename1)
    file1content = read_file(filename1)
    print("\nContent of first file : ", file1content)

    q2()