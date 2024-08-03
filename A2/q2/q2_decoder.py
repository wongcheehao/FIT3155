# Written By : Wong Chee Hao
# Student ID : 32734751
# Date modified: 05/05/2024
import sys
from bitarray import bitarray

##### Global variables
ASCII_START = 36
ASCII_END = 126
ASCII_RANGE = ASCII_END - ASCII_START + 1

def compute_order_and_rank(bwt_str):
    """
    Computes order and cumulative frequency (rank) for characters in the BWT string.
    
    Args:
    bwt_str (str): Burrows-Wheeler Transform of the original string.
    
    Returns:
    tuple: Contains the order of each character occurrence and cumulative rank.
    """
    # Initialize frequency and order counts
    char_freq = [0] * ASCII_RANGE
    char_order = [0] * len(bwt_str)

    # Calculate frequency of each character
    for char in bwt_str:
        char_freq[ord(char) - ASCII_START] += 1

    # Prepare cumulative frequency for rank calculation
    cumulative_freq = 0
    for i in range(len(char_freq)):
        temp = char_freq[i]
        char_freq[i] = cumulative_freq
        cumulative_freq += temp

    # Compute order of characters in BWT and adjust cumulative frequencies
    temp_count = [0] * ASCII_RANGE
    for idx, char in enumerate(bwt_str):
        char_index = ord(char) - ASCII_START
        temp_count[char_index] += 1
        char_order[idx] = temp_count[char_index]

    return char_order, char_freq

def rebuild_bwt_original_string(bwt_str):
    """
    Reconstructs the original string from its BWT representation using LF mapping.
    
    Args:
    bwt_str (str): Burrows-Wheeler Transform of the original string.
    
    Returns:
    str: The reconstructed original string.
    """
    char_order, cumulative_rank = compute_order_and_rank(bwt_str)
    reconstructed = []
    current_index = bwt_str.index('$')

    # Rebuild the original string from BWT
    for _ in range(len(bwt_str)):
        reconstructed.append(bwt_str[current_index])
        char_index = ord(bwt_str[current_index]) - ASCII_START
        current_index = cumulative_rank[char_index] + char_order[current_index] - 1

    # The original string is reversed
    return ''.join(reconstructed[::-1])

def bitarr_to_int(ba):
    """
    Convert a bitarr to an integer.

    Args:
    ba (bitarr): The bitarr to convert.

    Returns:
    int: The integer representation of the bitarr.
    """
    # Initialize the result integer
    result = 0

    # Iterate over each bit in the bitarr
    for bit in ba:
        # Shift the current result left by one bit to make room for the next bit
        result = (result << 1) | bit

    return result

def decode_elias_gamma(encoded_bits, index):
    """
    Decodes an Elias gamma encoded bitarr.

    Args:
    encoded_bits (bitarr): The Elias gamma encoded bitarr.
    index (int): The starting index to decode the Elias gamma code.

    Returns:
    tuple: A tuple containing the decoded number and the next index to start decoding.
    """
    i = index
    decoded_numbers = []
    # Length to be extracted
    length = 1

    while i < len(encoded_bits):
        # Extract the bits of given length
        number_bits = bitarray()

        while i < len(encoded_bits) and length > 0:
            number_bits.append(encoded_bits[i])
            length -= 1 
            i += 1 # move to the next bit
    
        # The length obtained is the number of bits for the next number
        if i + length > len(encoded_bits):
            raise ValueError("Invalid encoded data or incomplete data.")
        
        if number_bits[0] == 1: # If it is not length
            decoded_number = bitarr_to_int(number_bits)
            decoded_numbers.append(decoded_number)
            length = 1 # reset the length to 1 for next number

            return decoded_number, i
        
        else: # If it is length
            number_bits[0] = 1 # flip the fist bit to 1
            decoded_number = bitarr_to_int(number_bits)
            length = decoded_number + 1 # next length is the decoded number + 1

    # return decoded_numbers

class NodeHuffman:
    """
    Node class for Huffman tree.

    Attributes:
    children (list): List of children nodes.
    char (str): The character stored in the node.
    is_leaf (bool): Flag to indicate if the node is a leaf.
    """
    def __init__(self, char = None):
        self.children = [None] * 2  # 0 = left, 1 = right
        self.char = char            # val = bit value otherwise string char
        self.is_leaf = False   

    # Insert a child node with the given bit if None
    def insertBit(self, child, bit):
        if self.children[bit] is None:
            self.children[bit] = child

    # Function to store the character in the leaf node
    def addChar(self, char):
        self.char = char

class HuffmanTree:
    """
    class to build a Huffman tree from the unique characters and their codes.

    Attributes:
    unique_char (list): List of unique characters.
    code_table (list): List of Huffman codes for each character.
    root (NodeHuffman): The root node of the Huffman tree.
    """
    def __init__(self, unique_char: list[str], code_table: list[bitarray]):
        self.root = NodeHuffman()
        self.unique_char = unique_char
        self.code_table = code_table

        # Build the Huffman tree
        self.huffman_build_tree()

    def huffman_build_tree(self):
        """
        fUNCTION TO BUILD THE HUFFMAN TREE FROM THE UNIQUE CHARACTERS AND THEIR CODES.
        """
        for char in self.unique_char:
            
            # Build the tree by traversing the code table for each character
            current_node = self.root                      
            
            for bit in self.code_table[ord(char) - ASCII_START]:

                # Insert bit 
                current_node.insertBit(NodeHuffman(), bit)        
                
                # Move to the next node
                current_node = current_node.children[bit]              
            
            # Add the character to the leaf node
            current_node.addChar(char)   
            current_node.is_leaf = True 

    def huffman_decode_rebuild_string(self, bitarr):
        """
        FUNCTION TO DECODE THE BITARR USING THE HUFFMAN TREE
        """
        string = ""

        # Traverse the tree from root to decode the bitarr
        current_node = self.root                            
        
        for bit in bitarr:
            # Append the character if the current node is a leaf
            if current_node.is_leaf:          
                # Append the character to the string                       
                string += current_node.char

                # Reset the current node to the root
                current_node = self.root.children[bit]
            else:
                # Move to the next node
                current_node = current_node.children[bit] # GET NEXT NODE

        return string
    
def huffman_decode(bitarr, code_table, unique_char):
    """
    FUNCTION TO DECODE THE BITARR USING THE HUFFMAN TREE
    """
    T = HuffmanTree(unique_char, code_table) 
    decoded_string = T.huffman_decode_rebuild_string(bitarr)

    return decoded_string

def read_binary_file_to_bitarr(file_path):
    """
    Reads a binary file and returns its contents as a bitarr
    """
    with open(file_path, "rb") as file:
        byte_data = file.read()
    # Create a bitarr and fill it with the contents of byte_data
    bits = bitarray()
    bits.frombytes(byte_data)
    return bits

def decode_header(bitarr):
    """
    Decodes the header to extract length and unique character count.

    Returns:
    tuple: A tuple containing the remaining bitarr and the extracted values.
    """

    index = 0

    # Decode Elias code for bwt_length 
    bwt_length, index = decode_elias_gamma(bitarr, index)

    # Decode Elias code for number of unique characters
    num_unique_chars, index = decode_elias_gamma(bitarr, index)

    return bitarr[index:], bwt_length, num_unique_chars

def decode_unique_chars_info(bitarr, num_unique_chars):
    """
    Decodes the unique character information from the bit array.
    """
    index = 0
    huffman_table = {}

    for _ in range(num_unique_chars):
        
        # Convert bitarr slice to integer for ASCII character
        char_bits = bitarr[index:index+7]  # Get 7 bits for ASCII
        char = chr(bitarr_to_int(char_bits))  # Convert bitarr to integer and then to char
        index += 7

        # Decode the codelen for the character
        code_length, new_index = decode_elias_gamma(bitarr, index)
        index = new_index

        # Decode the Huffman code for the character
        huffman_code = bitarr[index:index+code_length]
        index += code_length

        # # Decode the Huffman code for the character
        # Store the character and its Huffman code in the table
        huffman_table[char] = huffman_code

    return bitarr[index:], huffman_table

def build_bwt_from_decoded(decoded_tuples):
    """
    Reconstruct the BWT string from a list of tuples containing characters and their run lengths.
    
    Args:
    decoded_tuples (list of tuples): Each tuple contains a character and its run length.
    
    Returns:
    str: The reconstructed BWT string.
    """
    bwt_string = ""
    for char, run_length in decoded_tuples:
        bwt_string += char * run_length  # Repeat character by its run length and append to the result string
    return bwt_string

def decode_data(bitarr, huffman_dict, bwt_length):
    """
    Function to decode the BWT data using the Huffman dictionary.
    """
    # Initialize variables
    decoded_bwt = []
    index = 0
    num_symbols_decoded = 0

    # Convert Huffman codes in dictionary to bitarr for easier comparison
    huffman_bitarrs = {}
    # Convert the Huffman codes to bitarrays
    for char, code in huffman_dict.items():
        code_ba = bitarray()
        code_ba.extend(code)  # Convert the code to bitarray

        # Store the Huffman code bitarray for the character
        huffman_bitarrs[char] = code_ba

    # Decode the BWT data using the Huffman dictionary
    while index < len(bitarr) and num_symbols_decoded < bwt_length:
        found = False

        # Iterate over the Huffman codes to find a match
        for char, code_ba in huffman_bitarrs.items():

            # Check if the current slice of bitarr matches the Huffman code
            code_len = len(code_ba)
            if bitarr[index:index + code_len] == code_ba:  # Compare slice of bitarr to Huffman code bitarr
                # Update the index by the code_len
                index += code_len

                run_length, new_index = decode_elias_gamma(bitarr, index)
                index = new_index

                # Append the decoded character and its run length to the list
                decoded_bwt.append((char, run_length))
                
                # Update the number of symbols decoded
                num_symbols_decoded += run_length
                found = True
                break
        if not found:
            raise ValueError("Invalid Huffman code encountered or improper alignment in bitarr")

    if num_symbols_decoded != bwt_length:
        raise ValueError("The decoded BWT does not match the expected length.")
    
    return decoded_bwt

def decode_bwt_file(file_path):
    """
    Function to decode the binary file and reconstruct the original text.
    """
    # Read the binary file to get the bit array
    bitarr = read_binary_file_to_bitarr(file_path)

    # Decode the header to understand how to process the rest
    bitarr, bwt_length, num_unique_chars = decode_header(bitarr)

    # Decode the information about unique characters and their codes
    bitarr, huffman_table = decode_unique_chars_info(bitarr, num_unique_chars)

    # Decode the actual data part using the Huffman table
    decoded_tuples = decode_data(bitarr, huffman_table, bwt_length)

    # Reconstruct the BWT string from the decoded tuples
    bwt = build_bwt_from_decoded(decoded_tuples)

    # Rebuild the original string from the BWT
    original_text = rebuild_bwt_original_string(bwt)
    
    # Return the original text
    return original_text

def q2():
    original_text = decode_bwt_file("q2_encoder_output.bin")

    with open("q2_decoder_output.txt", "w") as f:
        f.write(f"{original_text}")

if __name__ == '__main__':
    q2()

