# Written By : Wong Chee Hao
# Student ID : 32734751
# Date modified: 04/05/2024
import sys

##### Global variables
ASCII_START = 36
ASCII_END = 126
ASCII_RANGE = ASCII_END - ASCII_START + 1

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
        def navigate(node, remaining_length):
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
            if remaining_length == 0 or node.is_leaf:
                return node

            # Update active length to the current length
            self.active_length = remaining_length
            # Update active node for use in following recursive steps
            self.active_node = node

            # Fetch the edge that starts with the character at the calculated index
            next_node = node.children[ord(self.text[end_index - remaining_length]) - ASCII_START]

            # If the edge does not exist, return the current node
            if next_node is None:
                return node

            ## Calculate the length of the edge to determine if traversal should continue down this path
            # If the edge ends at a leaf, the length is the global end index
            if next_node.is_leaf:
                edge_span = next_node.end.get_index - next_node.start

            # Otherwise, calculate the length based on the edge's start and end indices    
            else:
                edge_span = next_node.end - next_node.start

            # If the edge span is greater than or equal to the remaining length, return the current node
            if edge_span >= remaining_length:
                return node

            # Recursively navigate down the tree, adjusting the current length by the edge span
            return navigate(next_node, remaining_length - edge_span)

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

def increment_list(numbers):
    """
    Function to increment each element in a list by 1.
    """
    # Using a list comprehension to add 1 to each element
    return [x + 1 for x in numbers]

def find_index(numbers, target):
    """
    Returns the index of the target integer in the list if it is found.
    If the target is not found, it returns -1.
    
    :param numbers: List of integers where each integer is unique.
    :param target: Integer whose index needs to be found.
    :return: Index of the target integer or -1 if not found.
    """
    try:
        # Use the list.index() method which raises ValueError if the target is not found
        return numbers.index(target)
    except ValueError:
        # Return -1 if the target is not in the list
        return -1

def find_and_write_indices(targets, suffix_array, file_name):
    """
    Finds the indices of each target in the suffix_array and writes the indices to a file.
    Each index is written on a new line. If a target is not found, writes -1.

    :param targets: List of target integers to find in suffix_array.
    :param suffix_array: List from which indices are to be found.
    :param file_name: Name of the file to write the indices.
    """
    with open(file_name, "w") as file:
        for target in targets:
            try:
                # Try to find the index of the target in suffix_array 
                index = suffix_array.index(target) + 1 # (1-indexed)
            except ValueError:
                # If the target is not in the suffix_array, use -1
                index = -1
            # Write the index to the file followed by a newline
            file.write(f"{index}\n")

def read_integers_from_file(file_name):
    """
    Reads a file where each line contains an integer and returns a list of these integers.

    :param file_name: Name of the file to read from.
    :return: List of integers read from the file.
    """
    integers = []
    try:
        # Open the file in read mode
        with open(file_name, "r") as file:
            # Read each line in the file
            for line in file:
                # Convert the line to an integer and add to the list
                integers.append(int(line.strip()))
    except FileNotFoundError:
        print(f"Error: The file '{file_name}' does not exist.")
    except ValueError:
        print("Error: The file must contain only integers.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
    
    return integers

def q1():
    if len(sys.argv) != 3:
        print("Usage: python q1.py <text filename> <pattern filename>")
        sys.exit(1)

    stringFileName, positionsFileName = sys.argv[1], sys.argv[2]
    string = read_file(stringFileName)
    positions = read_integers_from_file(positionsFileName)

    ukkonnen_tree = Ukkonnen(string[0])
    suffix_array = ukkonnen_tree.get_suffix_array()
    suffix_array_1_index = increment_list(suffix_array)

    print("Suffix Array:", suffix_array_1_index)
    find_and_write_indices(positions, suffix_array_1_index, "output_q1.txt")

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
    file2content = read_integers_from_file(filename2)
    print("\nContent of second file : ", file2content)

    q1()