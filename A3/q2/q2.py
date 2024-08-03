# Written By : Wong Chee Hao
# Student ID : 32734751
# Date modified: 25/05/2024
class Node:
    def __init__(self, t, leaf=True):
        # leaf node or not
        self.leaf = leaf

        # items in the node
        self.items = []

        # links to the children nodes
        self.children = []

        # t is the minimum degree of the B-tree
        self.t = t

    def split(self, parent, payload):
        new_node = Node(self.t, self.leaf)

        # find the mid point of the node, size is odd number
        mid_point = self.size // 2

        split_value = self.items[mid_point]
        
        # insert the mid point item into the parent node
        parent.items.insert(payload, split_value)
        
        # link the parent to the new node
        parent.children.insert(payload + 1, new_node)
        
        # SET the second half of the items into the new node
        new_node.items = self.items[mid_point + 1:]
        
        # SET the first half of the items as the current node items
        self.items = self.items[:mid_point]

        # if the node is not a leaf node, move the children to the new node
        if not self.leaf:
            
            # SET the second half of the children into the new node
            new_node.children = self.children[mid_point + 1:]
            
            # SET the first half of the children as the current node children
            self.children = self.children[:mid_point + 1]

    @property
    def size(self):
        """
        Return the number of items in the node
        """
        return len(self.items)

class BTree:
    def __init__(self, t):
        self.root = Node(t)

        ## Minimum degree of the B-tree
        self.t = t

    def insert(self, word):
        root = self.root
        
        ## If root is full, split the root
        if root.size == (2 * self.t) - 1:
            ## Create a new root
            new_root = Node(self.t, leaf=False)

            ## Set the node to be split as the child of the new root
            new_root.children.append(self.root)

            ## Set the new root as the root of the tree
            self.root = new_root

            ## Split the old root
            root.split(new_root, 0)

            ## Insert the word into the appropriate child
            self._insert_non_full(new_root, word)
        else:
            self._insert_non_full(root, word)

    def _insert_non_full(self, node, word):
        
        ## If the node is a leaf, insert the word into the node
        if node.leaf:
            if word not in node.items:
                # Add a placeholder for the new word
                node.items.append(None)
                
                # Find the position to insert the word and shift items to the right
                i = len(node.items) - 2  # Start from the second last item (last item is the placeholder)
                while i >= 0 and word < node.items[i]:
                    node.items[i + 1] = node.items[i]  # Move the item one position to the right
                    i -= 1
                
                # Insert the word at the correct position
                node.items[i + 1] = word

        else:
            i = len(node.items) - 1

            ## Find the child to descend into
            while i >= 0 and word < node.items[i]:
                i -= 1

            i += 1

            ## If the child is full, split the child
            if len(node.children[i].items) == (2 * self.t) - 1:
                node.children[i].split(node, i)

                ## Determine which of the two children to descend into
                if word > node.items[i]:
                    i += 1
            
            ## Recursively insert the word into the child
            self._insert_non_full(node.children[i], word)

    def delete(self, word):
        self._delete(self.root, word)

    def _delete(self, node, word):
        t = self.t

        ## Case 1: : If x belongs to a leaf node with strictly more than minimum number of elements (> t − 1) elements (root exempted from this rule)
        if node.leaf and node.size > t - 1 or node.leaf and node == self.root:
            if word in node.items:
                node.items.remove(word)
                return
        else:
            for i, item in enumerate(node.items):
                
                ## Case 2:If x belongs to an internal node:
                if word == item:
                    
                    if node.children:
                        
                        ##  Case 2a: If the left child node has at least t items, find the predecessor
                        if node.children[i].size >= t:
                            
                            ## Find the predecessor
                            predecessor = self._get_predecessor(node, i)

                            ## Replace the item with the predecessor
                            node.items[i] = predecessor

                            ## Recursively delete the predecessor from the left child node
                            self._delete(node.children[i], predecessor)

                        ## Case 2b: If the right child node has at least t items, find the successor
                        elif node.children[i + 1].size >= t:
                            
                            ## Find the successor
                            successor = self._get_successor(node, i)
                            
                            ## Replace the item with the successor
                            node.items[i] = successor

                            ## Recursively delete the successor from the right child node
                            self._delete(node.children[i + 1], successor)

                        ## Case 2c: If both children have t - 1 items, merge the children
                        else:
                            self._merge(node, i)

                            ## Recursively delete the word from the merged child
                            self._delete(node.children[i], word)

                    return
                
                ## If the word in the subtree
                elif word < item:
                    ## Case 3:  the traversal is stopped because the ‘appropriate’ subtree containing x has a node with exactly t − 1 elements
                    if node.children[i].size < t:
                        self._fill(node, i)

                    ## Recursively delete the word from the appropriate child
                    self._delete(node.children[i], word)
                    return
            
            if node.children:
                ## Handle the case where the word is in the subtree of the last child
                ## Case 3:  the traversal is stopped because the ‘appropriate’ subtree containing x has a node with exactly t − 1 elements
                if node.children[-1].size < t:
                    self._fill(node, len(node.items))

                ## Recursively delete the word from the appropriate child
                self._delete(node.children[-1], word)
    
    def _get_predecessor(self, node, idx):
        """
        This function returns the predecessor of the item at the given index
        """

        ## Left child node
        current = node.children[idx]

        ## Traverse to the rightmost child of the current node
        while not current.leaf:
            current = current.children[-1]
        
        ## Return the rightmost item of the current node
        return current.items[-1]

    def _get_successor(self, node, idx):
        
        ## Right child node
        current = node.children[idx + 1]

        ## Traverse to the leftmost child of the current node
        while not current.leaf:
            current = current.children[0]
        
        ## Return the leftmost item of the current node
        return current.items[0]

    def _merge(self, node, idx):
        
        ## Left child node
        child = node.children[idx]

        ## Right child node
        sibling = node.children[idx + 1]

        ## Move down the item from the current node to the left child node
        child.items.append(node.items[idx])

        ## Move all items from the right child node to the left child node
        child.items.extend(sibling.items)

        ## If the left child is not a leaf, move all children from the right child node to the left child node
        if not child.leaf:
            child.children.extend(sibling.children)

        ## Remove the item from the current node
        node.items.pop(idx)

        ## Delete the right child node
        node.children.pop(idx + 1)

    def _fill(self, node, idx):
        t = self.t

        ## Case 3a.1: If the left sibling has more than t - 1 items, borrow an item from the left sibling
        # if idx == 0, there is no left sibling
        if idx != 0 and node.children[idx - 1].size >= t:
            self._borrow_from_left(node, idx)
        
        ## Case 3a.2: Elif the right sibling has more than t - 1 items, borrow an item from the right sibling
        # if idx == len(node.items), there is no right sibling
        elif idx != len(node.items) and node.children[idx + 1].size >= t:
            self._borrow_from_right(node, idx)

        ## Case 3b: Else, merge the child with one of its siblings
        else:
            
            ## Merge with the right sibling if it exists
            if idx != len(node.items):
                self._merge(node, idx)
            
            ## Merge with the left sibling
            else:
                self._merge(node, idx - 1)

    def _borrow_from_left(self, node, idx):
        
        ## Child node borrowing the item
        child = node.children[idx]

        ## Child node's left sibling
        sibling = node.children[idx - 1]

        ## Move down the item from the current node to the child node
        child.items.insert(0, node.items[idx - 1])

        ## Handle the children of the sibling node
        if not child.leaf:
            
            ## Move the last child of the sibling node to the child node
            child.children.insert(0, sibling.children.pop())
        
        ## Move the last item of the sibling node to the current node
        node.items[idx - 1] = sibling.items.pop()

    def _borrow_from_right(self, node, idx):
        
        ## Child node borrowing the item
        child = node.children[idx]

        ## Child node's right sibling
        sibling = node.children[idx + 1]

        ## Move down the item from the current node to the child node
        child.items.append(node.items[idx])

        ## Handle the children nodes of the sibling node
        if not child.leaf:
            
            ## Move the first child of the sibling node to the child node
            child.children.append(sibling.children.pop(0))

        ## Move the first item of the sibling node to the current node
        node.items[idx] = sibling.items.pop(0)

    def traverse(self):
        """
        In-order traverse the B-tree and return a list of sorted words
        """
        def _traverse(node):
            if node:
                for i in range(len(node.items)):
                    ## Recursively traverse the left child
                    if not node.leaf:
                        _traverse(node.children[i])
                    
                    ## Append the item to the result list
                    result.append(node.items[i])

                ## Handle the rightmost child
                if not node.leaf:
                    _traverse(node.children[len(node.items)])
        
        result = []

        ## Start traversing from the root node
        _traverse(self.root)
        return result

def main():
    import sys

    # Read command line arguments
    t = int(sys.argv[1])
    dictionary_file = sys.argv[2]
    commands_file = sys.argv[3]

    # Create a B-tree instance
    btree = BTree(t)

    # Read words from dictionary file and insert into B-tree
    with open(dictionary_file, 'r') as f:
        for line in f:
            word = line.strip()
            btree.insert(word)

    # Execute commands from commands file
    with open(commands_file, 'r') as f:
        for line in f:
            parts = line.strip().split(' ', 2)
            command = parts[0]
            word = parts[1]
            if command == 'insert':
                btree.insert(word)
            elif command == 'delete':
                btree.delete(word)

    # Traverse the final state of the B-tree and output sorted words
    words = list(btree.traverse())
    with open('output_q2.txt', 'w') as f:
        for i, word in enumerate(words):
            if i < len(words) - 1:
                f.write(word + '\n')
            else:
                f.write(word)


if __name__ == '__main__':
    main()