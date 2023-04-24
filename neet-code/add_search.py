class TrieNode:
    def __init__(self):
        self.children = {}  # a : TrieNode
        self.word = False


class WordDictionary:
    def __init__(self):
        self.root = TrieNode()

    def addWord(self, word: str) -> None:
        cur = self.root
        for c in word:
            if c not in cur.children:
                cur.children[c] = TrieNode()
            cur = cur.children[c]
        cur.word = True

    def search(self, word: str) -> bool:
        def dfs(j, root):
            cur = root

            for i in range(j, len(word)):
                c = word[i]
                if c == ".":
                    for child in cur.children.values():
                        if dfs(i + 1, child):
                            return True
                    return False
                else:
                    if c not in cur.children:
                        return False
                    cur = cur.children[c]
            return cur.word

        return dfs(0, self.root)

"""
Trie

# Grokking
class TrieNode():
  def __init__(self):
    self.nodes = []
    self.complete = False
    for i in range (0, 26):
      self.nodes.append(None)

class WordDictionary:
    # initialise the root with trie_node() and set the can_find boolean to False
    def __init__(self):
        self.root = TrieNode()
        self.can_find = False

    # get all words in the dictionary
    def get_words(self):
        wordsList = []
        # return empty list if there root is NULL
        if not self.root:
            return []
        # else perform depth first search on the trie
        return self.dfs(self.root, "", wordsList)

    def dfs(self, node, word, wordsList):
        # if the node is NULL, return the wordsList
        if not node:
            return wordsList
        # if the word is complete, add it to the wordsList
        if node.complete:
            wordsList.append(word)

        for j in range(ord('a'), ord('z')+1):
            n = word + chr(j)
            wordsList = self.dfs(node.nodes[j - ord('a')], n, wordsList)
        return wordsList

    # adding a new word to the dictionary
    def add_word(self, word):
        n = len(word)
        cur_node = self.root
        for i, val in enumerate(word):
            # place the character at its correct index in the nodes list
            index = ord(val) - ord('a')
            if cur_node.nodes[index] is None:
                cur_node.nodes[index] = TrieNode()
            cur_node = cur_node.nodes[index]
            if i == n - 1:
                # if the word is complete, it's already present in the dictionary
                if cur_node.complete:
                    print("\tWord already present")
                    return
                # once all the characters are added to the trie, set the complete variable to True
                cur_node.complete = True
        print("\tWord added successfully!")

    # searching for a word in the dictionary
    def search_word(self, word):
        # set the can_find variable as false
        self.can_find = False
        # perform depth first search to iterate over the nodes
        self.depth_first_search(self.root, word, 0)
        return self.can_find

    def depth_first_search(self, root, word, i):
        # if word found, return true
        if self.can_find:
            return
        # if node is NULL, return
        if not root:
            return
        # if there's only one character in the word, check if it matches the query
        if len(word) == i:
            if root.complete:
                self.can_find = True
            return

        # if the word contains ".", match it with all alphabets
        if word[i] == '.':
            for j in range(ord('a'), ord('z') + 1):
                self.depth_first_search(root.nodes[j - ord('a')], word, i + 1)
        else:
            index = ord(word[i]) - ord('a')
            self.depth_first_search(root.nodes[index], word, i + 1)

"""