class TrieNode:
    def __init__(self):
        # character to children dict
        self.children = {}
        # character to probability dict
        self.children_prob = {}


class Trie:
    def __init__(self, word_to_prob):
        self.root = self.getNewNode()
        self.buildTrie(word_to_prob)
        self.curr = self.root

    def getNewNode(self):
        return TrieNode()

    def buildTrie(self, word_to_prob):
        for word, prob in word_to_prob.items():
            pCrawl = self.root
            for i in range(len(word)):
                charac = word[i]
                # if current character is not present
                if charac not in pCrawl.children:
                    pCrawl.children[charac] = self.getNewNode()
                    pCrawl.children_prob[charac] = 0.0
                pCrawl.children_prob[charac] += prob

                pCrawl = pCrawl.children[charac]

            # When the char is end of word, its children is None
            pCrawl.children[' '] = None
            pCrawl.children_prob[' '] = prob

    # need to call advance_curr() before this
    def get_next_char(self):
        best_char_prob = {}
        least = float('inf')
        least_key = ''
        for char, prob in self.curr.children_prob.items():
            if char in best_char_prob.keys():
                best_char_prob[char] = max(prob, best_char_prob[char])

                if best_char_prob[char] < least:
                    least = best_char_prob[char]
                    least_key = char

            else:
                if len(best_char_prob) < 3:
                    least, least_key = append_to_dict(char, prob, best_char_prob)
                elif prob > least:
                    del best_char_prob[least_key]
                    least, least_key = append_to_dict(char, prob, best_char_prob)

        return best_char_prob.keys()

    # return 0 if the str is not present in the trie
    # return 1 if curr is advanced
    def advance_curr(self, str):
        # depending on if we are given sequential inputs or not
        self.reset_curr()

        for i in range(len(str)):
            if str[i] in self.curr.children:
                self.curr = self.curr.children[str[i]]
            else:
                #print("This str is not in the trie.")
                return 0

        return 1

    def reset_curr(self):
        self.curr = self.root


def append_to_dict(char, prob, best_char_prob):
    best_char_prob[char] = prob

    least = float('inf')
    least_key = ''
    for char, prob in best_char_prob.items():
        if prob < least:
            least = prob
            least_key = char
    
    return least, least_key
