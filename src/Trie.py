import torch

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
        prob_list = torch.tensor(list(self.curr.children_prob.values()))
        char_list = torch.tensor(list(self.curr.children_prob.keys()))
        indices_list = torch.topk(prob_list, 3).indices
        best_chars = []
        
        for indices in indices_list:
          index = indices.item()
          best_chars.append(char_list[index])
            
        return best_chars

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
