import json

class LabelTokenizer():
    def __init__(self, id2text_path, word2id_path):
        '''the first id indicates the relation id(label id), the second means the token id'''
        with open(id2text_path, 'r') as f:
            self.id2text = json.load(f)
        with open(word2id_path, 'r') as f:
            self.word2id = json.load(f)

        self.id2word = {y:x for x, y in self.word2id.items()}
    
    def __call__(self, id):
        label_text = self.id2text[id].lower()
        label_word = label_text.split(' ')
        label_token = [self.word2id[word] if (word in self.word2id.keys()) else (self.word2id['[UNK]']) for word in label_word]
        label_token.append(len(self.word2id))   # append [eos]
        return label_token

# lt = LabelTokenizer(r'data\relid2Text.json', r'pretrain\glove\glove_word2id.json')
# print(lt('P921'))