import numpy as np


PATH_TO_WORDS_FILE = 'data/words/common_words_3000.txt'

class Words_Completion_3000:
    def __init__(self, dataPath = None, letter_index_to_predict = -1):
        if dataPath is None:
            dataPath = PATH_TO_WORDS_FILE
        self.datas = []
        self.letter_index_to_predict = letter_index_to_predict
        lines = []
        with open(dataPath, 'r') as f:
            lines = f.readlines()
            f.close()

        lines = lines[:100]
        for l in lines:
            preped_l = l.strip().lower()
            if len(preped_l) < 2 or (not preped_l.isalpha()):
                continue
            self.datas.append(preped_l)
        
        self.indexes = np.arange(len(self.datas))

    
    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(i)
    
    def __len__(self):
        return len(self.datas)
    
    def _letter_to_indexes(self, letter):
        # return 0 based letter index in alph, 97 is the ASCII offset
        return ord(letter) - 97

    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, i):
        word = self.datas[i]
        providedLetters = word[:-1]
        providedLettersIndexes = [self._letter_to_indexes(c) for c in providedLetters]

        predictingTargets = [word[-1]]
        
        # search upward
        # print('word:', word, end='\r')
        for upOffSet in range(1, i + 1):
            previousWord = self.datas[i - upOffSet]
            if len(providedLetters) < len(previousWord) and providedLetters == previousWord[:len(providedLetters)]:
                l = previousWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break
        
        for nextWordIndexes in range(i + 1, len(self.datas)):
            nextWord = self.datas[nextWordIndexes]
            if len(providedLetters) < len(nextWord) and providedLetters == nextWord[:len(providedLetters)]:
                l = nextWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break 
        predictingTargetsIndexes = [self._letter_to_indexes(c) for c in predictingTargets]
        xs = []
        for idx in providedLettersIndexes:
            x = np.zeros(26)
            x[idx] = 1
            xs.append(x)
        
        y = np.zeros(26)
        for idx in predictingTargetsIndexes:
            y[idx] = 1
        return xs, y

class Words_Completion_small:
    def __init__(self, dataPath = None, letter_index_to_predict = -1, customized_data = None):
        if dataPath is None:
            dataPath = PATH_TO_WORDS_FILE
        self.datas = []
        self.letter_index_to_predict = letter_index_to_predict
        if customized_data:
            lines = customized_data
        else:
            lines = []
            with open(dataPath, 'r') as f:
                lines = f.readlines()
                f.close()
            # lines = lines[0:20] 
            maxWordLength = 6
            maxNumbersWordNeed = 10
        # print(lines)
        for l in lines:
            preped_l = l.strip().lower()
            if len(preped_l) < 2 or (not preped_l.isalpha()):
                continue
            if maxWordLength != 0 and len(preped_l) > maxWordLength:
                continue
            
            if maxNumbersWordNeed != 0 and len(self.datas) >= maxNumbersWordNeed:
                break
            self.datas.append(preped_l)
        print('maxWordLength', maxWordLength, 'maxNumbersWordNeed',maxNumbersWordNeed)
        print(self.datas)
        print(f'total of {len(self.datas)} records in dataset')
        self.indexes = np.arange(len(self.datas))

    
    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(i)
    
    def __len__(self):
        return len(self.datas)
    
    def _letter_to_indexes(self, letter):
        # return 0 based letter index in alph, 97 is the ASCII offset
        return ord(letter) - 97

    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, i):
        word = self.datas[i]
        providedLetters = word[:-1]
        providedLettersIndexes = [self._letter_to_indexes(c) for c in providedLetters]

        predictingTargets = [word[-1]]
        
        # search upward
        # print('word:', word, end='\r')
        for upOffSet in range(1, i + 1):
            previousWord = self.datas[i - upOffSet]
            if len(providedLetters) == len(previousWord) and providedLetters == previousWord[:len(providedLetters)]:
                l = previousWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break
        
        for nextWordIndexes in range(i + 1, len(self.datas)):
            nextWord = self.datas[nextWordIndexes]
            if len(providedLetters) == len(nextWord) and providedLetters == nextWord[:len(providedLetters)]:
                l = nextWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break 
        predictingTargetsIndexes = [self._letter_to_indexes(c) for c in predictingTargets]
        xs = []
        for idx in providedLettersIndexes:
            x = np.zeros(26)
            x[idx] = 1
            xs.append(x)
        
        y = np.zeros(26)
        for idx in predictingTargetsIndexes:
            y[idx] = 1
        return xs, y
    
class Words_Completion_small_instant:
    def __init__(self, dataPath = None, letter_index_to_predict = -1, customized_data = None):
        if dataPath is None:
            dataPath = PATH_TO_WORDS_FILE
        self.datas = []
        self.letter_index_to_predict = letter_index_to_predict
        if customized_data:
            lines = customized_data
        else:
            lines = []
            with open(dataPath, 'r') as f:
                lines = f.readlines()
                f.close()
            lines = lines[29:] 
            maxWordLength = 6
            maxNumbersWordNeed = 5
        # print(lines)
        for l in lines:
            preped_l = l.strip().lower()
            if len(preped_l) < 2 or (not preped_l.isalpha()):
                continue
            if maxWordLength != 0 and len(preped_l) > maxWordLength:
                continue
            
            if maxNumbersWordNeed != 0 and len(self.datas) >= maxNumbersWordNeed:
                break
            self.datas.append(preped_l)
        print('maxWordLength', maxWordLength, 'maxNumbersWordNeed',maxNumbersWordNeed)
        print(self.datas)
        print(f'total of {len(self.datas)} records in dataset')
        self.indexes = np.arange(len(self.datas))

    
    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(i)
    
    def __len__(self):
        return len(self.datas)
    
    def _letter_to_indexes(self, letter):
        # return 0 based letter index in alph, 97 is the ASCII offset
        return ord(letter) - 97

    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, i):
        word = self.datas[i]
        providedLetters = word[:-1]
        providedLettersIndexes = [self._letter_to_indexes(c) for c in providedLetters]

        predictingTargets = [word[-1]]
        
        # search upward
        # print('word:', word, end='\r')
        for upOffSet in range(1, i + 1):
            previousWord = self.datas[i - upOffSet]
            if len(word) == len(previousWord) and providedLetters == previousWord[:len(providedLetters)]:
                l = previousWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break
        
        for nextWordIndexes in range(i + 1, len(self.datas)):
            nextWord = self.datas[nextWordIndexes]
            if len(word) == len(nextWord) and providedLetters == nextWord[:len(providedLetters)]:
                l = nextWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break 
        predictingTargetsIndexes = [self._letter_to_indexes(c) for c in predictingTargets]
        xs = []
        x = np.zeros(26)
        
        for idx in providedLettersIndexes:
            x[idx] = 1
        for i in range(10):  
            xs.append(x)
        
        y = np.zeros(26)
        for idx in predictingTargetsIndexes:
            y[idx] = 1
        return xs, y


class Words_Completion_debug:
    def __init__(self, dataPath = None, letter_index_to_predict = -1, inputs = None):
        
        self.datas = []
        self.letter_index_to_predict = letter_index_to_predict
        # lines = ['abandon', 'ability', 'able']
        lines = ['abandon', 'able']
        if inputs:
            lines = inputs
        for l in lines:
            preped_l = l.strip().lower()
            if len(preped_l) < 2 or (not preped_l.isalpha()):
                continue
            self.datas.append(preped_l)
        
        self.indexes = np.arange(len(self.datas))

    
    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(i)
    
    def __len__(self):
        return len(self.datas)
    
    def _letter_to_indexes(self, letter):
        # return 0 based letter index in alph, 97 is the ASCII offset
        return ord(letter) - 97

    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, i):
        word = self.datas[i]
        providedLetters = word[:-1]
        providedLettersIndexes = [self._letter_to_indexes(c) for c in providedLetters]

        predictingTargets = [word[-1]]
        
        # search upward
        # print('word:', word, end='\r')
        for upOffSet in range(1, i + 1):
            previousWord = self.datas[i - upOffSet]
            if len(providedLetters) < len(previousWord) and providedLetters == previousWord[:len(providedLetters)]:
                l = previousWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break
        
        for nextWordIndexes in range(i + 1, len(self.datas)):
            nextWord = self.datas[nextWordIndexes]
            if len(providedLetters) < len(nextWord) and providedLetters == nextWord[:len(providedLetters)]:
                l = nextWord[len(providedLetters)]
                if l not in predictingTargets:
                    predictingTargets.append(l)
            else:
                break 
        predictingTargetsIndexes = [self._letter_to_indexes(c) for c in predictingTargets]
        xs = []
        for idx in providedLettersIndexes:
            x = np.zeros(26)
            x[idx] = 1
            xs.append(x)
        
        y = np.zeros(26)
        for idx in predictingTargetsIndexes:
            y[idx] = 1
        return xs, y
