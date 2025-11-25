import numpy as np

class XOR_DataSet:
    def __init__(self, repeat = 1, inputTimes = 10, inputs=None):
        self.datas = []
        self.inputTimes = inputTimes

        if inputs:
            for i in range(repeat):
                for input in inputs:
                    self.datas.append(input)
        else:
            for i in range(repeat):
                # self.datas.append([[0, 0], [0]])
                self.datas.append([[1, 0], [1]])
                self.datas.append([[0, 1], [1]])
                # self.datas.append([[1, 0], [1]])
                self.datas.append([[1, 1], [0]])
        
        self.indexes = np.asarray(list(range(len(self.datas))))

    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(self.datas[i][0]), self.datas[i][1]
    
    def __len__(self):
        return len(self.datas)
    
    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, x):
        r = []
        for i in range(self.inputTimes):
            r.append(x)
        return r


class Delayed_XOR_DataSet:
    # works with 3x3, no weight bound (or maybe with), no refractory time reset
    # works with 7x7, with weight bound, with refractory time reset
    # work with 3x3 lstm, final reading
    def __init__(self, repeat = 1, inputTimes = 10, inputs=None):
        self.datas = []
        self.inputTimes = inputTimes
        if inputs:
            for i in range(repeat):
                for input in inputs:
                    self.datas.append(input)
        else:
            for i in range(repeat):
                # self.datas.append([[0, 0], [0]])
                self.datas.append([[1, 0], [1]])
                self.datas.append([[0, 1], [1]])
                # self.datas.append([[1, 0], [1]])
                self.datas.append([[1, 1], [0]])
        
        self.indexes = np.asarray(list(range(len(self.datas))))

    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(self.datas[i][0]), self.datas[i][1]
    
    def __len__(self):
        return len(self.datas)
    
    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, x):
        r = []
        for i in range(self.inputTimes//2):
            if x[0] == 1:
                r.append([1, 0])
            else:
                r.append([0, 0])
        for i in range(self.inputTimes//2):
            if x[1] == 1:
                r.append([0, 1])
            else:
                r.append([0, 0])
        return r 
    
class Single_Channel_XOR_DataSet:
    # work with 7x7, unbounded weights, without refractory reset
    # work with 3x3, unbounded weights, without refractory reset
    # work with 3x3 lstm, final reading
    def __init__(self, repeat = 1, inputTimes = 10, inputs=None):
        self.datas = []
        self.inputTimes = inputTimes
        if inputs:
            for i in range(repeat):
                for input in inputs:
                    self.datas.append(input)
        else:
            for i in range(repeat):
                # self.datas.append([[0, 0], [0]])
                self.datas.append([[1, 0], [1]])
                self.datas.append([[0, 1], [1]])
                # self.datas.append([[1, 0], [1]])
                self.datas.append([[1, 1], [0]])
        
        self.indexes = np.asarray(list(range(len(self.datas))))

    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            
            yield self.encode(self.datas[i][0]), self.datas[i][1]
    
    def __len__(self):
        return len(self.datas)
    
    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, x):
        r = []
        for i in range(self.inputTimes//2):
            if x[0] == 1:
                r.append([1])
            else:
                r.append([0])
        for i in range(self.inputTimes//2):
            if x[1] == 1:
                r.append([1])
            else:
                r.append([0])

        return r 


class Single_Channel_XOR_DataSet_with_invert:
    # work with 7x7, reset refractory, weight bounded
    # work with 3x3, without reset refractory, or bound weight (occasionally not working, investigate)
    def __init__(self, repeat = 1, inputTimes = 10, inputs=None):
        self.datas = []
        self.inputTimes = inputTimes
        if inputs:
            for i in range(repeat):
                for input in inputs:
                    self.datas.append(input)
        else:
            for i in range(repeat):
                # self.datas.append([[0, 0], [0]])
                self.datas.append([[1, 0], [1]])
                self.datas.append([[0, 1], [1]])
                # self.datas.append([[1, 0], [1]])
                self.datas.append([[1, 1], [0]])
        
        self.indexes = np.asarray(list(range(len(self.datas))))

    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(self.datas[i][0]), self.datas[i][1]
    
    def __len__(self):
        return len(self.datas)
    
    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, x):
        r = []
        for i in range(self.inputTimes//2):
            if x[0] == 1:
                r.append([1])
            else:
                r.append([0])
        for i in range(self.inputTimes//2):
            if x[1] == 1:
                r.append([-1])
            else:
                r.append([0])

        return r 
    

class Single_Channel_XOR_DataSet_with_invert_with_interruption:
    # work with 7x7, reset refractory, weight bounded
    # doesn't work with LSTM  7x7, with final reading
    # doesn't work with LSTM  7x7, with continuous reading
    def __init__(self, repeat = 1, inputTimes = 10, inputs=None):
        self.datas = []
        self.inputTimes = inputTimes
        if inputs:
            for i in range(repeat):
                for input in inputs:
                    self.datas.append(input)
        else:
            for i in range(repeat):
                # self.datas.append([[0, 0], [0]])
                self.datas.append([[1, 0], [1]])
                self.datas.append([[0, 1], [1]])
                # self.datas.append([[1, 0], [1]])
                self.datas.append([[1, 1], [0]])
        
        self.indexes = np.asarray(list(range(len(self.datas))))

    def __iter__(self):
        # np.random.shuffle(self.indexes)
        for i in self.indexes:
            yield self.encode(self.datas[i][0]), self.datas[i][1]
    
    def __len__(self):
        return len(self.datas)
    
    def shuffle(self):
        return np.random.shuffle(self.indexes)
    
    def encode(self, x):
        r = []
        for i in range(self.inputTimes//3):
            if x[0] == 1:
                r.append([1])
            else:
                r.append([0])
        for i in range(self.inputTimes//3):
            if x[1] == 1:
                r.append([-1])
            else:
                r.append([0])

        for i in range(self.inputTimes//3):
            if x[0] == 1:
                r.append([1])
            else:
                r.append([0])
        return r 