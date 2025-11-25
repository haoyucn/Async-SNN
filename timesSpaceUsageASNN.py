# %%
import os
from network import MultiAttributeSpikingNetwork
# from network_stdp import MultiAttributeSpikingNetwork
import pickle as pkl
import global_var
import time
import numpy as np
from custom_datasets.XOR_dataset import XOR_DataSet as Dataset
from memory_profiler import profile

logDir = './experiment-logs/'
if not os.path.exists(logDir):
    os.makedirs(logDir)

logDir = './savedModels/'
if not os.path.exists(logDir):
    os.makedirs(logDir)


def print_weights(neuron):
    for d in neuron.dendrites:
        print(d.synapticAttribute.postSynReceptorAmp)
        
def print_clearance(neuron):
    for d in neuron.dendrites:
        print(d.synapticAttribute.transmitterClearanceRate)

@profile
def train(trails = 1, layerSizes = [2,3,1], epochesPerTries = 100, network = None, dataset = None, outputThreshold = 2, useAccumulativeOutput = True, debug = False, train = True, loadPath = None):
    totalTime = 0
    totalEpochs = 0
    for t in range(trails):
        hist = []
        test_network = network
        if test_network is None:
            test_network = MultiAttributeSpikingNetwork(layerSizes, outputThreshold = outputThreshold, periodsPerInput = 6)
            if loadPath:
                test_network.importParameters(loadPath)
        startTime = time.time()
        acc, convergeEpoch, errorSum, hist = test_network.run(dataset, epoches = epochesPerTries, train=train, useAccumulativeOutput=useAccumulativeOutput)
        epochTimeUsed = time.time() - startTime
        totalTime += epochTimeUsed
        totalEpochs += convergeEpoch
        
        print('trail:', t, 'acc:', acc, 'convergeEpoch', convergeEpoch, 'timePerEpoch', epochTimeUsed / (convergeEpoch + 1), 'errorSum', errorSum, '\t\t\t')
    return hist, test_network

# %% [markdown]
# ## XOR Dataset

# %%
from custom_datasets.XOR_dataset import XOR_DataSet as Dataset

# hist, test_network = train(trails = 10, layerSizes = [2,3,1], dataset = Dataset(), epochesPerTries = 800, useAccumulativeOutput = False)

# %% [markdown]
# ## delayed XOR

# %%
from custom_datasets.XOR_dataset import Delayed_XOR_DataSet as Dataset

# hist, test_network = train(trails = 10, layerSizes = [2,3,1], dataset = Dataset(), epochesPerTries = 800, useAccumulativeOutput = False)

# %% [markdown]
# ## single channel XOR

# %%
from custom_datasets.XOR_dataset import Single_Channel_XOR_DataSet_with_invert as Dataset

# hist, test_network = train(trails = 10, layerSizes = [1,3,1], dataset = Dataset(), epochesPerTries = 1600, useAccumulativeOutput = False)

# %% [markdown]
# ## single channel with interruption

# %%
from custom_datasets.XOR_dataset import Single_Channel_XOR_DataSet_with_invert_with_interruption as Dataset

# hist, test_network = train(trails = 10, layerSizes = [1,3,1], dataset = Dataset(), epochesPerTries = 1800, useAccumulativeOutput = False)

# %% [markdown]
# ## word completion test

# %%

from custom_datasets.word_completion_dataset import Words_Completion_small as Dataset

hist,test_network = train(trails = 1, layerSizes =  [26, 72, 26], dataset = Dataset(),  epochesPerTries = 1, useAccumulativeOutput = False)


