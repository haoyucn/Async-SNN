import numpy as np
import global_var
import pickle as pkl


def safeDivide(np1, np2):
    return np1 / np.where((np2 == 0), 1, np2) * (np2 != 0)

class MultiAttributeSpikingLayer:
    def __init__(self, nInput = 3, nOutput = 3):
        self.numNeurons = nOutput
        self.parentLayer = None
        self.childLayer = None
        self.nOutput = nOutput
        self.nInput = nInput
        self.time = -1
        self.debug = False

        self.n_voltageLeakRate = np.ones(self.nOutput) * 0.05
        self.n_v = np.zeros(self.nOutput)
        self.n_timesFired = np.zeros(self.nOutput)
        self.s_postSynReceptorAmp = np.random.uniform(-1, 1,(self.nOutput, self.nInput))
        self.n_firingFlag = np.zeros(self.nOutput)
        self.n_lastFiringTime = np.zeros(self.nOutput)

        self.n_weightBuffer = np.zeros((self.nOutput, self.nInput))
    
    def addParentAndChildLayers(self, pl, cl):
        self.parentLayer = pl
        self.childLayer = cl

    def reset(self, keepCal = False):
        # neuronal attribute
        self.time = -1
        self.n_v = np.zeros(self.nOutput)
        self.n_timesFired = np.zeros(self.nOutput)
        self.n_lastFiringTime = np.zeros(self.nOutput)

        # print('self.n_lastFiringTime', self.n_lastFiringTime)
        # print('self.time', self.time)
        # print('')

    def updateState(self):
        self.n_v = np.multiply(self.n_v, 1 - self.n_voltageLeakRate)
        self.time = self.time + 1

    def bufferUpdate(self):
        # print('pre', self.s_postSynReceptorAmp)
        self.s_postSynReceptorAmp = np.clip(self.s_postSynReceptorAmp + self.n_weightBuffer, -1.4, 1.4)
        # print('self.n_weightBuffer', np.sum(self.n_weightBuffer))
        # if np.sum(self.n_weightBuffer) != 0:
        #     print('self.n_weightBuffer', self.n_weightBuffer)
        self.n_weightBuffer = self.n_weightBuffer * 0
        return

    def triggerNeuronFiring(self):
        if self.childLayer:
            self.childLayer.recieveSig(self.n_firingFlag)
        # if self.parentLayer and global_var.isTraining and np.sum(self.n_firingFlag) != 0:
        #     print('self.n_lastFiringTime', self.n_lastFiringTime)
        self.n_lastFiringTime = self.n_lastFiringTime * (self.n_firingFlag == 0) + (self.n_firingFlag != 0) * self.time
        if self.parentLayer and global_var.isTraining and np.sum(self.n_firingFlag) != 0:
            timeDiff = self.parentLayer.n_lastFiringTime - self.n_lastFiringTime.reshape(-1,1)
            tau = 15
            changeAmount = np.clip(np.exp(timeDiff / tau), 0, 1)
            # if np.sum(timeDiff) != 0 and np.sum(changeAmount) != 0:
            #     print('changeAmount', changeAmount * self.n_firingFlag.reshape(-1,1))
            #     # print('timeDiff', timeDiff)
            #     print('self.n_lastFiringTime', self.n_lastFiringTime)
            #     # print('self.parentLayer.n_lastFiringTime', self.parentLayer.n_lastFiringTime)
            #     # print('self.time', self.time)
            #     print('')
            wNormalized = (self.s_postSynReceptorAmp + 1.4) / 2.8
            changeAmount = (changeAmount - 0.36) * 0.01 * self.n_firingFlag.reshape(-1,1) * np.abs(1.1 - wNormalized) * np.abs(wNormalized + 0.1)
            self.n_weightBuffer = self.n_weightBuffer + changeAmount
            
        self.n_v = self.n_v * (self.n_firingFlag == 0)
        self.n_timesFired = self.n_timesFired + self.n_firingFlag
        self.n_firingFlag = self.n_firingFlag * 0
        return

    def recieveSig(self, firingNeurons):
        self.n_v = self.n_v + np.sum(firingNeurons * self.s_postSynReceptorAmp, axis=1)
        return

    def triggerVoltageCheck(self):
        self.n_firingFlag = self.n_firingFlag + (self.n_v >= 1)
        return


class MultiAttributeSpikingNetwork:
    def __init__(self, layersSizes, periodsPerInput = 6, outputThreshold = 2):
        self.layers = []
        for i in range(len(layersSizes)):
            inputSize = 0
            if i > 0:
                inputSize = layersSizes[i-1]
            outputSize = layersSizes[i]
            self.layers.append(MultiAttributeSpikingLayer(nInput=inputSize, nOutput=outputSize))
        self.periodsPerInput = periodsPerInput
        self.outputThreshold = outputThreshold

        for i in range(len(self.layers)):
            pl = None
            cl = None
            if i  > 0:
                pl = self.layers[i - 1]
            if i + 1 < len(self.layers):
                cl = self.layers[i + 1]
            self.layers[i].addParentAndChildLayers(pl, cl)


    def run(self, dataset, train = False, useAccumulativeOutput = False, epoches = 1000):
        global_var.isTraining = False
        errorSumHist = []
        global_var.LEARNING_RATE_ADOPTIVE = 1
        for e in range(epoches):
            dataset.shuffle()
            self.reset()
            errorNum = 0
            errorSum = 0
            
            for xs, y in dataset:

                self.reset()
                xs = np.asarray(xs)
                y = np.asarray(y)
                for x in xs:
                    for i in range(self.periodsPerInput):
                        if i == 0:
                            self.triggerAllActivities(x=x)
                        else:
                            self.triggerAllActivities()
                
                maxPeriod = 10
                while not self.hasAllActivitiCeased()  and maxPeriod > 0:
                    
                    self.triggerAllActivities_postInput()
                    maxPeriod -= 1

                y_hat = None
                if useAccumulativeOutput:
                    y_hat = self.getAccumulativeOutput()
                else:
                    y_hat = self.useFinalOutput(xs)
                
                error = y - y_hat
                if np.sum(np.absolute(error)) > 0:
                    errorSum = np.sum(np.absolute(error)) + errorSum
                    errorNum += 1
                    if train:
                        self.reset()
                        global_var.isTraining = True
                        self.reset(keepCal=True)
                        for x in xs:
                            for i in range(self.periodsPerInput):
                                if i == 0:
                                    self.triggerAllActivities(x=x)
                                else:
                                    self.triggerAllActivities()
                        maxPeriod = 10
                        while not self.hasAllActivitiCeased() and maxPeriod > 0:
                            self.triggerAllActivities_postInput()
                            maxPeriod -= 1
                global_var.isTraining = False
            errorSumHist.append(errorSum)
            accuracy = (len(dataset) - errorNum) / len(dataset)
            if accuracy >= 0.5 and global_var.LEARNING_RATE_ADOPTIVE > 0.1:
                # print('learning rate set to 0.1')
                global_var.LEARNING_RATE_ADOPTIVE = 0.1
                # print('found 0.0006')
            if accuracy >= 0.9 and global_var.LEARNING_RATE_ADOPTIVE > 0.01:
                # print('learning rate set to 0.01')
                global_var.LEARNING_RATE_ADOPTIVE = 0.01
            print('epoch', e, 'acc', accuracy, 'errorSum', errorSum, '\t\t\t', end='\r')
            if accuracy == 1 or train == False:
                # newAcc, e1, errorSum1, errorSumHist1 = self.run(dataset, train = False, useAccumulativeOutput = useAccumulativeOutput, epoches = epoches)
                # print('retest newAcc', newAcc)
                # stop
                # for l in self.layers:
                #     if np.sum((np.sign(l.s_postSynReceptorAmp) * np.sign(l.s_beginningAmp)) < 0) > 0:
                #         print('\nfound sign change,', l.nOutput)
                #         print('l.s_postSynReceptorAmp', l.s_postSynReceptorAmp)
                #         print('l.s_beginningAmp', l.s_beginningAmp)
                return accuracy, e, errorSum, errorSumHist
            
            for l in self.layers:
                l.bufferUpdate()
        print('failed to converge')
        return accuracy, epoches, errorSum, errorSumHist

    def triggerAllActivities(self, x=None, correction=None):
        for l in self.layers:
            l.updateState()
        if not (x is None):
            self.layers[0].n_firingFlag = x
        for l in self.layers:
            l.triggerNeuronFiring()
        for l in self.layers:
            l.triggerVoltageCheck()

    def triggerAllActivities_postInput(self):
        for l in self.layers:
            l.updateState()
        for l in self.layers:
            l.triggerNeuronFiring()
        for l in self.layers:
            l.triggerVoltageCheck()

    def hasAllActivitiCeased(self):
        # for l in self.layers:
        #     if np.sum(l.n_firingTimeLeft) > 0:
        #         return False
        return True
    
    def reset(self, keepCal=False):
        for l in self.layers:
            l.reset(keepCal = keepCal)

    def getAccumulativeOutput(self):
        return self.layers[-1].n_timesFired >= self.outputThreshold
                    
        
    def useFinalOutput(self, xs):
        acceptableTime = (len(xs) - 1) * self.periodsPerInput + len(self.layers)
        return (self.layers[-1].n_lastFiringTime >= acceptableTime) * 1
    
    def useArgMax(self):
        maxIndex = np.argmax(self.layers[-1].n_timesFired, axis=None)
        y_hat = np.zeros(self.layers[-1].n_timesFired.shape)
        y_hat[maxIndex] = 1

        if len(maxIndex.shape) > 1:
            print(maxIndex)
        return y_hat
    

