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
        

        self.n_v = np.zeros(self.nOutput)
        self.n_v_pos = np.zeros(self.nOutput)
        self.n_v_neg = np.zeros(self.nOutput)
        self.n_daV = np.zeros(self.nOutput)
        self.n_daV_pos = np.zeros(self.nOutput)
        self.n_daV_neg = np.zeros(self.nOutput)
        self.n_sigDuration = np.ones(self.nOutput) * 2
        self.n_refractoryPeriod = np.ones(self.nOutput) * 3
        self.n_voltageLeakRate = np.ones(self.nOutput) * 0.05
        self.n_calRemain = np.zeros(self.nOutput)
        self.n_calRemain_v_pos = np.zeros(self.nOutput)
        self.n_calRemain_v_neg = np.zeros(self.nOutput)
        self.n_calRemain_daV_pos = np.zeros(self.nOutput)
        self.n_calRemain_daV_neg = np.zeros(self.nOutput)
        self.n_refactoryTimeLeft = np.zeros(self.nOutput)
        self.n_firingTimeLeft = np.zeros(self.nOutput)
        self.n_firingPolarity = np.ones(self.nOutput)
        self.n_firingTypes = np.zeros(self.nOutput) # 0 for reg, 1 for da
        self.n_forceFiring = np.zeros(self.nOutput)
        self.n_forcedFiringType = np.zeros(self.nOutput)
        self.n_nueronRecievedSigIdx = np.zeros(self.nOutput)
        self.n_timesFired_neg = np.zeros(self.nOutput)
        self.n_timesFired_pos = np.zeros(self.nOutput)
        self.n_lastFiringTime = np.zeros(self.nOutput)
        self.n_lastFirePolarity = np.zeros(self.nOutput)
        self.n_firingCalResidule = np.zeros(self.nOutput)

        # this is the weight, in a traditonal sense
        self.s_postSynReceptorAmp = np.random.uniform(-1, 1,(self.nOutput, self.nInput))
        # this is the new type of weight introduced in paper, to control signal duration
        self.s_transmitterClearanceRate = np.random.uniform(0.1, 0.7, (self.nOutput, self.nInput))

        self.s_regTransmitterRemain = np.zeros((self.nOutput, self.nInput))
        self.s_dATransmitterRemain = np.zeros((self.nOutput, self.nInput))
        self.s_calRemain = np.zeros((self.nOutput, self.nInput))
        self.s_calRemain_pos = np.zeros((self.nOutput, self.nInput))
        self.s_calRemain_neg = np.zeros((self.nOutput, self.nInput))
        self.s_daCalRemain = np.zeros((self.nOutput, self.nInput))
        self.s_daCalRemain_neg = np.zeros((self.nOutput, self.nInput))
        self.s_daCalRemain_pos = np.zeros((self.nOutput, self.nInput))
        self.n_releaseTrace = np.zeros((self.nOutput, self.nInput))
        self.s_ampMod = np.zeros((self.nOutput, self.nInput))
        self.s_updateAmp = np.zeros((self.nOutput, self.nInput))

        self.buffer_amp = 0
        self.buffer_clearanceRate = 0

    def reset(self, keepCal = False):
        # neuronal attribute
        self.time = -1
        self.n_v = np.zeros(self.nOutput)
        self.n_v_pos = np.zeros(self.nOutput)
        self.n_v_neg = np.zeros(self.nOutput)
        self.n_daV = np.zeros(self.nOutput)
        self.n_daV_pos = np.zeros(self.nOutput)
        self.n_daV_neg = np.zeros(self.nOutput)
        
        self.n_calRemain = np.zeros(self.nOutput)
        self.n_calRemain_v_pos = np.zeros(self.nOutput)
        self.n_calRemain_v_neg = np.zeros(self.nOutput)
        self.n_calRemain_daV_pos = np.zeros(self.nOutput)
        self.n_calRemain_daV_neg = np.zeros(self.nOutput)
        self.n_refactoryTimeLeft = np.zeros(self.nOutput)
        self.n_firingTimeLeft = np.zeros(self.nOutput)
        self.n_firingPolarity = np.ones(self.nOutput)
        self.n_firingTypes = np.zeros(self.nOutput) # 0 for reg, 1 for da
        self.n_forceFiring = np.zeros(self.nOutput)
        self.n_forcedFiringType = np.zeros(self.nOutput)
        self.n_nueronRecievedSigIdx = np.zeros(self.nOutput)
        self.n_timesFired_neg = np.zeros(self.nOutput)
        self.n_timesFired_pos = np.zeros(self.nOutput)
        self.n_lastFirePolarity = np.zeros(self.nOutput)
        self.n_lastFiringTime = np.zeros(self.nOutput)
        if not keepCal:
            self.n_firingCalResidule = np.zeros(self.nOutput)

        self.s_ceilingLr = np.ones((self.nOutput, self.nInput))
        self.s_regTransmitterRemain = np.zeros((self.nOutput, self.nInput))
        self.s_dATransmitterRemain = np.zeros((self.nOutput, self.nInput))
        self.s_calRemain = np.zeros((self.nOutput, self.nInput))
        self.s_calRemain_pos = np.zeros((self.nOutput, self.nInput))
        self.s_calRemain_neg = np.zeros((self.nOutput, self.nInput))
        self.s_daCalRemain = np.zeros((self.nOutput, self.nInput))
        self.s_daCalRemain_neg = np.zeros((self.nOutput, self.nInput))
        self.s_daCalRemain_pos = np.zeros((self.nOutput, self.nInput))
        self.n_releaseTrace = np.zeros((self.nOutput, self.nInput))
        self.s_ampMod = np.zeros((self.nOutput, self.nInput))
        self.s_updateAmp = np.zeros((self.nOutput, self.nInput))

    def postFiringReset(self, neuronsToReset):
        neuronValueToKeep = neuronsToReset == 0
        self.n_v = self.n_v * neuronValueToKeep
        self.n_v_pos = self.n_v_pos * neuronValueToKeep
        self.n_v_neg = self.n_v_neg * neuronValueToKeep
        self.n_daV = self.n_daV * neuronValueToKeep
        self.n_daV_pos = self.n_daV_pos * neuronValueToKeep
        self.n_daV_neg = self.n_daV_neg * neuronValueToKeep
        self.n_forcedFiringType = self.n_forcedFiringType * neuronValueToKeep
        self.n_firingPolarity = self.n_firingPolarity * neuronValueToKeep
        self.n_firingTimeLeft = self.n_firingTimeLeft * neuronValueToKeep
        self.n_firingTypes = self.n_firingTypes * neuronValueToKeep
        self.n_calRemain = self.n_calRemain * neuronValueToKeep
        self.n_calRemain_v_pos = self.n_calRemain_v_pos * neuronValueToKeep
        self.n_calRemain_v_neg = self.n_calRemain_v_neg * neuronValueToKeep
        self.n_calRemain_daV_pos = self.n_calRemain_daV_pos * neuronValueToKeep
        self.n_calRemain_daV_neg = self.n_calRemain_daV_neg * neuronValueToKeep

    def updateState(self):
        self.n_v = np.multiply(self.n_v, 1 - self.n_voltageLeakRate)
        self.n_v_pos = np.multiply(self.n_v_pos, 1 - self.n_voltageLeakRate)
        self.n_v_neg = np.multiply(self.n_v_neg, 1 - self.n_voltageLeakRate)

        self.n_daV = np.multiply(self.n_daV, 1 - self.n_voltageLeakRate)
        self.n_daV_pos = np.multiply(self.n_daV_pos, 1 - self.n_voltageLeakRate)
        self.n_daV_neg = np.multiply(self.n_daV_neg, 1 - self.n_voltageLeakRate)

        self.n_refactoryTimeLeft = np.maximum(self.n_refactoryTimeLeft - 1, 0)

        self.n_calRemain = np.multiply(self.n_calRemain, 1 - self.n_voltageLeakRate)
        self.n_calRemain_v_pos = np.multiply(self.n_calRemain_v_pos, 1 - self.n_voltageLeakRate)
        self.n_calRemain_v_neg = np.multiply(self.n_calRemain_v_neg, 1 - self.n_voltageLeakRate)
        self.n_calRemain_daV_pos = np.multiply(self.n_calRemain_daV_pos, 1 - self.n_voltageLeakRate)
        self.n_calRemain_daV_neg = np.multiply(self.n_calRemain_daV_neg, 1 - self.n_voltageLeakRate)
        self.n_nueronRecievedSigIdx = np.zeros(self.nOutput)

        self.n_firingCalResidule = self.n_firingCalResidule * 0.9
        

        self.s_regTransmitterRemain = np.multiply(self.s_regTransmitterRemain, 1 - self.s_transmitterClearanceRate)
        self.s_dATransmitterRemain = np.multiply(self.s_dATransmitterRemain, 1 - self.s_transmitterClearanceRate)

        self.s_calRemain = np.multiply(self.s_calRemain, 1 - self.s_transmitterClearanceRate)
        self.s_calRemain_pos = np.multiply(self.s_calRemain_pos, 1 - self.s_transmitterClearanceRate)
        self.s_calRemain_neg = np.multiply(self.s_calRemain_neg, 1 - self.s_transmitterClearanceRate)

        self.n_releaseTrace = np.multiply(self.n_releaseTrace, 1 - self.s_transmitterClearanceRate)

        self.s_daCalRemain = np.multiply(self.s_daCalRemain, 1 - self.s_transmitterClearanceRate)
        self.s_daCalRemain_neg = np.multiply(self.s_daCalRemain_neg, 1 - self.s_transmitterClearanceRate)
        self.s_daCalRemain_pos = np.multiply(self.s_daCalRemain_pos, 1 - self.s_transmitterClearanceRate)
        self.s_updateAmp = np.zeros((self.nOutput, self.nInput))
        self.time += 1

    def bufferUpdate(self):
        self.s_postSynReceptorAmp = self.s_postSynReceptorAmp + self.buffer_amp * 0.001
        ampAbs = np.abs(self.s_postSynReceptorAmp)


        self.s_transmitterClearanceRate = np.clip(self.s_transmitterClearanceRate + self.buffer_clearanceRate * 0.0001, 0.01, 0.7)
        self.buffer_amp = 0
        self.buffer_clearanceRate = 0

    def triggerNeuronFiring(self):

        firingNeuron = self.n_firingTimeLeft > 0
        regFiringNeuron = (self.n_firingTypes == global_var.SIG_TYPES['regular']) * firingNeuron
        daFiringNeuron = (self.n_firingTypes == global_var.SIG_TYPES['ISI']) * firingNeuron

        self.n_firingCalResidule = self.n_firingCalResidule + 1 * regFiringNeuron

        if self.childLayer:
            self.childLayer.s_regTransmitterRemain = self.childLayer.s_regTransmitterRemain + self.n_firingPolarity * regFiringNeuron * 1
            self.childLayer.n_releaseTrace = self.childLayer.n_releaseTrace + (self.childLayer.n_releaseTrace == 0) * regFiringNeuron * 0.5
        if self.parentLayer:
            self.s_dATransmitterRemain = self.s_dATransmitterRemain + np.absolute(self.s_postSynReceptorAmp) * self.n_firingPolarity.reshape(-1,1) * daFiringNeuron.reshape(-1,1)

        self.n_firingTimeLeft = np.maximum(self.n_firingTimeLeft - (regFiringNeuron + daFiringNeuron), 0)

        doneFiringNeuron = firingNeuron * (self.n_firingTimeLeft == 0)
        self.n_timesFired_pos = self.n_timesFired_pos + doneFiringNeuron * (self.n_firingPolarity > 0)
        self.n_timesFired_neg = self.n_timesFired_neg + doneFiringNeuron * (self.n_firingPolarity < 0)

        self.n_refactoryTimeLeft = self.n_refactoryTimeLeft + doneFiringNeuron * self.n_refractoryPeriod
        
        
        self.n_lastFirePolarity = np.where(((doneFiringNeuron * self.n_firingPolarity * regFiringNeuron) != 0), self.n_firingPolarity, self.n_lastFirePolarity)
        self.n_lastFiringTime = np.where((doneFiringNeuron * regFiringNeuron), self.time, self.n_lastFiringTime)

        self.n_firingPolarity = self.n_firingPolarity * (doneFiringNeuron == 0)
        self.postFiringReset(doneFiringNeuron)

    

    def triggerSynapseSendSig(self):
        # send reg signals --------------------------------------------------------------------------
        regAmp = self.s_regTransmitterRemain * (self.s_postSynReceptorAmp + self.s_ampMod)
        self.accumulateSynapticRegCalcium()
        self.recieveSig(regAmp)

        # -------------------------------------------------------------------------------------------
        
        if self.parentLayer:
            # send correction signals ---------------------------------------------------------------------------
            dASynapses = self.s_dATransmitterRemain != 0
            
            self.s_updateAmp = self.s_dATransmitterRemain * np.absolute(self.s_postSynReceptorAmp + self.s_ampMod) * dASynapses

            self.s_daCalRemain = self.s_daCalRemain + 0.5 * dASynapses
            self.s_daCalRemain_pos = self.s_daCalRemain_pos + 0.5 * (self.s_dATransmitterRemain > 0)
            self.s_daCalRemain_neg = self.s_daCalRemain_neg + 0.5 * (self.s_dATransmitterRemain < 0)

            signModifier = np.where((self.s_calRemain_neg > self.s_calRemain_pos), -1, 1) * dASynapses
            signModifier = dASynapses
            
            daAmp = np.transpose(self.s_updateAmp * np.sign(self.s_postSynReceptorAmp + self.s_ampMod) * np.maximum(0.1, np.abs(self.s_calRemain)) * signModifier)
            
            self.parentLayer.recieveSig(daAmp, sigType = global_var.SIG_TYPES['ISI'])
            # -------------------------------------------------------------------------------------------
        self.clearTransmitterIfShould()
    
    def clearTransmitterIfShould(self):
        self.s_dATransmitterRemain = np.where((np.absolute(self.s_dATransmitterRemain) < 0.1), 0, self.s_dATransmitterRemain)
        self.s_regTransmitterRemain = np.where((np.absolute(self.s_regTransmitterRemain) < self.s_transmitterClearanceRate), 0, self.s_regTransmitterRemain)
        self.n_releaseTrace = np.where((self.s_regTransmitterRemain == 0), 0 , self.n_releaseTrace)

    def accumulateSynapticRegCalcium(self): 
        self.s_calRemain = self.s_calRemain + 0.5 * (self.s_regTransmitterRemain != 0)
        self.s_calRemain_pos = self.s_calRemain_pos + 0.5 * (self.s_regTransmitterRemain > 0)
        self.s_calRemain_neg = self.s_calRemain_neg + 0.5 * (self.s_regTransmitterRemain < 0)

    def recieveSig(self, amp, sigType = global_var.SIG_TYPES['regular']):
        aviliableNeurons = self.getNonActivingNeurons()
        if sigType == global_var.SIG_TYPES['regular']:
            
            regSynapses_pos = amp > 0
            regSynapses_neg = amp < 0
            
            self.n_v_pos = self.n_v_pos + np.sum(regSynapses_pos * amp, axis = 1) * aviliableNeurons
            self.n_calRemain_v_pos = self.n_calRemain_v_pos + np.sum(regSynapses_pos * 0.5, axis = 1) * aviliableNeurons
            self.n_v_neg = self.n_v_neg + np.sum(regSynapses_neg * amp, axis = 1) * aviliableNeurons
            self.n_calRemain_v_neg = self.n_calRemain_v_neg + np.sum(regSynapses_neg * 0.5, axis = 1) * aviliableNeurons

        elif sigType == global_var.SIG_TYPES['ISI']:
            neuronAmount = amp.shape[0]
            if self.numNeurons != neuronAmount:
                print('ERROR', 'self.numNeurons', self.numNeurons, 'neuronAmount', neuronAmount, 'amp', amp, 'self.childLayer.s_dATransmitterRemain', self.childLayer.s_dATransmitterRemain)
            daSynapses_pos = amp > 0
            daSynapses_neg = amp < 0

            self.n_daV = self.n_daV + np.sum(amp, axis = 1) * aviliableNeurons
            self.n_daV_pos = self.n_daV_pos + np.sum(daSynapses_pos * amp, axis = 1) * aviliableNeurons 
            self.n_calRemain_daV_pos = self.n_calRemain_daV_pos + np.sum(daSynapses_pos * 0.5, axis = 1) * aviliableNeurons
            self.n_daV_neg = self.n_daV_neg + np.sum(daSynapses_neg * amp, axis = 1) * aviliableNeurons
            self.n_calRemain_daV_neg = self.n_calRemain_daV_neg + np.sum(daSynapses_neg * 0.5, axis = 1) * aviliableNeurons
        
        self.n_v = self.n_v + np.sum(amp, axis = 1) * aviliableNeurons
        self.n_calRemain = self.n_calRemain + aviliableNeurons * np.sum((amp != 0) * 0.5, axis = 1)
        self.n_nueronRecievedSigIdx = (np.sum(amp != 0, axis=1, keepdims=False) * aviliableNeurons + self.n_nueronRecievedSigIdx) != 0

    def triggerVoltageCheck(self, forceFiringIdx = 0, forceFiringPolarity = 0, forceSigType = 0):
        neuronsToCheck = (self.n_nueronRecievedSigIdx != 0) * self.getNonActivingNeurons()
        nonForcedNeedToFire = self.n_nueronRecievedSigIdx * (np.absolute(self.n_v) >= 1) * neuronsToCheck
        checkLowVoltageNeuron = (nonForcedNeedToFire == 0) * neuronsToCheck
        nonForcedNeedToFire = nonForcedNeedToFire + self.getLowAmpNeurons(checkLowVoltageNeuron)
        nonForcedNeedToFire_da = nonForcedNeedToFire * (self.n_daV != 0)
        nonForcedNeedToFire_reg = nonForcedNeedToFire * (nonForcedNeedToFire_da == 0)
        nonForcedNeedToFire_da_polarity = np.sign(self.n_daV) * nonForcedNeedToFire_da
        nonForcedNeedToFire_reg_polarity = np.sign(self.n_v) * nonForcedNeedToFire_reg
        combinedFiringTypes = nonForcedNeedToFire_da * global_var.SIG_TYPES['ISI'] + nonForcedNeedToFire_reg * global_var.SIG_TYPES['regular']
        
        self.n_firingTypes = np.where(combinedFiringTypes != 0, combinedFiringTypes, self.n_firingTypes)
        combinedPolarity = nonForcedNeedToFire_da_polarity + nonForcedNeedToFire_reg_polarity
        self.n_firingPolarity = np.where(combinedPolarity != 0, combinedPolarity, self.n_firingPolarity)
        self.n_firingTimeLeft = np.where(combinedFiringTypes != 0, self.n_sigDuration, self.n_firingTimeLeft)

        voltageClear = nonForcedNeedToFire == 0
        self.n_v = self.n_v * voltageClear
        self.n_daV = self.n_daV * voltageClear


    def triggerForceFiring(self, forceFiringIdx, forceFiringPolarity, forceSigType):
        self.n_forcedFiringType = np.where(forceFiringIdx != 0, forceSigType, self.n_forcedFiringType)
        self.n_firingTypes = np.where(forceFiringIdx != 0, forceSigType, self.n_firingTypes)
        self.n_firingPolarity = np.where(forceFiringIdx != 0, forceFiringPolarity, self.n_firingPolarity)
        self.n_firingTimeLeft = np.where(forceFiringIdx != 0, self.n_sigDuration, self.n_firingTimeLeft)
        self.n_refactoryTimeLeft = self.n_refactoryTimeLeft * (forceFiringIdx == 0)


    def getNeuronType(self):
        nonForedDATypes = (self.n_daV != 0) * (self.n_forcedFiringType == 0) * global_var.SIG_TYPES['ISI']
        nonForcedRegTypes = (self.n_daV == 0) * (self.n_v != 0) * (self.n_forcedFiringType == 0) * global_var.SIG_TYPES['regular']
        return nonForedDATypes + nonForcedRegTypes + self.n_forcedFiringType
    
    def getLowAmpNeurons(self, neuronsToCheck): # checked: 1
        sigTypes = self.getNeuronType()
        regTransmitterCleared = np.sum(self.s_regTransmitterRemain != 0, axis=1, keepdims=False) == 0
        regFiringNeurons =  (sigTypes == global_var.SIG_TYPES['regular']) * regTransmitterCleared * (np.absolute(self.n_v) < (self.n_calRemain / 5))
        regLowFiringNeurons = regFiringNeurons * ( (np.absolute(self.n_v_neg) < (self.n_calRemain_v_neg / 5)) + (self.n_v_pos < (self.n_calRemain_v_neg / 5)))
        daLowFiringNeurons = (np.absolute(self.n_v) < (self.n_calRemain / 5)) * (sigTypes == global_var.SIG_TYPES['ISI'])
        lowAmpNeurons = (regLowFiringNeurons + daLowFiringNeurons) * neuronsToCheck

        
        vNeeded = np.where((self.n_daV <0),  -1 - self.n_daV,  1 - self.n_daV) * (sigTypes == global_var.SIG_TYPES['ISI']) + np.where((self.n_v <0),  -1 - self.n_v,  1 - self.n_v) * (sigTypes == global_var.SIG_TYPES['regular'])
        self.modTempAmp(lowAmpNeurons, vNeeded)
        if self.childLayer:
            if self.childLayer.parentLayer != self:
                print('error , line 388')
            self.childLayer.modTempAmp(lowAmpNeurons, vNeeded, parentChecking = True)
        return lowAmpNeurons
    
    # Section: weight update
    def modTempAmp(self, lowAmpNeurons, vNeeded, parentChecking = False):
        
        daSynSig = self.s_dATransmitterRemain * (self.s_postSynReceptorAmp + self.s_ampMod)
        regSynSig = self.s_regTransmitterRemain * (self.s_postSynReceptorAmp + self.s_ampMod) * (daSynSig == 0)

        if parentChecking:
            lowAmpSynapses = ((np.absolute(daSynSig) < (self.s_calRemain / 4)) + (np.absolute(regSynSig) < (self.s_calRemain / 4))) * lowAmpNeurons
        else:
            lowAmpSynapses = ((np.absolute(daSynSig) < (self.s_calRemain / 4)) + (np.absolute(regSynSig) < (self.s_calRemain / 4))) * lowAmpNeurons.reshape(-1,1)
        
        
        daAdjustingSyn = (self.s_daCalRemain > self.s_calRemain) * lowAmpSynapses
        daAdjusingSynSign = (np.absolute(self.s_daCalRemain_pos) > np.absolute(self.s_daCalRemain_neg) + (np.absolute(self.s_daCalRemain_pos) < np.absolute(self.s_daCalRemain_neg)) * (-1)) * daAdjustingSyn
        regAdjustingSyn = (daAdjustingSyn == 0) * lowAmpSynapses
        regAdjusingSynSign = (np.absolute(self.s_calRemain_pos) > np.absolute(self.s_calRemain_neg) + (np.absolute(self.s_calRemain_pos) < np.absolute(self.s_calRemain_neg)) * (-1)) * regAdjustingSyn
        combinedAdjustingSign = regAdjusingSynSign + daAdjusingSynSign
        if parentChecking:
            ampUnsigned = safeDivide(np.absolute(self.s_calRemain) + self.s_daCalRemain, np.abs(self.parentLayer.n_calRemain)) * lowAmpSynapses * np.abs(vNeeded)
            amp = np.where((np.sign(vNeeded) == combinedAdjustingSign), 1, -1) * ampUnsigned
        else:
            ampUnsigned = safeDivide(np.absolute(self.s_calRemain) + self.s_daCalRemain, np.abs(self.n_calRemain).reshape(-1,1)) * lowAmpSynapses * np.abs(vNeeded).reshape(-1,1)
            amp = np.where((np.sign(vNeeded.reshape(-1,1)) == combinedAdjustingSign), 1, -1) * ampUnsigned
        self.s_ampMod = self.s_ampMod + amp


    def getNonActivingNeurons(self):
        return (self.n_refactoryTimeLeft + self.n_firingTimeLeft) == 0

    def triggerWeightUpdate(self):
        updateAmpAmount = self.updatePostSynAmp()
        self.updateTransmitterClearanceRate(updateAmpAmount)
        pass
    


    

    def updatePostSynAmp(self):
        if self.parentLayer is None:
            return 0
        maxAmp = 10
        remainingCal = self.s_calRemain + self.parentLayer.n_firingCalResidule
        if np.sum(np.abs(remainingCal)) == 0:
            return 0
        remainingCal = np.where(((remainingCal != 0) * (remainingCal < 0.01)), 0.01, remainingCal)
        signModifier = np.where((self.s_calRemain_neg > self.s_calRemain_pos), -1, 1) 
        updateAmp = self.s_updateAmp * signModifier
        updateAmount = np.clip((updateAmp / maxAmp), -1, 1) * remainingCal /5

        adapativeLr_pos = np.abs(1.5 - self.s_postSynReceptorAmp) / 3 * (updateAmount > 0)
        adapativeLr_neg = np.abs(-1.5 - self.s_postSynReceptorAmp) / 3 * (updateAmount < 0)


        updateAmount = updateAmount * (adapativeLr_pos + adapativeLr_neg)
        updateAmount = (updateAmount / self.s_ceilingLr * ((np.sign(updateAmount) == np.sign(self.s_postSynReceptorAmp)) * (updateAmount != 0))) + (updateAmount * self.s_ceilingLr * ((np.sign(updateAmount) != np.sign(self.s_postSynReceptorAmp)) * (updateAmount != 0)))
        
        
        updateAmount = np.where(((updateAmount < 0) * (np.absolute(updateAmount) < 0.01)), -0.01, updateAmount)
        updateAmount = np.where(((updateAmount > 0) * (np.absolute(updateAmount) < 0.01)), 0.01, updateAmount)
        updateAmount = updateAmount * (self.s_updateAmp != 0)
        updateAmount = updateAmount * 0.1

        if self.debug and global_var.isTraining and np.sum(np.abs(updateAmount)) != 0:
            print(self.s_postSynReceptorAmp, updateAmount, self.s_updateAmp, self.s_dATransmitterRemain)
        updatedAmp = self.s_postSynReceptorAmp + updateAmount
        self.s_ceilingLr = np.where((np.absolute(updatedAmp) > 1.4), (np.absolute(updatedAmp) / 1.4), 1)
        self.s_ceilingLr = np.where(self.s_ceilingLr == 0, 1, self.s_ceilingLr)

        self.s_postSynReceptorAmp += (updateAmount * global_var.LEARNING_RATE_ADOPTIVE)
        return updateAmount

    # Section: clearance rate update
    def updateTransmitterClearanceRate(self, updateAmpAmount):
        # return
        if np.sum(np.abs(updateAmpAmount)) == 0:
            return
        
        activityScore = np.maximum(0.1, safeDivide(1, self.s_calRemain))
        activityScore = np.where((self.s_regTransmitterRemain != 0) * (self.s_calRemain == 0), 0.1, activityScore)
        activityScore = np.where((self.s_regTransmitterRemain == 0) * (self.s_calRemain == 0), -1, activityScore)
        
        incrementedSyns = (np.sign(updateAmpAmount) == np.sign(self.s_postSynReceptorAmp)) * (np.sign(updateAmpAmount) != 0)
        decrementedSyns = (np.sign(updateAmpAmount) != np.sign(self.s_postSynReceptorAmp)) * (np.sign(updateAmpAmount) != 0)
        incScoreTotalPerNeuron = np.sum(activityScore * incrementedSyns, axis=1, keepdims=False)
        decScoreTotalPerNeuron = np.sum(activityScore * decrementedSyns, axis=1, keepdims=False)
        incTraceTotalPerNeuron = np.sum(self.n_releaseTrace * incrementedSyns, axis=1, keepdims=False)
        decTraceTotalPerNeuron = np.sum(self.n_releaseTrace * decrementedSyns, axis=1, keepdims=False)

        incrementNum = np.maximum(1, np.sum(incrementedSyns, axis=1, keepdims=False))
        decrementNum = np.maximum(1, np.sum(decrementedSyns, axis=1, keepdims=False))

        incScoreAvgPerNeuron = incScoreTotalPerNeuron / np.maximum(1, incrementNum) * (incrementNum != 0)
        decScoreAvgPerNeuron = decScoreTotalPerNeuron / np.maximum(1, decrementNum) * (decrementNum != 0)

        incTraceAvgPerNeuron = incTraceTotalPerNeuron / np.maximum(1, incrementNum) * (incrementNum != 0)
        decTraceAvgPerNeuron = decTraceTotalPerNeuron / np.maximum(1, decrementNum) * (decrementNum != 0)

        ampIncSynWithPeers = incrementedSyns * (incScoreAvgPerNeuron.reshape(-1,1) != 0) * (incrementNum > 1).reshape(-1,1)
        ampIncClippedTimeScore = np.minimum(activityScore * ampIncSynWithPeers, (incScoreAvgPerNeuron * 2).reshape(-1,1))
        ampIncSameDirAdj = safeDivide(incScoreAvgPerNeuron.reshape(-1,1) - ampIncClippedTimeScore, (incScoreAvgPerNeuron * 2).reshape(-1,1)) * np.square(self.s_transmitterClearanceRate)

        ampDecSynWithPeers = decrementedSyns * (decScoreAvgPerNeuron.reshape(-1,1) != 0) * (decrementNum > 1).reshape(-1,1)
        ampDecClippedTimeScore = np.minimum(activityScore , (decScoreAvgPerNeuron * 2).reshape(-1,1)) * ampDecSynWithPeers
        ampDecSameDirAdj = safeDivide(decScoreAvgPerNeuron.reshape(-1,1) - ampDecClippedTimeScore, (decScoreAvgPerNeuron * 2).reshape(-1,1)) * np.square(self.s_transmitterClearanceRate)
        sameDirAdj = (ampIncSameDirAdj * ampIncSynWithPeers + ampDecSameDirAdj * ampDecSynWithPeers) * 0.001
        sameDirAdj = np.sign(sameDirAdj) * 0.001
        


        roundedReleaseTrace = np.round(self.n_releaseTrace, 3)
        roundedDecTraceAvgPerNeuron = np.round(decTraceAvgPerNeuron.reshape(-1,1), 3)
        roundedIncTraceAvgPerNeuron = np.round(incTraceAvgPerNeuron.reshape(-1,1), 3)
        clearanceSquare = np.square(self.s_transmitterClearanceRate)
        inc_not_signle = np.sum((updateAmpAmount * incrementedSyns) != 0, axis=1).reshape(-1,1)
        dec_not_signle = np.sum((updateAmpAmount * decrementedSyns) != 0, axis=1).reshape(-1,1)

        oppositeDirAdj_inc_prolong = (roundedReleaseTrace <= decTraceAvgPerNeuron.reshape(-1,1)) * (activityScore >= decScoreAvgPerNeuron.reshape(-1,1)) * clearanceSquare * incrementedSyns   * (-1) * inc_not_signle
        oppositeDirAdj_inc_shorten = (roundedReleaseTrace >= decTraceAvgPerNeuron.reshape(-1,1)) * (activityScore <= decScoreAvgPerNeuron.reshape(-1,1)) * clearanceSquare * incrementedSyns * (oppositeDirAdj_inc_prolong == 0) * inc_not_signle
        oppositeDirAdj_dec_shorten = (roundedReleaseTrace <= incTraceAvgPerNeuron.reshape(-1,1)) * (activityScore <= incScoreAvgPerNeuron.reshape(-1,1)) * clearanceSquare * decrementedSyns * dec_not_signle
        oppositeDirAdj_dec_prolong = (roundedReleaseTrace >= incTraceAvgPerNeuron.reshape(-1,1)) * (activityScore >= incScoreAvgPerNeuron.reshape(-1,1)) * clearanceSquare * decrementedSyns * (oppositeDirAdj_dec_shorten == 0)  * (-1) * dec_not_signle

        oppositeDirAdj = (oppositeDirAdj_inc_prolong + oppositeDirAdj_inc_shorten + oppositeDirAdj_dec_prolong + oppositeDirAdj_dec_shorten) * 0.01


        optionalAdjustment = 0

        sameDirAdj= sameDirAdj * (inc_not_signle + dec_not_signle )

        self.s_transmitterClearanceRate = np.clip(self.s_transmitterClearanceRate + (sameDirAdj + oppositeDirAdj + optionalAdjustment) * global_var.LEARNING_RATE_ADOPTIVE, 0.01, 0.7)


    def addParentAndChildLayers(self, parentLayer, childLayer):
        self.parentLayer = parentLayer
        self.childLayer = childLayer

parameterNameMap = {'postSynReceptorAmp': 's_postSynReceptorAmp', 'transmitterClearanceRate': 's_transmitterClearanceRate'}

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

    def importParameters(self, loadPath):
        with open(loadPath, 'rb') as f:
            parameters = pkl.load(f)
            f.close()
        for attrName in parameters:
            for l, v in zip(self.layers, parameters[attrName]):
                setattr(l, parameterNameMap[attrName], np.array(v))
    
    def exportParameters(self, savePath):
        parameters = {}
        for attrName in parameterNameMap:
            networkRes = []
            for l in self.layers:
                networkRes.append(getattr(l, parameterNameMap[attrName]))
            parameters[attrName] = networkRes
        with open(savePath, 'wb') as f:
            pkl.dump(parameters, f)
            f.close()

    def run(self, dataset, train = False, useAccumulativeOutput = False, epoches = 1000, logFilePath='log.log'):
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
                        
                        global_var.isTraining = True
                        self.reset(keepCal=True)
                        correction = error 
                        for x in xs:
                            for i in range(self.periodsPerInput):
                                if i == 0:
                                    self.triggerAllActivities(x=x, correction = correction)
                                else:
                                    self.triggerAllActivities()
                                lastLayer = self.layers[-1]
                                lastFiringPos = (lastLayer.n_lastFiringTime == lastLayer.time) * (lastLayer.n_lastFirePolarity == 1)
                                correction = np.sign(correction - (y == 0) * lastFiringPos)
                        maxPeriod = 10
                        while not self.hasAllActivitiCeased() and maxPeriod > 0:
                            self.triggerAllActivities_postInput()
                            maxPeriod -= 1
                global_var.isTraining = False
            errorSumHist.append(errorSum)
            accuracy = (len(dataset) - errorNum) / len(dataset)
            if accuracy >= 0.5 and global_var.LEARNING_RATE_ADOPTIVE > 0.1:
                global_var.LEARNING_RATE_ADOPTIVE = 0.1
            if accuracy >= 0.9 and global_var.LEARNING_RATE_ADOPTIVE > 0.01:
                global_var.LEARNING_RATE_ADOPTIVE = 0.01
            print('epoch', e, 'acc', accuracy, 'errorSum', errorSum, '\t\t\t', end='\r')
            if accuracy == 1 or train == False:
                return accuracy, e, errorSum, errorSumHist
            
            for l in self.layers:
                l.bufferUpdate()
            
        return accuracy, epoches, errorSum, errorSumHist

    def triggerAllActivities(self, x=None, correction=None):
        for l in self.layers:
            l.updateState()
        if not (x is None):
            self.layers[0].triggerForceFiring(x, 1, global_var.SIG_TYPES['regular'])
        if correction is not None:
            self.layers[-1].triggerForceFiring(np.absolute(correction != 0), np.sign(correction), global_var.SIG_TYPES['ISI'])
        for l in self.layers:
            l.triggerNeuronFiring()
        for l in self.layers:
            l.triggerSynapseSendSig()
        for l in self.layers:
            l.triggerVoltageCheck()
        if global_var.isTraining:
            for l in self.layers:
                l.triggerWeightUpdate()

    def triggerAllActivities_postInput(self):
        for l in self.layers:
            l.updateState()
        for l in self.layers:
            l.triggerNeuronFiring()

    def hasAllActivitiCeased(self):
        for l in self.layers:
            if np.sum(l.n_firingTimeLeft) > 0:
                return False
        return True
    
    def reset(self, keepCal=False):
        for l in self.layers:
            l.reset(keepCal = keepCal)

    def getAccumulativeOutput(self):
        return self.layers[-1].n_timesFired_pos >= self.outputThreshold
                    
        
    def useFinalOutput(self, xs):
        acceptableTime = (len(xs) - 1) * self.periodsPerInput + len(self.layers)
        return (self.layers[-1].n_lastFirePolarity > 0) * (self.layers[-1].n_lastFiringTime >= acceptableTime) * 1
    
    def useArgMax(self):
        maxIndex = np.argmax(self.layers[-1].n_timesFired_pos, axis=None)
        y_hat = np.zeros(self.layers[-1].n_timesFired_pos.shape)
        y_hat[maxIndex] = 1

        if len(maxIndex.shape) > 1:
            print(maxIndex)
        return y_hat
    

