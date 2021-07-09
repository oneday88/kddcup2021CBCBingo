from pathlib import Path
import pickle, os, sys
from collections import deque

import random
import numpy as np
import pandas as pd
import gym

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_

from prioritized_memory import Memory

from agent.RLModels import DQNControl,DQNControl2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
path = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(path)
import gym_cfg
with open(path + "/gym_cfg.py", "r") as f:
    pass

class TestAgent():
    def __init__(self):
        ### The default parameters
        self.greenSec = 2
        self.maxPhase = 8
         ### The temporal tabular for decision making loop
          ### The temporal tabular for decision making loop
        self.phaseLaneMapIn = np.array([[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]])-1
        self.phaseLaneMapOut = np.array([[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]])-1
        self.agentIdList = []
        self.nowPhase = {}
        self.agentIdLabelDict = {}
        self.lastChangeStep = {}
        
        # the roadnet info
        self.phasePassableLane = {}
        self.intersections = {}
        self.roads = {}
        self.agents = {}
        self.agentLaneSpeedLimit = {}
        self.agentLaneLength = {}
        self.laneInfoDict = {}

        ### The memory for training
        #self.memory = deque(maxlen=30000)
        self.memory = Memory(80000)
        self.histBatchState = {}
        ### The model updating framework
        self.learningStart = 20
        self.updateModelFreq = 1
        self.updateTargetModelFreq = 17

        ### The parameters for the DQN model
        self.gamma = 0.8   # The discount rate
        self.epsilon = 0.2 # The exploration rate
        self.epsilonMin = 0.01
        self.epsilonDecay = 0.9995
        self.learningRate = 0.00005
        self.batchSize = 256
        self.ObsLength = 1+1+72+105+48
        self.agentSize = 1004
        self.stepSize = 36
        self.actionSpace = 8
        
        # should load the model while submit
        self.model = DQNControl2(self.agentSize,self.actionSpace,self.stepSize, self.ObsLength, self.actionSpace).to(device)
        self.targetModel = DQNControl2(self.agentSize,self.actionSpace,self.stepSize, self.ObsLength, self.actionSpace).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learningRate)
        self.criterion = torch.nn.SmoothL1Loss()
        self.loadModel(self.model, "Params","epoch_Traffic100_3_metrics_-35775.393.param") #1.4232
    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    ################################
    def load_agent_list(self,agentIdList):
        #self.agentIdList = [int(i) for i in agentIdList]
        self.agentIdList = agentIdList
        self.nowPhase = dict.fromkeys(self.agentIdList,1)
        self.lastChangeStep = dict.fromkeys(self.agentIdList,0)
        for i in range(len(self.agentIdList)):
            agentId = self.agentIdList[i]
            self.agentIdLabelDict[agentId] = i
    
    def reset(self,agentIdList):
        self.histState = np.zeros((self.agentSize, 40))
        self.load_agent_list(agentIdList)

    def load_roadnet(self,intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        """
        Get the lane limited speed and lane length
        """
        laneInfoDict = {}
        for key,value in self.roads.items():
            subSpeedLimit = value['speed_limit']
            subLength = value['length']
            subLaneDict = value['lanes']
            for lane in subLaneDict.keys():
                laneInfoDict[lane] = (subSpeedLimit, subLength)
        self.laneInfoDict = laneInfoDict

        for key in self.agentIdList:
            try:
                subLanes=self.intersections[int(key)]['lanes']
            except:
                print(self.intersections[int(key)])
                continue
            subAgentSpeedLimit = np.zeros(24)
            subAgentLength = np.zeros(24)
            for i in range(24):
                singleLane = subLanes[i]
                if(singleLane != -1):
                    subLaneSpeedLimit, subLaneLength = laneInfoDict[singleLane]
                    subAgentSpeedLimit[i] = max(subLaneSpeedLimit,4)
                    subAgentLength[i] = subLaneLength
                else:
                    subAgentSpeedLimit[i],subAgentLength[i]=15,-1
            self.agentLaneSpeedLimit[key] = np.array(subAgentSpeedLimit)
            self.agentLaneLength[key] = np.array(subAgentLength)

    ################################
    def act(self, obs):
        """ !!! MUST BE OVERRIDED !!!
        """
        # here obs contains all of the observations and infos
        actions = {}
        # observations is returned 'observation' of env.step()
        # info is returned 'info' of env.step()
        observations = obs['observations']
        info = obs['info']

        nowStep = info['step']

        observationsForAgent, infoForAgent = self.preprocessing(observations, info)

        batchState = self.getBatchStates(nowStep, observationsForAgent, infoForAgent)
        actions = self.act_(nowStep, batchState)
        return actions

    def preprocessing(self, observations, info):
        observationsForAgent = {}
        for key,val in observations.items():
            agentId = key
            agentObs = val['observation']
            observationsForAgent[agentId] = agentObs
        
        infoForAgent = {}
        if(len(info)>1):
            for k,v in info.items():
                if k =='step': continue
                carId = k
                subLaneId = int(v['drivable'][0])
                speed = v['speed'][0]
                distance = v['distance'][0]
                if(subLaneId not in infoForAgent.keys()):
                    infoForAgent[subLaneId] = {}
                    infoForAgent[subLaneId]['carId'] = [carId]
                    infoForAgent[subLaneId]['speed'] = [speed]
                    infoForAgent[subLaneId]['distance'] = [distance]
                else:
                    infoForAgent[subLaneId]['carId'] += [carId]
                    infoForAgent[subLaneId]['speed'] += [speed]
                    infoForAgent[subLaneId]['distance'] += [distance]

        return observationsForAgent, infoForAgent


    def act_(self, nowStep, batchState):
        actions = {}
        # The NN Qvalue based on the DQN models
        NNQValues, NNQValuesTwo = self.model(torch.FloatTensor((batchState)))
        NNQValues = (NNQValues).cpu().detach().numpy()
        ###Actions choose the max vehicle
        for i in range(len(self.agentIdList)):
            # Get the previous action
            agentId = self.agentIdList[i]
            subForbiddenPases = self.getForbiddenPhases(self.phasePassableLane[agentId])
            # The sub observation dict
            # Set a minimum green light time
            stepDiff = nowStep - self.lastChangeStep[agentId]
            # Do switching if exceed the greenSec
            currentAction = self.nowPhase[agentId]
            currentState = batchState[i,:]
            prevAreaQueuePressure100 =currentState[100:108]
            prevAreaVehiceNum100 = currentState[60:68]
            prevAreaQueueNum100 = currentState[76:84]
            prevDownStreamQueueNum100 = prevAreaQueueNum100-prevAreaQueuePressure100
            prevAreaVehiceNum60 = currentState[108:116]
            prevAreaDelayNum60 = currentState[116:124]
            prevAreaQueueNum60 = currentState[124:132]
            prevAreaQueuePressure60 =currentState[148:156]
            prevDownStreamQueueNum60 = prevAreaQueueNum60-prevAreaQueuePressure60
            flag = (stepDiff >= self.greenSec)
            flag2 = prevAreaQueueNum60[currentAction-1]==0
            flag3 = prevDownStreamQueueNum60[currentAction-1]>=8
            flag4 = prevAreaQueuePressure100[currentAction-1]<= -5
            if(flag or flag2 or flag3 or flag4):
                currentAction = self.nowPhase[agentId]
                currentState = batchState[i,:]
                subQValues = NNQValues[i,:]
                currentState = batchState[i,:]
                nnAction = self.getGreedyTrainAction(currentAction,currentState, subQValues)
                policyAction = nnAction
                # update the last change step
                if(policyAction != self.nowPhase[agentId]):
                    self.lastChangeStep[agentId] = nowStep
                # update the phase and action
                actions[agentId] = policyAction
                self.nowPhase[agentId] = policyAction
                if(i==90):
                    print("**************QValues, PredRewards, currentAction, QAction**************")
                    print(subQValues)
                    print(NNQValuesTwo[i,:])
                    print(currentAction)
                    print(nnAction)
            else:
                actions[agentId] = self.nowPhase[agentId]
        return actions

    def getGreedyTrainAction(self, currentAction,  currentState, NNQValues, isPrint=False):
        upStreamVehicleVec = currentState[12:20]
        action1 = np.argmax(upStreamVehicleVec)+1

        p = np.random.rand()
        if p < self.epsilon:
            action = np.random.randint(1, self.actionSpace) # random choose from 1~8
            return action
        if p < 2*self.epsilon:
            return action1
        else:
            winners = np.argwhere(NNQValues == np.amax(NNQValues))
            if(len(winners)==1):
                return winners[0][0]+1
            winnerActions = np.array(winners.flatten().tolist())+1
            return np.random.choice(winnerActions,1,replace=False)[0]


    def getGreedySubmitAction(self, currentAction, subForbiddenPases,currentState,  NNQValue, NNQValue2, NNQValue3, NNQValue4, NNQValue5):

        QvalueIndex = np.argsort(NNQValue).argsort()
        QvalueIndex2 = np.argsort(NNQValue2).argsort()
        QvalueIndex3 = np.argsort(NNQValue3).argsort()
        QvalueIndex4 = np.argsort(NNQValue4).argsort()
        QvalueIndex5 = np.argsort(NNQValue5).argsort()
        decisionValue = QvalueIndex+QvalueIndex4+QvalueIndex5*2

        upStreamVehicleVec = currentState[60:68]+currentState[68:76]

        upStreamIndex =  np.argsort(upStreamVehicleVec)[::-1]

        maxValue = -100000
        NNAction = -1
        
        flag = np.max(currentState[68:76])>=6
        if not flag:
            maxValue = -100000
            action1 = -1
            for i in range(8):
                if(i  in subForbiddenPases): continue
                if(upStreamVehicleVec[i]>maxValue):
                    maxValue = upStreamVehicleVec[i]
                    action1 = i
            if(upStreamVehicleVec[currentAction-1]*1.03>=upStreamVehicleVec[action1]):
                return currentAction
            #assert action1  == -1
            return action1+1
        else:
            maxValue = -100000
            NNAction = -1
            for i in range(8):
                if(i  in subForbiddenPases): continue
                if(i in upStreamIndex[6:]): continue
                if(decisionValue[i]>maxValue):
                    maxValue = decisionValue[i]
                    NNAction = i
            if(NNAction == -1):
                return np.argmax(decisionValue)+1
            else:
                return NNAction+1

    """
    (1) current phase, current step, and features
    """
    def getBatchStates(self, nowStep, observationsForAgent, infoForAgent):
        # The numpy array
        agentLen = len(self.agentIdList)
        phaseLen = self.maxPhase
        # Now step Vec
        agentLabelVec = np.zeros((agentLen,1))
        nowStepVec = np.zeros((agentLen,1))
        stepDiffVec = np.zeros((agentLen,1))
        agentNowPhaseVec = np.zeros((agentLen,1))
        #
        # total num
        agentAllVehicleNumVec = np.zeros((agentLen,phaseLen))
        agentAllVehiclePressureVec = np.zeros((agentLen,phaseLen))
        agentUpstreamAvalLaneNumVec = np.zeros((agentLen,phaseLen))
        agentDownstreamAvalLaneNumVec = np.zeros((agentLen,phaseLen))

        agentAreaVehicleNumVec200 = np.zeros((agentLen,phaseLen))
        agentAreaDelayNumVec200 = np.zeros((agentLen,phaseLen))
        agentAreaQueueNumVec200 = np.zeros((agentLen,phaseLen))
        agentAreaVehiclePressureVec200 = np.zeros((agentLen,phaseLen))
        agentAreaDelayPressureVec200 = np.zeros((agentLen,phaseLen))
        agentAreaQueuePressureVec200 = np.zeros((agentLen,phaseLen))
        
        agentAreaVehicleNumVec100 = np.zeros((agentLen,phaseLen))
        agentAreaDelayNumVec100 = np.zeros((agentLen,phaseLen))
        agentAreaQueueNumVec100 = np.zeros((agentLen,phaseLen))
        agentAreaVehiclePressureVec100 = np.zeros((agentLen,phaseLen))
        agentAreaDelayPressureVec100 = np.zeros((agentLen,phaseLen))
        agentAreaQueuePressureVec100 = np.zeros((agentLen,phaseLen))
        
        agentAreaVehicleNumVec60 = np.zeros((agentLen,phaseLen))
        agentAreaDelayNumVec60 = np.zeros((agentLen,phaseLen))
        agentAreaQueueNumVec60 = np.zeros((agentLen,phaseLen))
        agentAreaVehiclePressureVec60 = np.zeros((agentLen,phaseLen))
        agentAreaDelayPressureVec60 = np.zeros((agentLen,phaseLen))
        agentAreaQueuePressureVec60 = np.zeros((agentLen,phaseLen))


        for i in range(agentLen):
            ### The agentId and dictionary
            agentId = self.agentIdList[i]
            agentLabelVec[i] = i
            ### The observations and info for the agentId
            obsDict = observationsForAgent[agentId]
            ### the now phase
            nowPhase= self.nowPhase[agentId]
            agentNowPhaseVec[i] = nowPhase
            ### The now step and step diff
            ###
            stepDiff = nowStep - self.lastChangeStep[agentId]
            # The feature for NN model
            nowStepVec[i] = int(nowStep/100)
            stepDiffVec[i] = 1.0*stepDiff/10
            ### The number and speed
            laneVehicleNumVec = np.array(obsDict[0:24])
            laneSpeedVec = np.array(obsDict[24:48])
            forbiddenLaneVec = (laneVehicleNumVec == -1)
            self.phasePassableLane[agentId] = forbiddenLaneVec
            ### The information of cars
            lanes = self.intersections[int(agentId)]['lanes']
            #print(self.agentLaneSpeedLimit)
            laneSpeedLimit = 1.0* self.agentLaneSpeedLimit[agentId]
            laneLength = 1.0* self.agentLaneLength[agentId]
            laneSpeedLimit[laneSpeedLimit<=0]   =  15
            """
            Get th area infos
            """
            areaVehicleNumVec60,areaDelayNumVec60,areaQueueNumVec60,areaVehicleSpeedVec60,areaDelayIndexVec60,areaVehicleDensityVec60 = self.getAgentAreaInfo(60, 60*0.8,lanes,laneLength, laneVehicleNumVec,laneSpeedLimit,infoForAgent)
            areaVehicleNumVec100,areaDelayNumVec100,areaQueueNumVec100,areaVehicleSpeedVec100,areaDelayIndexVec100,areaVehicleDensityVec100 = self.getAgentAreaInfo(100, 100*0.8,lanes,laneLength, laneVehicleNumVec,laneSpeedLimit,infoForAgent)
            areaVehicleNumVec200,areaDelayNumVec200,areaQueueNumVec200,areaVehicleSpeedVec200,areaDelayIndexVec200,areaVehicleDensityVec200 = self.getAgentAreaInfo(200, 200*0.8,lanes,laneLength, laneVehicleNumVec,laneSpeedLimit,infoForAgent)
            # iterate 8 phases
            phaseUpstreamAllVehicleNumVec = np.zeros(phaseLen)
            phaseAllVehiclePressureVec = np.zeros(phaseLen)
            phaseUpstreamAvalLaneNumVec = np.zeros(phaseLen)
            phaseDownstreamAvalLaneNumVec = np.zeros(phaseLen)
            
            phaseUpstreamAreaVehicleNumVec200 = np.zeros(phaseLen)
            phaseUpstreamAreaDelayNumVec200 = np.zeros(phaseLen)
            phaseUpstreamAreaQueueNumVec200 = np.zeros(phaseLen)
            phaseAreaVehiclePressureVec200 = np.zeros(phaseLen)
            phaseAreaDelayPressureVec200 = np.zeros(phaseLen)
            phaseAreaQueuePressureVec200 = np.zeros(phaseLen)

            phaseUpstreamAreaVehicleNumVec100 = np.zeros(phaseLen)
            phaseUpstreamAreaDelayNumVec100 = np.zeros(phaseLen)
            phaseUpstreamAreaQueueNumVec100 = np.zeros(phaseLen)
            phaseAreaVehiclePressureVec100 = np.zeros(phaseLen)
            phaseAreaDelayPressureVec100 = np.zeros(phaseLen)
            phaseAreaQueuePressureVec100 = np.zeros(phaseLen)

            phaseUpstreamAreaVehicleNumVec60 = np.zeros(phaseLen)
            phaseUpstreamAreaDelayNumVec60 = np.zeros(phaseLen)
            phaseUpstreamAreaQueueNumVec60 = np.zeros(phaseLen)
            phaseAreaVehiclePressureVec60 = np.zeros(phaseLen)
            phaseAreaDelayPressureVec60 = np.zeros(phaseLen)
            phaseAreaQueuePressureVec60 = np.zeros(phaseLen)

            for j in range(phaseLen):
                inLanes = self.phaseLaneMapIn[j]
                outLanes = self.phaseLaneMapOut[j]
                subPhaseUpstreamAllVehicleNum = 0.
                subPhaseDownstreamAllVehicleNum = 0.

                subPhaseUpstreamAvalLaneNum = 0
                subPhaseDownstreamAvalLaneNum = 0

                subPhaseUpstreamAreaVehicleNum200 = 0.
                subPhaseUpstreamAreaDelayNum200 = 0.
                subPhaseUpstreamAreaQueueNum200 = 0.
                subPhaseDownstreamAreaVehicleNum200 = 0.
                subPhaseDownstreamAreaDelayNum200 = 0.
                subPhaseDownstreamAreaQueueNum200 = 0.
                
                subPhaseUpstreamAreaVehicleNum100 = 0.
                subPhaseUpstreamAreaDelayNum100 = 0.
                subPhaseUpstreamAreaQueueNum100 = 0.
                subPhaseDownstreamAreaVehicleNum100 = 0.
                subPhaseDownstreamAreaDelayNum100 = 0.
                subPhaseDownstreamAreaQueueNum100 = 0.
                
                subPhaseUpstreamAreaVehicleNum60 = 0.
                subPhaseUpstreamAreaDelayNum60 = 0.
                subPhaseUpstreamAreaQueueNum60 = 0.
                subPhaseDownstreamAreaVehicleNum60 = 0.
                subPhaseDownstreamAreaDelayNum60 = 0.
                subPhaseDownstreamAreaQueueNum60 = 0.


                for inLane in inLanes:
                    subPhaseUpstreamAllVehicleNum += max(laneVehicleNumVec[inLane],0)
                    subPhaseUpstreamAvalLaneNum += forbiddenLaneVec[inLane]
                    # 200
                    subPhaseUpstreamAreaVehicleNum200 += areaVehicleNumVec200[inLane]
                    subPhaseUpstreamAreaDelayNum200 += areaDelayNumVec200[inLane]
                    subPhaseUpstreamAreaQueueNum200 += areaQueueNumVec200[inLane]
                    # 100
                    subPhaseUpstreamAreaVehicleNum100 += areaVehicleNumVec100[inLane]
                    subPhaseUpstreamAreaDelayNum100 += areaDelayNumVec100[inLane]
                    subPhaseUpstreamAreaQueueNum100 += areaQueueNumVec100[inLane]
                    # 60
                    subPhaseUpstreamAreaVehicleNum60 += areaVehicleNumVec60[inLane]
                    subPhaseUpstreamAreaDelayNum60 += areaDelayNumVec60[inLane]
                    subPhaseUpstreamAreaQueueNum60 += areaQueueNumVec60[inLane]
                for outLane in outLanes:
                    subPhaseDownstreamAllVehicleNum += max(laneVehicleNumVec[outLane],0)
                    subPhaseDownstreamAvalLaneNum += forbiddenLaneVec[outLane]
                    # 200
                    subPhaseDownstreamAreaVehicleNum200 += areaVehicleNumVec200[outLane]
                    subPhaseDownstreamAreaDelayNum200 += areaDelayNumVec200[outLane]
                    subPhaseDownstreamAreaQueueNum200 += areaQueueNumVec200[outLane]
                    # 100
                    subPhaseDownstreamAreaVehicleNum100 += areaVehicleNumVec100[outLane]
                    subPhaseDownstreamAreaDelayNum100 += areaDelayNumVec100[outLane]
                    subPhaseDownstreamAreaQueueNum100 += areaQueueNumVec100[outLane]
                    # 60
                    subPhaseDownstreamAreaVehicleNum60 += areaVehicleNumVec60[outLane]
                    subPhaseDownstreamAreaDelayNum60 += areaDelayNumVec60[outLane]
                    subPhaseDownstreamAreaQueueNum60 += areaQueueNumVec60[outLane]
                ##The total
                subPhaseAllVehiclePressure = subPhaseUpstreamAllVehicleNum-subPhaseDownstreamAllVehicleNum/3
                # 200
                subPhaseAreaVehiclePressure200 = subPhaseUpstreamAreaVehicleNum200-subPhaseDownstreamAreaVehicleNum200/3
                subPhaseAreaDelayPressure200 = subPhaseUpstreamAreaDelayNum200-subPhaseDownstreamAreaDelayNum200/3
                subPhaseAreaQueuePressure200 = subPhaseUpstreamAreaQueueNum200-subPhaseDownstreamAreaQueueNum200/3
                # 100
                subPhaseAreaVehiclePressure100 = subPhaseUpstreamAreaVehicleNum100-subPhaseDownstreamAreaVehicleNum100/3
                subPhaseAreaDelayPressure100 = subPhaseUpstreamAreaDelayNum100-subPhaseDownstreamAreaDelayNum100/3
                subPhaseAreaQueuePressure100 = subPhaseUpstreamAreaQueueNum100-subPhaseDownstreamAreaQueueNum100/3
                # 60
                subPhaseAreaVehiclePressure60 = subPhaseUpstreamAreaVehicleNum60-subPhaseDownstreamAreaVehicleNum60/3
                subPhaseAreaDelayPressure60 = subPhaseUpstreamAreaDelayNum60-subPhaseDownstreamAreaDelayNum60/3
                subPhaseAreaQueuePressure60 = subPhaseUpstreamAreaQueueNum60-subPhaseDownstreamAreaQueueNum60/3

                phaseUpstreamAllVehicleNumVec[j] = subPhaseUpstreamAllVehicleNum
                phaseUpstreamAvalLaneNumVec[j] = subPhaseUpstreamAvalLaneNum
                phaseDownstreamAvalLaneNumVec[j] = subPhaseDownstreamAvalLaneNum

                phaseUpstreamAreaVehicleNumVec200[j] = subPhaseUpstreamAreaVehicleNum200
                phaseUpstreamAreaDelayNumVec200[j] = subPhaseUpstreamAreaDelayNum200
                phaseUpstreamAreaQueueNumVec200[j] = subPhaseUpstreamAreaQueueNum200
                phaseAreaVehiclePressureVec200[j] = subPhaseAreaVehiclePressure200
                phaseAreaDelayPressureVec200[j] = subPhaseAreaDelayPressure200
                phaseAreaQueuePressureVec200[j] = subPhaseAreaQueuePressure200
                
                phaseUpstreamAreaVehicleNumVec100[j] = subPhaseUpstreamAreaVehicleNum100
                phaseUpstreamAreaDelayNumVec100[j] = subPhaseUpstreamAreaDelayNum100
                phaseUpstreamAreaQueueNumVec100[j] = subPhaseUpstreamAreaQueueNum100
                phaseAreaVehiclePressureVec100[j] = subPhaseAreaVehiclePressure100
                phaseAreaDelayPressureVec100[j] = subPhaseAreaDelayPressure100
                phaseAreaQueuePressureVec100[j] = subPhaseAreaQueuePressure100
                
                phaseUpstreamAreaVehicleNumVec60[j] = subPhaseUpstreamAreaVehicleNum60
                phaseUpstreamAreaDelayNumVec60[j] = subPhaseUpstreamAreaDelayNum60
                phaseUpstreamAreaQueueNumVec60[j] = subPhaseUpstreamAreaQueueNum60
                phaseAreaVehiclePressureVec60[j] = subPhaseAreaVehiclePressure60
                phaseAreaDelayPressureVec60[j] = subPhaseAreaDelayPressure60
                phaseAreaQueuePressureVec60[j] = subPhaseAreaQueuePressure60

            if(i==9900):
                print("Debuging--phase aggregation ----")
                print(phaseUpstreamAllVehicleNumVec)
                print(phaseUpstreamAreaVehicleNumVec200)
                print(phaseUpstreamAreaDelayNumVec200)
                print(phaseUpstreamAreaQueueNumVec200)
                print(phaseAreaVehiclePressureVec200)
                print(phaseAreaDelayPressureVec200)
                print(phaseAreaQueuePressureVec200)
            # iterate 8 phases
            agentAllVehicleNumVec[i] = phaseUpstreamAllVehicleNumVec
            agentAllVehiclePressureVec[i] =phaseAllVehiclePressureVec

            agentAreaVehicleNumVec200[i] = phaseUpstreamAreaVehicleNumVec200
            agentAreaDelayNumVec200[i] = phaseUpstreamAreaDelayNumVec200
            agentAreaQueueNumVec200[i] = phaseUpstreamAreaQueueNumVec200
            agentAreaVehiclePressureVec200[i] =phaseAreaVehiclePressureVec200
            agentAreaDelayPressureVec200[i] = phaseAreaDelayPressureVec200
            agentAreaQueuePressureVec200[i] = phaseAreaQueuePressureVec200
            
            agentAreaVehicleNumVec100[i] = phaseUpstreamAreaVehicleNumVec100
            agentAreaDelayNumVec100[i] = phaseUpstreamAreaDelayNumVec100
            agentAreaQueueNumVec100[i] = phaseUpstreamAreaQueueNumVec100
            agentAreaVehiclePressureVec100[i] =phaseAreaVehiclePressureVec100
            agentAreaDelayPressureVec100[i] = phaseAreaDelayPressureVec100
            agentAreaQueuePressureVec100[i] = phaseAreaQueuePressureVec100
            
            agentAreaVehicleNumVec60[i] = phaseUpstreamAreaVehicleNumVec60
            agentAreaDelayNumVec60[i] = phaseUpstreamAreaDelayNumVec60
            agentAreaQueueNumVec60[i] = phaseUpstreamAreaQueueNumVec60
            agentAreaVehiclePressureVec60[i] =phaseAreaVehiclePressureVec60
            agentAreaDelayPressureVec60[i] = phaseAreaDelayPressureVec60
            agentAreaQueuePressureVec60[i] = phaseAreaQueuePressureVec60
            
            agentUpstreamAvalLaneNumVec[i] = phaseUpstreamAvalLaneNumVec
            agentDownstreamAvalLaneNumVec[i] = phaseDownstreamAvalLaneNumVec

        if(nowStep==0): self.histState = np.zeros((self.agentSize, 48))
        # delay number
        currentOne = np.concatenate((agentAreaVehicleNumVec100,agentAreaDelayNumVec100,agentAreaQueueNumVec100, agentAreaVehiclePressureVec100,agentAreaDelayPressureVec100,agentAreaQueuePressureVec100), axis=1)
        histIncrement = currentOne-self.histState

        batchState = np.concatenate((agentLabelVec, agentNowPhaseVec, nowStepVec, stepDiffVec, agentAllVehicleNumVec,\
        agentAreaVehicleNumVec200,agentAreaDelayNumVec200,agentAreaQueueNumVec200,agentAreaVehiclePressureVec200,agentAreaDelayPressureVec200,agentAreaQueuePressureVec200,\
        agentAreaVehicleNumVec100,agentAreaDelayNumVec100,agentAreaQueueNumVec100, agentAreaVehiclePressureVec100,agentAreaDelayPressureVec100,agentAreaQueuePressureVec100,\
        agentAreaVehicleNumVec60,agentAreaDelayNumVec60,agentAreaQueueNumVec60,agentAreaVehiclePressureVec60,agentAreaDelayPressureVec60,agentAreaQueuePressureVec60,\
        agentAllVehiclePressureVec, agentUpstreamAvalLaneNumVec, agentDownstreamAvalLaneNumVec, histIncrement), axis=1)

        self.histState = currentOne
        return  batchState

    def getForbiddenPhases(self, forbiddenLaneVec):
        forbiddenPases = []
        for i in range(self.maxPhase):
            phaseId = i
            subInLane = self.phaseLaneMapIn[i]
            subOutLane =self.phaseLaneMapOut[i]
            if sum(forbiddenLaneVec[subInLane]) >=2:
                #print("-----inLane-----")
                #print(sum(forbiddenLaneVec[subInLane]))
                forbiddenPases.append(phaseId)
            if sum(forbiddenLaneVec[subOutLane]) >=3:
                #print("-----outLane-----")
                #print(sum(forbiddenLaneVec[subInLane]))
                forbiddenPases.append(phaseId)
        return list(set(forbiddenPases))

    def getRewards(self, mark, upstreamThreshold, downstreamThreshold, observationsForAgent, nextObservationForAgent, infoForAgent,nextInfoForAgent, eRewardDict,rwds, isWeight=True):
        assert mark in set(['DQ','pressure','traffic'])
        rewardsDict = {}
        rewardsStats = 0.
        agentLen = len(self.agentIdList)
        for i in range(agentLen):
            ### The agentId and dictionary
            agentId = self.agentIdList[i]
            ### The observations and info for the agentId
            obsDict = observationsForAgent[agentId]
            nextObsDict = nextObservationForAgent[agentId]
            ### The number and speed
            laneVehicleNumVec = np.array(obsDict[0:24])
            nextLaneVehicleNumVec = np.array(nextObsDict[0:24])
            forbiddenLaneVec = (laneVehicleNumVec == -1)
            
            lanes = self.intersections[int(agentId)]['lanes']
            laneSpeedLimit = 1.0*(self.agentLaneSpeedLimit[agentId])
            laneLength = 1.0*(self.agentLaneLength[agentId])
            laneSpeedLimit[laneSpeedLimit<=0]   =  15
            """
            Get th area infos
            """
            areaVehicleNumVec,areaDelayNumVec,areaQueueNumVec,areaVehicleSpeedVec,areaDelayIndexVec, areaVehicleDensityVec = self.getAgentAreaInfo(upstreamThreshold, downstreamThreshold,lanes,laneLength, laneVehicleNumVec,laneSpeedLimit,infoForAgent)
            nextAreaVehicleNumVec, nextAreaDelayNumVec,nextAreaQueueNumVec,nextAreaVehicleSpeedVec,nextAreaDelayIndexVec, nextAreaVehicleDensityVec = self.getAgentAreaInfo(upstreamThreshold, downstreamThreshold,lanes,laneLength, nextLaneVehicleNumVec,laneSpeedLimit,nextInfoForAgent)
            if mark=='DQ':
                subReward = np.sum(nextAreaDelayNumVec[:24])+np.sum(nextAreaQueueNumVec[:24])
                subReward = subReward/48
            elif mark=='pressure':
                subReward1 = np.sum(nextAreaDelayNumVec[:12])+np.sum(nextAreaQueueNumVec[:12])
                subReward2 = np.sum(nextAreaDelayNumVec[12:24])+np.sum(nextAreaQueueNumVec[12:24])
                subReward = (subReward1-subReward2/2)/48
            elif mark=='traffic':
                subReward1 = np.sum(nextAreaDelayNumVec[:12])+np.sum(nextAreaQueueNumVec[:12])
                subReward2 = np.sum(nextAreaQueueNumVec[12:24])-np.sum(areaQueueNumVec[12:24])
                subReward3 = np.sum(nextAreaDelayNumVec[12:24])-np.sum(areaDelayNumVec[12:24])
                subReward = (subReward1+subReward2+subReward3)/48
            pressure = -1.0 * subReward
            if agentId in rewardsDict:
                rewardsDict[agentId] += pressure
            else:
                rewardsDict[agentId] = pressure
            laneVehicleNumVec[laneVehicleNumVec<0] = 0
            rewardsStats += pressure
            eRewardDict[agentId] += pressure
        return rewardsDict, eRewardDict, rewardsStats

    def getAgentAreaInfo(self, upstreamThreshold, downstreamThreshold,lanes, laneLength, laneVehicleNumVec, laneSpeedLimit, infoForAgent):
        ### The delay and queue
        areaVehicleNumVec = np.zeros(24)
        areaDelayNumVec = np.zeros(24)
        areaQueueNumVec = np.zeros(24)
        areaVehicleSpeedVec = np.copy(laneSpeedLimit)
        areaDelayIndexVec = np.zeros(24)
        areaVehicleDensityVec = np.zeros(24)
        for j in range(24):
            if(laneVehicleNumVec[j]<=0): continue
            distThreshold = upstreamThreshold if j <12 else downstreamThreshold
            subLane = lanes[j]
            subLaneLength = laneLength[j]
            
            subLaneSpeedLimit = laneSpeedLimit[j]*1.0
            subCarSpeedList = np.array(infoForAgent[subLane]['speed'])
            subCarDistanceList = np.array(infoForAgent[subLane]['distance'])
            areaVehiceNum, areaDelayNum, areaQueueNum,areaMeanSpeed, areaDelayIndex,areaVehicleDensity = self.getLaneAreaInfo(upstreamThreshold, downstreamThreshold, j, subLaneLength, subLaneSpeedLimit, subCarSpeedList, subCarDistanceList)
                #assign the new value
            areaVehicleNumVec[j] = areaVehiceNum
            areaDelayNumVec[j] = areaDelayNum
            areaQueueNumVec[j] = areaQueueNum
            areaVehicleSpeedVec[j] = areaMeanSpeed
            areaDelayIndexVec[j] = areaDelayIndex
            areaVehicleDensityVec[j] = areaVehicleDensity
            
        return areaVehicleNumVec,areaDelayNumVec,areaQueueNumVec,areaVehicleSpeedVec,areaDelayIndexVec,areaVehicleDensityVec

    def getLaneAreaInfo(self, upstreamThreshold, downstreamThreshold, laneIndex, laneLen, subLaneSpeedLimit, carSpeedList, carDistanceList):
        #print("***************Debug*********************")
        thresholdDist = downstreamThreshold
        if(laneIndex<12):
            thresholdDist = upstreamThreshold
            carDistanceList = laneLen-carDistanceList
        areaIndex = carDistanceList<thresholdDist
        delayIndex = carSpeedList/subLaneSpeedLimit
        delayIndex[delayIndex<0] = 0
        delayIndex[delayIndex>1] = 1

        areaVehiceNum = sum(areaIndex)
        areaSpeedMean = 1.0*subLaneSpeedLimit if areaVehiceNum==0  else np.mean(carSpeedList[areaIndex])
        areaDelayNum = areaVehiceNum-sum((delayIndex)[areaIndex])
        areaQueueNum = sum(carSpeedList[areaIndex]<0.3)
        areaDelyIndex = 0 if areaVehiceNum==0  else areaDelayNum/areaVehiceNum
        areaVehicleDensity = -1 if laneLen==-1 else areaVehiceNum/min(laneLen,thresholdDist)
        return areaVehiceNum, areaDelayNum, areaQueueNum, areaSpeedMean, areaDelyIndex,areaVehicleDensity


    def updateTargetNetwork(self):
        self.targetModel.load_state_dict(self.model.state_dict())

    def remember(self, state, action, reward, nextState,done):
        #self.memory.append((state, action, reward, nextState))
        self.memory.add(1, (state, action, reward, nextState, done))

    def replay(self):
        # update the Q network from the memory buffer
        if self.batchSize > self.memory.size():
            #minibatch = self.memory
            return
        else:
            #minibatch = random.sample(self.memory, self.batchSize)
            try:
                batch, idxs, is_weights = self.memory.sample(self.batchSize)
            #minibatch = random.sample(self.memory, self.batchSize)
            #print(f'len batch is {len(batch)}, batch 256 is {batch[255]}')
                batchStates, batchActions, batchRewards, batchNextStates, batch_done = zip(*batch)
            except:
                return

        #batchStates, batchActions, batchRewards, batchNextStates = [np.vstack(x) for x in np.array(minibatch).T]
        batchStates = torch.FloatTensor((batchStates)).to(device)
        batchActions = torch.FloatTensor(batchActions).unsqueeze(1).to(device)
        batchRewards = torch.FloatTensor(batchRewards).unsqueeze(1).to(device)
        batchNextStates = torch.FloatTensor(batchNextStates).to(device)

        # The rewards normalization
        with torch.no_grad():
            QNextValue,QNextValue2 = self.targetModel(batchNextStates)
            QTarget = batchRewards + self.gamma* torch.amax(QNextValue, axis=1).reshape(-1,1)
        #print(batchStates)
        #print(batchActions)
        #print(batchActions.long()-1)
        #print(self.model(batchStates))
        #print(self.model(batchStates).shape)
        #print(QTarget)
        QValue, QValue2 = self.model(batchStates)
        YPred, YPred2 = QValue.gather(1, batchActions.long()-1), QValue2.gather(1, batchActions.long()-1)
        ##The TD error
        errors = torch.abs(YPred - QTarget).data.numpy()
        for i in range(self.batchSize):
            idx = idxs[i]
            self.memory.update(idx, errors[i])
        loss1 = (torch.FloatTensor(is_weights) * F.smooth_l1_loss(YPred, QTarget)).mean()
        loss2 = (torch.FloatTensor(is_weights) * F.smooth_l1_loss(YPred2, batchRewards)).mean()
        #loss = loss1+loss2
        loss = loss1
        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()
        # decreasing the epsilon
        if self.epsilon > self.epsilonMin:
            self.epsilon *= self.epsilonDecay

        print("epsilon = {}".format(self.epsilon))

    def saveModel(self, mark, reward):
        savePath = os.path.join(path, "Params")
        if not os.path.exists(savePath):
            os.makedirs(savePath)
        filename = os.path.join(savePath, "epoch_{}_metrics_{:.3f}".format(mark, reward))
        filename += '.param'
        torch.save(self.model.state_dict(), filename)

    def initWeights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)

    def loadModel(self, currentModel, pathDir, filename=None):
        if(filename==None):
            print("Initialization")
            currentModel.apply(self.initWeights)
        else:
            print("load from "+filename)
            savePath = os.path.join(path, pathDir)
            filename = os.path.join(savePath,filename)
            currentModel.load_state_dict(torch.load(filename))

scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()

