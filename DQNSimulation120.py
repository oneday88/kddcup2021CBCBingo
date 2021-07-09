import json
import traceback
import argparse
import logging
import os
import sys
import time
from pathlib import Path
import re,random
import gym
import numpy as np

from agent.DQNAgent import TestAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

gym.logger.setLevel(gym.logger.ERROR)


def pretty_files(path):
    contents = os.listdir(path)
    return "[{}]".format(", ".join(contents))


def resolve_dirs(root_path: str, input_dir: str = None, output_dir: str = None, log_dir:str=None):
    root_path = Path(root_path)

    logger.info(f"root_path={pretty_files(root_path)}")

    if input_dir is not None:
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        log_dir = Path(log_dir)
        submission_dir = input_dir
        scores_dir = output_dir

        logger.info(f"input_dir={pretty_files(input_dir)}")
        logger.info(f"output_dir={pretty_files(output_dir)}")
        logger.info(f"log_dir={pretty_files(log_dir)}")
    else:
        raise ValueError('need input dir')

    if not scores_dir.exists():
        os.makedirs(scores_dir)

    logger.info(f"submission_dir={pretty_files(submission_dir)}")
    logger.info(f"scores_dir={pretty_files(scores_dir)}")

    if not submission_dir.is_dir():
        logger.warning(f"submission_dir={submission_dir} does not exist")

    return submission_dir, scores_dir, log_dir


def load_agent_submission(submission_dir: Path):
    logger.info(f"files under submission dir:{pretty_files(submission_dir)}")

    # find agent.py
    module_path = None
    cfg_path = None
    class_path = None
    for dirpath, dirnames, file_names in os.walk(submission_dir):
        for file_name in [f for f in file_names if f.endswith(".py")]:
            if file_name == "agent.py":
                module_path = dirpath

            if file_name == "gym_cfg.py":
                cfg_path = dirpath

            if file_name == 'CBEngine_round3.py':
                class_path = dirpath
    # error
    assert (
        module_path is not None
    ), "Cannot find file named agent.py, please check your submission zip"
    assert(
        cfg_path is not None
    ), "Cannot find file named gym_cfg.py, please check your submission zip"
    assert (
        class_path is not None
    ), "Cannot find file named CBEngine_round3.py, please check your submission zip"
    sys.path.append(str(module_path))


    # This will fail w/ an import error of the submissions directory does not exist
    import gym_cfg as gym_cfg_submission
    #import agent as agent_submission
    from CBEngine_round3 import CBEngine_round3 as CBEngine_rllib_class

    gym_cfg_instance = gym_cfg_submission.gym_cfg()

    #return  agent_submission.agent_specs,gym_cfg_instance,CBEngine_rllib_class
    return gym_cfg_instance,CBEngine_rllib_class

def read_config(cfg_file):
    configs = {}
    with open(cfg_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if(len(line) == 3 and line[0][0] != '#'):
                configs[line[0]] = line[-1]
    return configs

def process_roadnet(roadnet_file):
    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id

    intersections = {}
    roads = {}
    agents = {}

    agent_num = 0
    road_num = 0
    signal_num = 0
    with open(roadnet_file, 'r') as f:
        lines = f.readlines()
        cnt = 0
        pre_road = 0
        is_obverse = 0
        for line in lines:
            line = line.rstrip('\n').split(' ')
            if ('' in line):
                line.remove('')
            if (len(line) == 1):
                if (cnt == 0):
                    agent_num = int(line[0])
                    cnt += 1
                elif (cnt == 1):
                    road_num = int(line[0]) * 2
                    cnt += 1
                elif (cnt == 2):
                    signal_num = int(line[0])
                    cnt += 1
            else:
                if (cnt == 1):
                    intersections[int(line[2])] = {
                        'have_signal': int(line[3]),
                        'end_roads': [],
                        'start_roads': [],
                        'lanes':[]
                    }
                elif (cnt == 2):
                    if (len(line) != 8):
                        road_id = pre_road[is_obverse]
                        roads[road_id]['lanes'] = {}
                        for i in range(roads[road_id]['num_lanes']):
                            roads[road_id]['lanes'][road_id * 100 + i] = list(map(int, line[i * 3:i * 3 + 3]))
                        is_obverse ^= 1
                    else:
                        roads[int(line[-2])] = {
                            'start_inter': int(line[0]),
                            'end_inter': int(line[1]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[4]),
                            'inverse_road': int(line[-1])
                        }
                        roads[int(line[-1])] = {
                            'start_inter': int(line[1]),
                            'end_inter': int(line[0]),
                            'length': float(line[2]),
                            'speed_limit': float(line[3]),
                            'num_lanes': int(line[5]),
                            'inverse_road': int(line[-2])
                        }
                        intersections[int(line[0])]['end_roads'].append(int(line[-1]))
                        intersections[int(line[1])]['end_roads'].append(int(line[-2]))
                        intersections[int(line[0])]['start_roads'].append(int(line[-2]))
                        intersections[int(line[1])]['start_roads'].append(int(line[-1]))
                        pre_road = (int(line[-2]), int(line[-1]))
                else:
                    # 4 out-roads
                    signal_road_order = list(map(int, line[1:]))
                    now_agent = int(line[0])
                    in_roads = []
                    for road in signal_road_order:
                        if (road != -1):
                            in_roads.append(roads[road]['inverse_road'])
                        else:
                            in_roads.append(-1)
                    in_roads += signal_road_order
                    agents[now_agent] = in_roads
    for agent, agent_roads in agents.items():
        intersections[agent]['lanes'] = []
        for road in agent_roads:
            ## here we treat road -1 have 3 lanes
            if (road == -1):
                for i in range(3):
                    intersections[agent]['lanes'].append(-1)
            else:
                for lane in roads[road]['lanes'].keys():
                    intersections[agent]['lanes'].append(lane)

    return intersections, roads, agents



"""
The system configuration
"""
# arg parse
parser = argparse.ArgumentParser(prog="evaluation",description="1")
parser.add_argument("--input_dir",help="The path to the directory containing the reference data and user submission data.",default='agent',type=str,)
parser.add_argument("--output_dir",help="The path to the directory where the submission's scores.txt file will be written to.", default='out',type=str,)
parser.add_argument("--sim_cfg",help='The path to the simulator cfg',default = "cfg/simulator_round3_flow0.cfg",type=str)
parser.add_argument("--metric_period", help="period of scoring", default= 3600,  type=int)
parser.add_argument("--vehicle_info_path",help="path to log vehicle info to scoring",default='./log',type=str) 
 

logger.info("\n")
logger.info("*" * 40)

##The parameters
args = parser.parse_args()
simulator_cfg_file = args.sim_cfg

# get gym instance

submission_dir, scores_dir, log_dir = resolve_dirs(os.path.dirname(__file__), args.input_dir, args.output_dir, args.vehicle_info_path)
gym_cfg,CBEngine_rllib_class = load_agent_submission(submission_dir)
scenario = ['test']

simulator_configs = read_config(simulator_cfg_file)


gym_configs = gym_cfg.cfg
env_config = {
        "simulator_cfg_file": simulator_cfg_file,
        "thread_num": 4,
        "gym_dict": gym_configs,
        "metric_period": 200,
        "vehicle_info_path": "/starter-kit/log/"
    }
env = CBEngine_rllib_class(env_config)
scenario = ['test']


agent = TestAgent()
# read roadnet file, get data
roadnet_path = Path(simulator_configs['road_file_addr'])
intersections, roads, agents = process_roadnet(roadnet_path)
# env.set_warning(0)
env.set_log(0)
env.set_info(1)
# env.set_ui(0)
# get agent instance
infos = {'step':0}

observations = env.reset()
agent_id_list = []
for k in observations.keys():
    agent_id_list.append(k)
agent_id_list = list(set(agent_id_list))
agent.load_agent_list(agent_id_list)
#agent = agent_spec[scenario[0]]
agent.load_roadnet(intersections, roads, agents)
dones = {}
dones['__all__']=False

"""
The training framework
"""
totalDecisionNum = 0
saveModelFreq = 1
numEpisodes = 50
totalRewards =  np.zeros(numEpisodes)

for e in range(numEpisodes):
    # read roadnet file, get data
    randomIndex = random.randint(0,10)
    sub_simulator_cfg_file = "cfg/simulator_round3_flow"+str(randomIndex)+".cfg"
    print(sub_simulator_cfg_file)
    print(gym_configs)
    simulator_configs = read_config(sub_simulator_cfg_file)
    gym_configs = gym_cfg.cfg
    env_config = {
        "simulator_cfg_file": sub_simulator_cfg_file,
        "thread_num": 4,
        "gym_dict": gym_configs,
        "metric_period": 200,
        "vehicle_info_path": "/starter-kit/log/"
    }
    print(env_config)
    env = CBEngine_rllib_class(env_config)
    # env.set_warning(0)
    env.set_log(0)
    env.set_info(1)

    roadnet_path = Path(simulator_configs['road_file_addr'])
    intersections, roads, agents = process_roadnet(roadnet_path)
    agent.load_agent_list(agent_id_list)
    agent.load_roadnet(intersections, roads, agents)
    print("----------------------------------------------------{}/{}".format(e, numEpisodes))
    eDecisionNum = 0
    #The reward and reward dictionary
    eCumReward = 0
    eRewardDict = {}
    for agentId in agent_id_list:
        eRewardDict[agentId] = 0.
    observations = env.reset()
    step = 0
    sim_start = time.time()
    actions = {}
    observations, rewards, dones, infos = env.step(actions)
    infos['step'] = 0

    observationForAgent, infoForAgent = agent.preprocessing(observations, infos)
    states = agent.getBatchStates(step, observationForAgent,infoForAgent)
    while not dones['__all__']:
        step+=1
        """
        all_info = {
            'observations':observations,
            'info':infos
        }
        actions = agent.act(all_info)
        """
        prevActions = dict(agent.nowPhase)
        actions = agent.act_(step-1, states)
        observations_, rwds, dones, infos_ = env.step(actions)
        infos_['step'] = step
        nextObservationForAgent, nextInfoForAgent = agent.preprocessing(observations_,infos_)
        nextStates = agent.getBatchStates(step, nextObservationForAgent,nextInfoForAgent)
        ### The rewards
        upstreamDist = 120
        rewards, eRewardDict, rewardsStats = agent.getRewards("traffic",upstreamDist,upstreamDist*0.8, observationForAgent, nextObservationForAgent, infoForAgent, nextInfoForAgent, eRewardDict,rwds,True)
        # update the cumulative episode reward for training
        eCumReward += rewardsStats
        # The memory buffer for the model updating
        sampleCount = 0
        for i in range(len(agent_id_list)):
            agentId = agent_id_list[i]
            agent.remember(states[i,:], actions[agentId], rewards[agentId],nextStates[i,:],dones[str(agentId)])
            if(i==90):
                subReward = rewards[agentId]
                print(subReward)
            sampleCount += 1
        totalDecisionNum += 1
        # updating for the next loop
        prevActions = actions
        observations = observations_
        infos = infos_
        observationForAgent = nextObservationForAgent
        infoForAgent = nextInfoForAgent
        states = nextStates
        ### update the policy weights using the data from memory
        if totalDecisionNum > agent.learningStart and totalDecisionNum % agent.updateModelFreq == agent.updateModelFreq -1:
            print("agent replay")
            for k in range(10):agent.replay()
        if totalDecisionNum > agent.learningStart and totalDecisionNum % agent.updateTargetModelFreq == agent.updateTargetModelFreq -1:
            print(step)
            print("update Target network")
            agent.updateTargetNetwork()
        if all(dones.values()):
            break
    ### Save the model
    print(eRewardDict)
    if e % saveModelFreq == 0:
        mark = "DQN120"+"_"+str(e)
        agent.saveModel(mark, eCumReward)
    ### The rewards
    totalRewards[e] = eCumReward
    logger.info("Mean episode Reward: {}".format(eCumReward))
    sim_end = time.time()
    logger.info("simulation cost : {}s".format(sim_end-sim_start))
