import minerl
import gym
import pygame
import pickle
import os
import imageio
import time
from pathlib import Path
import numpy as np


def handle_input(task, env, keys, paused):
    action = env.action_space.noop()

    if keys[pygame.K_p]:
        if not paused:
            paused = True
        else:
            paused = False

    if keys[pygame.K_1]:
        action["jump"] = 1
        action["forward"] = 1
    elif keys[pygame.K_2]:
        action["attack"] = 1
    elif keys[pygame.K_3]:
        action["left"] = 1
    elif keys[pygame.K_4]:
        action["right"] = 1

    # # camera:
    # if keys[pygame.K_UP]:
    #     action["camera"][0] = -9.0
    #
    # if keys[pygame.K_DOWN]:
    #     action["camera"][0] = 9.0
    #
    # if keys[pygame.K_RIGHT]:
    #     action["camera"][1] = 9.0
    #
    # if keys[pygame.K_LEFT]:
    #     action["camera"][1] = -9.0
    #
    # # movement
    # if keys[pygame.K_w]:
    #     action["forward"] = 1
    #
    # if keys[pygame.K_s]:
    #     action["back"] = 1
    #
    # if keys[pygame.K_a]:
    #     action["left"] = 1
    #
    # if keys[pygame.K_d]:
    #     action["right"] = 1
    #
    # if keys[pygame.K_SPACE]:
    #     action["jump"] = 1
    #
    # if keys[pygame.K_LCTRL]:
    #     action["sneak"] = 1
    #
    # if keys[pygame.K_LSHIFT]:
    #     action["sprint"] = 1
    #
    # if keys[pygame.K_q]:
    #     action["attack"] = 1

    # # place
    # if (task != 'MineRLTreechop-v0'):
    #     if keys[pygame.K_x]:
    #         action["place"] = 1 #dirt

    # if task in obtain_tasks:
    #     if keys[pygame.K_c]:
    #         action["place"] = 2 #stone
    #
    #     if keys[pygame.K_v]:
    #         action["place"] = 3 #cobblestone
    #
    #     if keys[pygame.K_b]:
    #         action["place"] = 4 #crafting_table
    #
    #     if keys[pygame.K_n]:
    #         action["place"] = 5 #furnace
    #
    #     if keys[pygame.K_m]:
    #         action["place"] = 6 #torch
    #
    #     # craft
    #
    #
    #     # nearbyCraft
    #     if keys[pygame.K_e]:
    #         action["nearbyCraft"] = 1 #wooden_axe
    #
    #     if keys[pygame.K_r]:
    #         action["nearbyCraft"] = 2 #wooden_pickaxe
    #
    #     if keys[pygame.K_t]:
    #         action["nearbyCraft"] = 3 #stone_axe
    #
    #     if keys[pygame.K_z]:
    #         action["nearbyCraft"] = 4 #stone_pickaxe
    #
    #     if keys[pygame.K_u]:
    #         action["nearbyCraft"] = 5 #iron_axe
    #
    #     if keys[pygame.K_i]:
    #         action["nearbyCraft"] = 6 #iron_pickaxe
    #
    #     if keys[pygame.K_o]:
    #         action["nearbyCraft"] = 7 #furnace
    #
    #     # nearbySmelt
    #     if keys[pygame.K_f]:
    #         action["nearbySmelt"] = 1 #iron_ingot
    #
    #     if keys[pygame.K_g]:
    #         action["nearbySmelt"] = 2 #coal
    #
    #     # equip
    #     if keys[pygame.K_5]:
    #         action["equip"] = 1 #wooden_axe
    #
    #     if keys[pygame.K_6]:
    #         action["equip"] = 2 #wooden_pickaxe
    #
    #     if keys[pygame.K_7]:
    #         action["equip"] = 3 #stone_axe
    #
    #     if keys[pygame.K_8]:
    #         action["equip"] = 4 #stone_pickaxe
    #
    #     if keys[pygame.K_9]:
    #         action["equip"] = 5 #iron_axe
    #
    #     if keys[pygame.K_0]:
    #         action["equip"] = 6 #iron_pickaxe


    return action, paused

def dict_to_string(dictionary):
    dict_str = ""
    for x in dictionary:
        dict_str += "{0}: {1}, ".format(x,dictionary[x])
    return dict_str


# parameters for logging
# episode_log = {}
# episode_id = time.strftime("%x").replace('/', '_') + '__' + time.strftime("%X")
# episode_path = './data/episode_' + episode_id + '/images/'

# try:
#     os.makedirs(episode_path)
# except:
#     pass

ARCH_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk")
COMPUTE_JAVA_PATH = Path(os.environ["HOME"], "jdk8u222-b10")
UBUNTU_JAVA_PATH = Path("/usr/lib/jvm/java-8-openjdk-amd64")

if ARCH_JAVA_PATH.is_dir():
    os.environ["JAVA_HOME"] = str(ARCH_JAVA_PATH)
elif COMPUTE_JAVA_PATH.is_dir():
    os.environ["JAVA_HOME"] = str(COMPUTE_JAVA_PATH)
elif UBUNTU_JAVA_PATH.is_dir():
    os.environ["JAVA_HOME"] = str(UBUNTU_JAVA_PATH)
else:
    print(f"No Java 8 instance was found on this machine!")
    sys.exit(1)

os.environ["PATH"] = os.environ["JAVA_HOME"] + "/bin:" + os.environ["PATH"]


# set up environment
# task_list = ['MineRLTreechop-v0', 'MineRLNavigateDense-v0', 'MineRLNavigate-v0', 'MineRLNavigateExtremeDense-v0', 'MineRLNavigateExtreme-v0', 'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0', 'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0']
# obtain_tasks = ['MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0', 'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0']
# task = input("Enter one of the following task ids:\n {0}\n".format(task_list))
# while task not in task_list:
#     print("Task does not exist, try again with on of: ", task_list)
#     task = input("Enter task: ")

task = 'MineRLTreechop-v0'
env = gym.make('MineRLTreechop-v0')
env.seed(420)
env.reset()

pygame.init()
pygame.display.set_mode((100, 100))


done = False
step = 0
episode_reward = 0

paused = False

t = time.time()

while True:
    env.render(mode="human")
    keys = pygame.key.get_pressed()
#    print('np.argmax(keys)', np.argmax(keys))
    hits = 0
    while paused:
        keys = pygame.key.get_pressed()
        _, paused = handle_input(task, env, keys, paused)
        action = env.action_space.noop()
        action["attack"] = 1
        env.step(action)
        hits = hits + 1
        print("{0} hits.".format(hits))
        env.render(mode="human")
        time.sleep(2)
        
    action, paused = handle_input(task, env, keys, paused)
    pygame.event.pump()
    obs, reward, done, info = env.step(action)

    image = obs['pov']
    # imageio.imwrite(episode_path + '{:05d}.png'.format(step), image)

    # episode_log[step] = "Done: {0}, Step: {1}, Reward: {2}, Actions: {3} Time: {4}".format(done, step, reward, dict_to_string(action), time.time()- t) #[done, step, reward, dict_to_string(action), time.time() - t]

    step += 1
    episode_reward += reward
    print("step: ", step, " total reward: ", episode_reward)

# with open('./data/episode_' + episode_id + '/episode_log.pkl', 'wb') as handle:
#     pickle.dump(episode_log, handle, protocol=pickle.HIGHEST_PROTOCOL)

# print('Log written to ./data/episode_episode' + episode_id + '/')

print("Episode reward: ", episode_reward)
