import argparse
import gym
import torch
import numpy as np
import random
import json
import gym_multi_car_racing
from tqdm import tqdm
import torch
from algo import off_policy_svg0, update
from networks import PolicyNetwork, CriticNetwork
from utils import *
import os
from pyglet.window import key
from torch.utils.tensorboard import SummaryWriter
import imageio
ACTION_DIM = 3  
NUM_AGENTS = 4

######## HYPERPARAMETERS ########
BATCH_SIZE = 64                
GAMMA = 0.99
POPULATION_SIZE = 12               
MUTATION_RATE = 0.1             
MUTATION_SCALE = 0.2            
INITIAL_RATING = 1200           
K = 32                                                                              
K_STEPS = 20
TSELECT = 0.35                  
LEARNING_RATE_CRITIC = 0.0002 #0.002 for 1 agent # 0.0008 for 4 agents #0.0002 for 12 agents
LEARNING_RATE_POLICY = [0.000002 + i * 0.000001 for i in range(POPULATION_SIZE)] #0.000005 for 1 agent and 0.000004 for 4 agents
#################################

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
preview = False
# TensorBoard
writer = SummaryWriter("logging/training")
def key_press(k, mod):
    """If the space bar is pressed, the scene will be rendered."""
    if k == key.SPACE:
        global preview
        preview = True


def key_release(k, mod):
    """If the space bar is released, the scene will not be rendered."""
    if k == key.SPACE:
        global preview
        preview = False

class CriticManager:
    def __init__(self, learning_rate_critic):
        self.critic_network = CriticNetwork(ACTION_DIM, learning_rate_critic).to(device)
        self.target_critic = CriticNetwork(ACTION_DIM, learning_rate_critic).to(device)
        self.target_critic.load_state_dict(self.critic_network.state_dict())

class Agent:
    def __init__(self, learning_rate_policy):
        self.policy_network = PolicyNetwork(ACTION_DIM, learning_rate_policy).to(device)
        self.target_policy = PolicyNetwork(ACTION_DIM, learning_rate_policy).to(device)
        self.replay_buffer = []
        self.frames_processed = 0
        self.eligible = False
        self.rating = INITIAL_RATING
        self.device = device

        self.target_policy.load_state_dict(self.policy_network.state_dict())

def evaluate_agents(agents, environment, population, critic_manager=None, eval=False, update_networks=False, generation=0):
    """ 
    Evaluate the agents and update their networks in the environment.
    
    Args:
        agents: The agents to evaluate.
        environment: The environment to evaluate the agents in.
        population: The population of agents.
        critic_manager: The critic manager that contains the critic networks.
        eval: True when evaluating the agents to render the environment and not update the networks.
        update_networks: Whether to update the networks or not, used to fill the replay buffer before updating the networks.
        
    Returns:
        The total reward obtained by each agent.
    """
    indexes = [population.index(agent) for agent in agents]
    print(f"Evaluating agents {indexes}...")

    total_reward = np.zeros(len(agents))
    environment.step(None)
    states = environment.reset()
    states = preprocess(states)
    done = False
    pbar = tqdm(total=1000)

    # GIF Recording: Initialize a list of lists for frames, one per agent
    if eval:
        agent_frames = [[] for _ in range(len(agents))]

    while not done:
        states_tensor = torch.tensor(states, dtype=torch.float32).permute(0, 3, 1, 2).to(device)
        actions = []
        # Get actions from the policy of each agent
        for agent, state_tensor in zip(agents, states_tensor):
            state_tensor = state_tensor.view(1, 3, 96, 96).to(device)

            action = agent.policy_network(state_tensor)
            action = action.cpu().detach().numpy()

            # Clip actions to the environment's action space
            for act in action:
                act[0] = np.clip(act[0], -1, 1)
                act[1] = np.clip(act[1], 0, 1)
                act[2] = np.clip(act[2], 0, 1)
            actions.append(action)
        
        # Step in the environment
        next_states, rewards, done, info = environment.step(np.array(actions))
        next_states = preprocess(next_states)

        if not eval:
            # For each agent, append the experience to the replay buffer, update the networks and empty the cache
            for agent, state, action, reward, next_state in zip(agents, states_tensor, actions, rewards, next_states):
                state = state.view(1, 3, 96, 96).cpu()
                next_state = torch.tensor(next_state, dtype=torch.float32).permute(2, 0, 1)
                next_state = next_state.view(1, 3, 96, 96)
                action = torch.tensor(action, dtype=torch.float32)

                agent.replay_buffer.append((state, action, reward, next_state))
                agent.frames_processed += 1
                update_replay_buffer(agent)
                
                if update_networks:
                    update(agent, population, GAMMA, BATCH_SIZE, critic_manager.critic_network, critic_manager.target_critic)
                    torch.cuda.empty_cache()

        if eval:
            # Render frames and save separately for each agent
            rendered_frames = environment.render(mode="rgb_array")
            for i, frame in enumerate(rendered_frames):
                agent_frames[i].append(frame)

        pbar.update(1)
        total_reward += rewards
        states = next_states

    pbar.close()
    avg_reward = np.mean(total_reward)
    
    # Log rewards to TensorBoard
    if eval:
        writer.add_scalar("Eval Reward", avg_reward, generation)
        
        # Save GIFs for each agent
        for i, frames in enumerate(agent_frames):
            gif_path = f"logging/eval_gif_agent_{i}_generation_{generation}.gif"
            with imageio.get_writer(gif_path, mode='I') as writer_gif:
                for frame in frames:
                    writer_gif.append_data(frame)
            print(f"Saved evaluation GIF for agent {i} to {gif_path}")
        
    else:
        writer.add_scalar("Training Reward", avg_reward, generation)
    
    return total_reward


def save_checkpoint(population, critic_manager):
    """ 
    Save the current state of the population.
    
    Args:
        population: The population of agents.
        critic_manager: The critic manager that contains the critic networks.
    """
    for agent in population:
        if not os.path.exists(f'checkpoint/agent{population.index(agent)}'):
            os.makedirs(f'checkpoint/agent{population.index(agent)}')
        torch.save(agent.policy_network, f'checkpoint/agent{population.index(agent)}/policy_network.pth')
        torch.save(agent.target_policy, f'checkpoint/agent{population.index(agent)}/target_policy_network.pth')
        torch.save(agent.policy_network.optimizer.state_dict(), f'checkpoint/agent{population.index(agent)}/optimizer.pth')

    torch.save(critic_manager.target_critic, f'checkpoint/target_critic_network.pth')
    torch.save(critic_manager.critic_network, f'checkpoint/critic_network.pth')

    ratings = [agent.rating for agent in population]
    with open('checkpoint/ratings.json', 'w') as f:
        json.dump(ratings, f)
    
    frame_counts = [agent.frames_processed for agent in population]
    with open('checkpoint/frame_counts.json', 'w') as f:
        json.dump(frame_counts, f)

    eligible = [agent.eligible for agent in population]
    with open('checkpoint/eligible.json', 'w') as f:
        json.dump(eligible, f)

def pbt_training(population, environment, generations, critic_manager, checkpoint=False, update_networks=False):
    """
    Train the population of agents using the Population Based Training algorithm.

    Args:
        population: The population of agents.
        environment: The environment to train the agents in.
        generations: The number of generations to train the agents for.
        critic_manager: The critic manager that contains the critic networks.
        checkpoint: True if we are loading a checkpoint, False otherwise.
        update_networks: True if we are updating the networks, False otherwise.
    """
    best_rewards = [-1000, -1000]

    for generation in range(generations):
        print(f"Generation {generation + 1}...")
        print(f"Current best rewards: {best_rewards[0]:.2f} || {best_rewards[1]:.2f}") 

        population_copy = population.copy()
        population_generation = []

        # If we load models from a checkpoint, we fill the replay buffers before starting updating the networks
        if checkpoint and generation > 3:
            update_networks = True

        # Choosing 4 random agents each time until the population is exhausted
        while len(population_copy) > 0:
            # Select NUM_AGENTS (4) random agents
            agents = random.sample(population_copy, NUM_AGENTS)
            population_generation.extend(agents)

            # Run episode on the selected agents
            rewards = evaluate_agents(agents, 
                                      environment, 
                                      population,
                                      critic_manager=critic_manager, 
                                      update_networks=update_networks,
                                      generation=generation)

            # Remove agents from the population copy
            population_copy = [agent for agent in population_copy if agent not in agents]
            
            # Collect rewards
            for agent, reward in zip(agents, rewards):
                agent.reward = reward

        # Save checkpoint
        save_checkpoint(population, critic_manager) 
        
        # Save the policy if we find a new best reward
        if not os.path.exists('best_models'):
            os.makedirs('best_models')

        for agent in population:
            if agent.reward > best_rewards[0]:
                print(f"New best reward found: {agent.reward}")
                best_rewards[1] = best_rewards[0]
                best_rewards[0] = agent.reward
                if best_rewards[1] > -1000:
                    old_best_policy = torch.load('best_models/best_policy_network0.pth')
                    torch.save(old_best_policy, 'best_models/best_policy_network1.pth')
                torch.save(agent.policy_network, 'best_models/best_policy_network0.pth')
                
            elif agent.reward > best_rewards[1]:
                print(f"New second best reward found: {agent.reward}")
                best_rewards[1] = agent.reward
                torch.save(agent.policy_network, 'best_models/best_policy_network1.pth')

        # For match results, update Elo ratings
        for i in range(0, len(population_generation), 4):
            # Given 4 agents (1,2,3,4) we have that:
            # Team 1 is always composed of agents 1 and 3
            # Team 2 is always composed of agents 2 and 4
            team1_reward = population_generation[i].reward + population_generation[i+2].reward
            team2_reward = population_generation[i+1].reward + population_generation[i+3].reward
            result = 'win' if team1_reward > team2_reward else 'lose' if team1_reward < team2_reward else 'draw'
            update_elo_rating(population_generation[i], population_generation[i+1], result, K)
            update_elo_rating(population_generation[i+2], population_generation[i+3], result, K)

        # Selection and Mutation
        for agent in population:
            if eligible(agent):
                agent2 = random.choice([a for a in population if a != agent])
                if eligible(agent2):
                    selo = 1 / (1 + 10 ** ((agent2.rating - agent.rating) / 400))
                    if selo < TSELECT:
                        print(f"Agent {population.index(agent)} is mutated copying agent {population.index(agent2)}")
                        agent.frames_processed = 0

                        agent.policy_network.load_state_dict(agent2.policy_network.state_dict())
                        agent.policy_network.optimizer.load_state_dict(agent2.policy_network.optimizer.state_dict())

                        agent.target_policy.load_state_dict(agent2.target_policy.state_dict())

                        mutate(agent.policy_network, MUTATION_RATE, MUTATION_SCALE)

def main():
    parser = argparse.ArgumentParser(description='Run training and evaluation')
    parser.add_argument('-t', '--train', action='store_true')
    parser.add_argument('-e', '--evaluate', action='store_true')
    parser.add_argument('-c', '--checkpoint', action='store_true')
    args = parser.parse_args()

    environment = gym.make("MultiCarRacing-v0", num_agents=NUM_AGENTS, direction='CCW', 
                           use_random_direction=True, backwards_flag=True, 
                           h_ratio=0.25, use_ego_color=True)
    environment.reset()

    for i in range(NUM_AGENTS):
        environment.viewer[i].window.on_key_press = key_press
        environment.viewer[i].window.on_key_release = key_release

    num_generations = 10000

    population = initialize_population(POPULATION_SIZE, LEARNING_RATE_POLICY)

    if not args.evaluate:
        critic_manager = CriticManager(LEARNING_RATE_CRITIC)

    if args.train:
        if args.checkpoint:
            print("Loading checkpoint...")               
            ratings = json.load(open('checkpoint/ratings.json'))
            frame_counts = json.load(open('checkpoint/frame_counts.json'))
            eligible = json.load(open('checkpoint/eligible.json'))

            for agent in population:
                agent.policy_network = torch.load(f'checkpoint/agent{population.index(agent)}/policy_network.pth').to(device)
                agent.target_policy = torch.load(f'checkpoint/agent{population.index(agent)}/target_policy_network.pth').to(device)
                agent.policy_network.optimizer.load_state_dict(torch.load(f'checkpoint/agent{population.index(agent)}/optimizer.pth'))
                agent.rating = ratings[population.index(agent)]
                agent.frames_processed = frame_counts[population.index(agent)]
                agent.eligible = eligible[population.index(agent)]

            critic_manager.critic_network = torch.load('checkpoint/critic_network.pth').to(device)
            critic_manager.target_critic = torch.load('checkpoint/target_critic_network.pth').to(device)

            pbt_training(population, environment, num_generations, critic_manager, checkpoint=True)
        else:
            pbt_training(population, environment, num_generations, critic_manager, update_networks=True)

    if args.evaluate:
        if args.checkpoint:
            agents = random.sample(population, NUM_AGENTS)
            for agent in agents:
                agent.policy_network = torch.load(f'checkpoint/agent{population.index(agent)}/policy_network.pth').to(device)
                agent.policy_network.eval()
            rewards = evaluate_agents(agents, environment, population, eval=True)
        else:
            for agent in population[:NUM_AGENTS]:
                agent.policy_network = torch.load(f'best_models/best_policy_network{population.index(agent)}.pth').to(device)
                agent.policy_network.eval()
            rewards = evaluate_agents(population[:NUM_AGENTS], environment, population, eval=True)

        if rewards[0] + rewards[1] > rewards[1] + rewards[3]:
            print("Team Blue won!")
        else:
            print("Team Red won!")

if __name__ == "__main__":
    main()
