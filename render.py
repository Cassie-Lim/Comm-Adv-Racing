import torch
import imageio
import numpy as np
from pettingzoo.butterfly import pistonball_v6
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from pistonball import Agent, NUM_PISTONS
import pistonball
COMM_ACTION = False
COMM_STATES = False
pistonball.COMM_SIZE = NUM_PISTONS
if COMM_STATES:
    pistonball.COMM_SIZE += 512
# Set up the environment
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = pistonball_v6.parallel_env(render_mode="rgb_array", continuous=False)
env = color_reduction_v0(env)
env = resize_v1(env, 64, 64)
env = frame_stack_v1(env, stack_size=4)
model_path = "models/2024-11-29 23:37:48.084824_ca_False.pt"  # Replace with your model path
checkpoint = torch.load(model_path)
num_agents = len(env.possible_agents)
num_actions = env.action_space(env.possible_agents[0]).n
agent = Agent(num_actions=num_actions)  # Replace with the number of actions
agent.load_state_dict(checkpoint['agent_state_dict'])
agent.eval().to(device)
def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    obs = np.stack([obs[a] for a in obs], axis=0)
    obs = obs.transpose(0, -1, 1, 2)
    obs = torch.tensor(obs).to(device)
    return obs

def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x

# Render and save GIFs
def render_and_save_gif(agent, env, num_episodes=5, gif_path="rendered_episode.gif"):
    frames = []
    for episode in range(num_episodes):
        obs, infos = env.reset(seed=None)
        obs = batchify_obs(obs, device)
        terms = [False]
        truncs = [False]
        while not any(terms) and not any(truncs):
            if COMM_ACTION:
                comm = agent.get_comm(obs)
            else:
                comm = None
                comm_batch = None
            if comm is not None and COMM_ACTION:
                comm_batch = torch.tile(comm.view((1, pistonball.COMM_SIZE)), (obs.shape[0], 1)).to(device)
            else:
                comm_batch = torch.zeros((obs.shape[0], NUM_PISTONS)).to(device)

            actions, logprobs, _, values = agent.get_action_and_value(obs, action=None, comm=comm_batch)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            obs = batchify_obs(obs, device)
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]

            # Capture the frame
            frame = env.render()
            frames.append(frame)

    # Save the frames as a GIF
    imageio.mimsave(gif_path, frames, fps=30)

# Call the function to render and save the GIF
render_and_save_gif(agent, env, num_episodes=5, gif_path="rendered_episode.gif")