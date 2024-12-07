
import numpy as np
import torch
import imageio
import os

def batchify_obs(obs, device):
    """Converts PZ style observations to batch of torch arrays."""
    # convert to list of np arrays
    obs = np.stack([obs[a] for a in obs], axis=0)
    # transpose to be (batch, channel, height, width)
    obs = obs.transpose(0, -1, 1, 2)
    # convert to torch
    obs = torch.tensor(obs).to(device)

    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x


def render_and_save_gif(env, agent, 
                        num_pistons, comm_action, comm_size_compressed,
                        gif_path="rendered_episode.gif", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                        ):
    """Renders one episode, saves it as a GIF, and returns helpful info."""
    frames = []
    total_reward = 0
    agent.eval()

    with torch.no_grad():
        obs, infos = env.reset(seed=None)
        obs = batchify_obs(obs, device)
        terms = [False]
        truncs = [False]
        while not any(terms) and not any(truncs):
            ids = torch.arange(num_pistons).unsqueeze(-1).to(device)
            comm = agent.get_comm(obs, ids)
            if comm is not None and comm_action:
                comm_batch = comm.to(device)
            else:
                comm_batch = torch.zeros((obs.shape[0], comm_size_compressed)).to(device)

            actions, logprobs, _, values = agent.get_action_and_value(obs, ids, comm=comm_batch)
            obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
            obs = batchify_obs(obs, device)
            terms = [terms[a] for a in terms]
            truncs = [truncs[a] for a in truncs]

            # Capture the frame
            frame = env.render()
            frames.append(frame)

            # Accumulate rewards
            total_reward += sum(rewards.values())

    # Save the frames as a GIF
    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, fps=30)

    return {
        "total_reward": total_reward,
        "num_frames": len(frames),
        "gif_path": gif_path
    }