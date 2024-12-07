import matplotlib.pyplot as plt
import seaborn as sns
from agent import Agent
import torch
from supersuit import color_reduction_v0, frame_stack_v1, resize_v1
from pettingzoo.butterfly import pistonball_v6
from utils import batchify_obs, batchify, render_and_save_gif, unbatchify
# Simulate agent features
comm_action = True
num_pistons = 20
comm_size = 512 + 1
comm_size_compressed = 32
hidden_dim = 64
output_dim = 32
n_heads = 4
frame_size = (64, 64)
max_cycles = 125
stack_size = 4
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load trained model (or create a new one)
state_dict_path = "/home/cassie/Workspace/DRL/Comm-Adv-Racing/models/2024-12-07 16:19:02.883092_ca_True.pt"
state_dict = torch.load(state_dict_path, weights_only=True)
agent = Agent(num_actions=3, comm_size=comm_size, num_pistons=num_pistons, device=device, comm_size_compressed=output_dim).to(device)
agent.load_state_dict(state_dict['agent_state_dict'])

env = pistonball_v6.parallel_env(
    render_mode="rgb_array", time_penalty=-0.5,
    continuous=False,
    max_cycles=max_cycles,
    ball_mass=5.0,
    ball_friction=0.5,
    ball_elasticity=0.75
)
env = color_reduction_v0(env)
env = resize_v1(env, frame_size[0], frame_size[1])
env = frame_stack_v1(env, stack_size=stack_size)

agent.eval()
# attention_weights_all = []
with torch.no_grad():
    obs, infos = env.reset(seed=None)
    obs = batchify_obs(obs, device)
    terms = [False]
    truncs = [False]
    # while not any(terms) and not any(truncs):
    ids = torch.arange(num_pistons).unsqueeze(-1).to(device)
    hidden = agent.network(obs / 255.0)
    agent_features = torch.concatenate([hidden, ids], dim=1)
    # Get attention weights
    _, attention_weights = agent.comm_aggregator(agent_features, return_attention=True)

# Visualize attention for each head
for head in range(n_heads):
    plt.figure(figsize=(8, 6))
    sns.heatmap(attention_weights[:, head, :].detach().cpu().numpy(), cmap="Blues", cbar=True)
    plt.title(f"Attention Weights - Head {head + 1}")
    plt.xlabel("Attended Agent")
    plt.ylabel("Querying Agent")
    # plt.show()
    plt.savefig(f"attn_viz/attention_head_{head + 1}.png")

# plot average attention weights
plt.figure(figsize=(8, 6))
sns.heatmap(attention_weights.mean(dim=1).detach().cpu().numpy(), cmap="Blues", cbar=True)
plt.title("Average Attention Weights")
plt.xlabel("Attended Agent")
plt.ylabel("Querying Agent")
# plt.show()
plt.savefig("attn_viz/attention_average.png")