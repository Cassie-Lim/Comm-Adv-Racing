import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np

# List of event log paths and corresponding experiment names
event_logs = [
    ('runs/Nov30_21-58-11_cassie-cs', 'Comm-ActionState'),
    ('runs/Nov29_19-39-28_cassie-baseline', 'Baseline'),
    ('runs/Dec01_08-50-45_cassie', 'Comm-Action'),
]

# Load the event logs
event_accumulators = [(EventAccumulator(path), name) for path, name in event_logs]
for event_acc, _ in event_accumulators:
    event_acc.Reload()

# Extract scalar data
def extract_scalar_data(event_acc, tag):
    return event_acc.Scalars(tag)

# Helper function to extract steps and values
def extract_steps_and_values(scalar_data, window_size=500):
    steps = [entry.step for entry in scalar_data[:window_size]]
    values = [entry.value for entry in scalar_data[:window_size]]
    return steps, values

# Helper function to calculate mean and standard deviation over a moving window
def moving_avg_std(values, window_size=10):
    means = []
    stds = []
    for i in range(len(values) - window_size + 1):
        window = values[i:i + window_size]
        means.append(np.mean(window))
        stds.append(np.std(window))
    return np.array(means), np.array(stds)

# Extract and process data for each event log
data = {}
for event_acc, name in event_accumulators:
    return_data = extract_scalar_data(event_acc, 'Episode/Return')
    value_loss_data = extract_scalar_data(event_acc, 'Episode/Value_Loss')
    policy_loss_data = extract_scalar_data(event_acc, 'Episode/Policy_Loss')
    
    steps_return, values_return = extract_steps_and_values(return_data)
    steps_value_loss, values_value_loss = extract_steps_and_values(value_loss_data)
    steps_policy_loss, values_policy_loss = extract_steps_and_values(policy_loss_data)
    
    mean_return, std_return = moving_avg_std(values_return)
    mean_value_loss, std_value_loss = moving_avg_std(values_value_loss)
    mean_policy_loss, std_policy_loss = moving_avg_std(values_policy_loss)
    
    steps_return_smooth = steps_return[len(steps_return) - len(mean_return):]
    steps_value_loss_smooth = steps_value_loss[len(steps_value_loss) - len(mean_value_loss):]
    steps_policy_loss_smooth = steps_policy_loss[len(steps_policy_loss) - len(mean_policy_loss):]
    
    data[name] = {
        'steps_return': steps_return_smooth,
        'mean_return': mean_return,
        'std_return': std_return,
        'steps_value_loss': steps_value_loss_smooth,
        'mean_value_loss': mean_value_loss,
        'std_value_loss': std_value_loss,
        'steps_policy_loss': steps_policy_loss_smooth,
        'mean_policy_loss': mean_policy_loss,
        'std_policy_loss': std_policy_loss,
    }

# Function to plot data
def plot_data(subplot_index, data, key, ylabel, title):
    plt.subplot(3, 1, subplot_index)
    for name, values in data.items():
        steps = values[f'steps_{key}']
        mean = values[f'mean_{key}']
        std = values[f'std_{key}']
        plt.plot(steps, mean, label=f'{name} {key.capitalize()} (smoothed)')
        plt.fill_between(steps, mean - std, mean + std, alpha=0.3)
    plt.xlabel('Steps')
    plt.ylabel(ylabel)
    plt.legend()
    plt.title(title)

# Plot the data
plt.figure(figsize=(12, 8))

plot_data(1, data, 'return', 'Return', 'Episode Return')
plot_data(2, data, 'value_loss', 'Value Loss', 'Value Loss')
plot_data(3, data, 'policy_loss', 'Policy Loss', 'Policy Loss')


plt.tight_layout()
plt.show()