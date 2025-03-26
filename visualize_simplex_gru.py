# visualize_simplex_gru.py
# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import pickle
import types

# %%
# Transition matrices - prob of emitting A, B, C given transitioning from i to j
tA = [[0.765, 0.00375, 0.00375],
      [0.0425, 0.0675, 0.00375],
      [0.0425, 0.00375, 0.0675]]

tB = [[0.0675, 0.0425, 0.00375],
      [0.00375, 0.765, 0.00375],
      [0.00375, 0.0425, 0.0675]]

tC = [[0.0675, 0.00375, 0.0425],
      [0.00375, 0.0675, 0.0425],
      [0.00375, 0.00375, 0.765]]

print(np.sum(tA), np.sum(tB), np.sum(tC))

# %%
def update_belief(belief, observation, tA, tB, tC):
    """
    Update a belief state based on a new observation using Bayes' rule.
    """
    # Select the appropriate transition matrix
    if observation == 'A':
        t_matrix = tA
    elif observation == 'B':
        t_matrix = tB
    elif observation == 'C':
        t_matrix = tC
    
    # Apply Bayesian update: η' = (η·T(x))/(η·T(x)·1)
    numerator = np.dot(belief, t_matrix)
    denominator = np.sum(numerator)
    
    # Ensure we don't divide by zero
    if denominator < 1e-10:
        raise ValueError("Zero probability observation")
    
    updated_belief = numerator / denominator
    
    return updated_belief

# %%
def get_stationary_distribution(tA, tB, tC):
    """
    Calculate the stationary distribution (initial belief state) for the HMM.
    """
    # Combined transition matrix
    T = np.array(tA) + np.array(tB) + np.array(tC)
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(T.T)
    
    # Find the index of eigenvalue closest to 1
    idx = np.argmin(np.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector
    stationary = np.real(eigenvectors[:, idx])
    
    # Normalize to ensure it sums to 1
    stationary = stationary / np.sum(stationary)
    
    return stationary

# %%
def generate_belief_states(initial_belief, tA, tB, tC, max_depth=8, tolerance=1e-6):
    """
    Generate all unique belief states up to max_depth using breadth-first search
    with pruning of already-seen states.
    """
    all_states = [initial_belief]  # Start with initial belief
    current_level = {tuple(np.round(initial_belief, 8)): initial_belief}
    seen_states = set(current_level.keys())  # Track states we've seen
    
    # Process level by level (BFS)
    for depth in range(max_depth):
        next_level = {}
        
        # For each state in the current level
        for state in current_level.values():
            # Try all possible emissions
            for emission in ['A', 'B', 'C']:
                new_state = update_belief(state, emission, tA, tB, tC)
                rounded = tuple(np.round(new_state, 8))
                
                # Only add if we haven't seen this state before
                if rounded not in seen_states:
                    next_level[rounded] = new_state
                    seen_states.add(rounded)
                    all_states.append(new_state)
        
        # Move to next level
        current_level = next_level
        
        # If no new states were found, we can stop early
        if not current_level:
            break
    
    return all_states

# %%
def plot_simplex(belief_states, ax=None, s=2):
    """
    Plot belief states in a 2-simplex (triangle) - vectorized for speed.
    """
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
    # Convert list of belief states to a numpy array if it's not already
    if not isinstance(belief_states, np.ndarray):
        belief_states = np.array(belief_states)
    
    # Define the corners of the simplex
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Draw the outline of the simplex
    for i in range(3):
        ax.plot([corners[i][0], corners[(i+1)%3][0]], 
                [corners[i][1], corners[(i+1)%3][1]], 'k-', alpha=0.5)
    
    # Vectorized conversion from barycentric to Cartesian coordinates
    x = belief_states[:, 0] * corners[0, 0] + belief_states[:, 1] * corners[1, 0] + belief_states[:, 2] * corners[2, 0]
    y = belief_states[:, 0] * corners[0, 1] + belief_states[:, 1] * corners[1, 1] + belief_states[:, 2] * corners[2, 1]
    
    # Plot all points at once
    ax.scatter(x, y, c=belief_states, s=s, alpha=0.8)
    
    # Set the plot limits and remove axes
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.95)
    ax.axis('off')
    
    return ax

# %%
# Convert transition matrices to numpy arrays
tA_np = np.array(tA)
tB_np = np.array(tB) 
tC_np = np.array(tC)

# Calculate initial belief state (stationary distribution)
initial_belief = get_stationary_distribution(tA_np, tB_np, tC_np)
print("Initial belief state:", initial_belief)

# Generate belief states
max_depth = 10
point_size = 10000 / 2 ** max_depth

belief_states = generate_belief_states(
    initial_belief,
    tA_np,
    tB_np,
    tC_np,
    max_depth=max_depth
)
print(f"Generated {len(belief_states)} unique belief states")

# Plot the belief state geometry
fig, ax = plt.subplots(figsize=(10, 8))
plot_simplex(belief_states, ax, s=point_size)
plt.title("Mess3 Belief State Geometry")
plt.tight_layout()
plt.show()

# %%
# GRU model setup
class RNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers, output_dim):
        super(RNNModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # RNN layers (vanilla RNN instead of GRU)
        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            nonlinearity='tanh'
        )
        
        # Output layer
        self.output = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x, hidden=None):
        # x shape: (batch_size, seq_length)
        batch_size = x.size(0)
        
        # Initialize hidden state if not provided
        if hidden is None:
            hidden = self.init_hidden(batch_size)
        
        # Embedding layer
        embedded = self.embedding(x)  # (batch_size, seq_length, embedding_dim)
        
        # RNN layers
        output, hidden = self.rnn(embedded, hidden)
        # output shape: (batch_size, seq_length, hidden_dim)
        # hidden shape: (num_layers, batch_size, hidden_dim)
        
        # Reshape output for prediction
        output = self.output(output)  # (batch_size, seq_length, output_dim)
        
        return output, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state with zeros
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(next(self.parameters()).device)
    
    def get_hidden_states(self, x):
        """
        Return the hidden states for all layers at the final position.
        """
        # Run the model normally, but extract hidden states
        with torch.no_grad():
            # Forward pass through the model
            _, hidden = self.forward(x)  
            # hidden is already (num_layers, batch_size, hidden_dim)
        
        # Return all layer hidden states
        return hidden

# %%
# Set up data generation functions
def get_stationary_distribution_tensor(tA, tB, tC):
    """Calculate the stationary distribution (initial belief state) for the HMM."""
    # Combined transition matrix
    T = tA + tB + tC
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(T.T)
    
    # Find the index of eigenvalue closest to 1
    idx = torch.argmin(torch.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector
    stationary = torch.real(eigenvectors[:, idx])
    
    # Normalize to ensure it sums to 1
    stationary = stationary / torch.sum(stationary)
    
    return stationary

def generate_sequence(tA, tB, tC, length=10, initial_state=None):
    """
    Generate a sequence from the Mess3 HMM.
    
    Args:
        tA, tB, tC: Transition matrices
        length: Length of sequence
        initial_state: Initial state (if None, sample from stationary distribution)
    
    Returns:
        tokens: List of tokens (0 for A, 1 for B, 2 for C)
        states: List of hidden states
    """
    # Get stationary distribution if initial state not provided
    if initial_state is None:
        stationary = get_stationary_distribution_tensor(tA, tB, tC)
        initial_state = torch.multinomial(stationary, 1).item()
    
    tokens = []
    states = [initial_state]
    current_state = initial_state
    
    for _ in range(length):
        # Get transition probabilities from current state
        t_probs = torch.cat([
            tA[current_state].unsqueeze(0),
            tB[current_state].unsqueeze(0),
            tC[current_state].unsqueeze(0)
        ], dim=0)
        
        # Flatten and normalize to get joint probability of (token, next_state)
        flat_probs = t_probs.flatten()
        flat_probs = flat_probs / flat_probs.sum()
        
        # Sample from joint distribution
        idx = torch.multinomial(flat_probs, 1).item()
        
        # Extract token and next state
        token = idx // 3  # 0 for A, 1 for B, 2 for C
        next_state = idx % 3
        
        tokens.append(token)
        states.append(next_state)
        current_state = next_state
    
    return tokens, states

def generate_batch(tA, tB, tC, batch_size=64, seq_length=10):
    """Generate a batch of sequences from the Mess3 HMM."""
    batch_tokens = []
    batch_states = []
    
    # Get stationary distribution for initial state sampling
    stationary = get_stationary_distribution_tensor(tA, tB, tC)
    
    for _ in range(batch_size):
        # Sample initial state from stationary distribution
        initial_state = torch.multinomial(stationary, 1).item()
        tokens, states = generate_sequence(tA, tB, tC, seq_length, initial_state)
        batch_tokens.append(tokens)
        batch_states.append(states)
    
    # Convert to PyTorch tensors
    batch_tokens = torch.tensor(batch_tokens)
    batch_states = torch.tensor(batch_states)
    
    return batch_tokens, batch_states

# %%
# Training setup
context_length = 10
vocab_size = 3  # A, B, C
embedding_dim = 3
hidden_dim = 3
num_layers = 1
output_dim = vocab_size

# Convert transition matrices to PyTorch tensors
tA_tensor = torch.tensor(tA, dtype=torch.float32)
tB_tensor = torch.tensor(tB, dtype=torch.float32)
tC_tensor = torch.tensor(tC, dtype=torch.float32)

# Create RNN model (instead of GRU)
model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)
print(f"RNN model created with {sum(p.numel() for p in model.parameters())} parameters")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 20000

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Create directory for models
os.makedirs("models", exist_ok=True)

# %%
# Training function
def train_rnn(model, tA, tB, tC, criterion, optimizer, batch_size, num_epochs, device):
    """Train the RNN model on Mess3 HMM data."""
    model.train()
    
    # For tracking progress
    losses = []
    log_interval = 1000  # Log loss every 1000 epochs
    save_interval = 10000  # Save model every 10000 epochs
    
    # For tracking average loss between logging intervals
    current_interval_losses = []
    
    # Training loop
    for epoch in tqdm(range(num_epochs)):
        # Generate a batch of sequences
        batch_tokens, _ = generate_batch(tA, tB, tC, batch_size=batch_size, seq_length=context_length)
        batch_tokens = batch_tokens.to(device)
        
        # Input is all tokens except the last
        inputs = batch_tokens[:, :-1]
        
        # Target is all tokens except the first
        targets = batch_tokens[:, 1:]
        
        # Forward pass
        outputs, _ = model(inputs)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = outputs.shape
        outputs = outputs.reshape(batch_size * seq_len, vocab_size)
        targets = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(outputs, targets)
        
        # Track current loss for averaging
        current_interval_losses.append(loss.item())
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track and log progress
        if epoch % log_interval == 0:
            # Calculate average loss for the interval
            avg_loss = sum(current_interval_losses) / len(current_interval_losses)
            losses.append(avg_loss)
            print(f"Epoch {epoch}, Avg Loss: {avg_loss:.4f}")
            
            # Reset for next interval
            current_interval_losses = []
        
        # Save model checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f"models/rnn_mess3_epoch{epoch}.pt")
    
    # Save the final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if len(losses) > 0 else None,
    }, "models/rnn_mess3_final.pt")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, num_epochs, log_interval), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.savefig("models/rnn_training_loss.png")
    plt.show()
    
    return model, losses

# %%
# Execute training
print("Starting RNN training...")
trained_model, loss_history = train_rnn(model, tA_tensor, tB_tensor, tC_tensor, criterion, optimizer, batch_size, num_epochs, device)
print("Training complete!")

# %%
# Hidden State Analysis (equivalent to the Residual Stream Analysis in transformer)
print("Extracting hidden states for analysis...")

def update_belief_tensor(belief, observation, tA, tB, tC):
    """Update a belief state based on observation using Bayes' rule with tensors."""
    # Select the appropriate transition matrix
    if observation == 0:  # A
        t_matrix = tA
    elif observation == 1:  # B
        t_matrix = tB
    elif observation == 2:  # C
        t_matrix = tC
    
    # Apply Bayesian update: η' = (η·T(x))/(η·T(x)·1)
    numerator = torch.matmul(belief, t_matrix)
    denominator = torch.sum(numerator)
    
    # Ensure we don't divide by zero
    if denominator < 1e-10:
        raise ValueError("Zero probability observation")
    
    updated_belief = numerator / denominator
    
    return updated_belief

def extract_hidden_states(model, sequences, device):
    """Extract hidden states from the RNN for given sequences."""
    model.eval()
    hidden_states = []
    outputs = []
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Extracting hidden states"):
            # Convert to tensor
            input_tensor = torch.tensor([seq], device=device)
            
            # Run model and get the hidden state
            output, final_hidden = model(input_tensor)
            
            # Extract hidden state from the last layer
            last_layer_hidden = final_hidden[-1, 0, :].cpu()
            last_output = output[-1, -1, :].cpu()
            hidden_states.append(last_layer_hidden)
            outputs.append(last_output)
            # penultimate_layer_hidden = final_hidden[-2, 0, :].cpu()
            # hidden_states.append(penultimate_layer_hidden)
    
    return hidden_states, outputs

# Generate all possible input sequences and calculate ground truth belief states
print("Generating all possible sequences...")
import itertools

# Maximum depth for generating sequences (as used in the paper)
max_depth = 10

# Generate all possible sequences up to max_depth
all_sequences = []
for length in range(1, max_depth + 1):
    for seq in itertools.product([0, 1, 2], repeat=length):
        all_sequences.append(list(seq))

print(f"Generated {len(all_sequences)} unique sequences")

# Calculate ground truth belief states for each sequence
belief_states = []
for seq in tqdm(all_sequences, desc="Calculating belief states"):
    belief = torch.tensor(initial_belief, dtype=torch.float32)
    for token in seq:
        belief = update_belief_tensor(belief, token, tA_tensor, tB_tensor, tC_tensor)
    belief_states.append(belief)
    
# Extract hidden states
hidden_states, outputs = extract_hidden_states(model, all_sequences, device)
probs = torch.softmax(torch.stack(outputs), dim=1).numpy()

# %%
# Perform linear regression to find projection from hidden states to belief states
print("Performing linear regression...")

# Convert to numpy arrays
# X = torch.stack(hidden_states).numpy()
# X = torch.stack(outputs).numpy()
X = probs
Y = torch.stack(belief_states).numpy()

# Fit linear regression model
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X, Y)

# Project hidden states to belief simplex
projected_hidden = regressor.predict(X)

# Calculate mean squared error
mse = np.mean(np.sum((projected_hidden - Y)**2, axis=1))
print(f"Mean squared error of projection: {mse:.6f}")
print(f"R-squared score: {regressor.score(X, Y):.6f}")

# %%
# Visualize results
print("Visualizing results...")

def plot_simplex(belief_states, ax=None, s=2, title=None, c=None, colorbar=False, alpha=0.5):
    """
    Plot belief states in a 2-simplex (triangle).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    if c is None:
        c = belief_states
    
    # Convert list of belief states to a numpy array if it's not already
    if isinstance(belief_states[0], torch.Tensor):
        belief_states = np.array([state.numpy() for state in belief_states])
    else:
        belief_states = np.array(belief_states)
    
    # Define the corners of the simplex
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Draw the outline of the simplex
    for i in range(3):
        ax.plot([corners[i][0], corners[(i+1)%3][0]], 
                [corners[i][1], corners[(i+1)%3][1]], 'k-', alpha=0.5)
    
    # Vectorized conversion from barycentric to Cartesian coordinates
    x = belief_states[:, 0] * corners[0, 0] + belief_states[:, 1] * corners[1, 0] + belief_states[:, 2] * corners[2, 0]
    y = belief_states[:, 0] * corners[0, 1] + belief_states[:, 1] * corners[1, 1] + belief_states[:, 2] * corners[2, 1]
    
    # Plot all points at once
    scatter = ax.scatter(x, y, c=c.clip(0, 1), s=s, alpha=alpha)
    
    # Set the plot limits
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 0.95)
    
    ax.axis('off')
    
    if title:
        ax.set_title(title)
    
    if colorbar:
        plt.colorbar(scatter, ax=ax)
    
    return ax, scatter

# Plot comparison between ground truth and projected belief states
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot ground truth
plot_simplex(Y, ax=ax1, s=5, title="Ground Truth Belief States")

# Plot projected hidden states
plot_simplex(projected_hidden, ax=ax2, s=5, title="Projected RNN Hidden States")

plt.tight_layout()
plt.savefig("rnn_belief_state_projection_comparison.png")
plt.show()

# Create a visualization similar to the paper's Figure 5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot ground truth
plot_simplex(Y, ax=ax1, s=5, title="Ground Truth Belief State Geometry")

# Plot projected activations
plot_simplex(
    projected_hidden,
    ax=ax2,
    s=5,
    c=Y,
    title="RNN Hidden State Representation",
    alpha=0.7
)

plt.tight_layout()
plt.savefig("rnn_mess3_belief_representation.png")
plt.show()

# Compare with transformer results
print("Analysis complete!")
print("The figures show how well the RNN's hidden state linearly represents the belief state geometry.")
print("These results can be compared with the transformer results to see if the belief state representation is architecture-specific.")

# Save the regression model and results
with open("rnn_belief_regression_results.pkl", "wb") as f:
    pickle.dump({
        "regressor": regressor,
        "mse": mse,
        "ground_truth_beliefs": Y,
        "projected_hidden_states": projected_hidden,
        "input_sequences": all_sequences
    }, f)

print("Regression model and results saved to rnn_belief_regression_results.pkl")
# %%
model.output.weight
# %%
import plotly.express as px
data = torch.stack(hidden_states).cpu()

fig = px.scatter_3d(data, x=0, y=1, z=2)
fig.layout.scene.camera.projection.type = "orthographic"
# set size of points
fig.update_traces(marker=dict(size=1))
fig.show()
# %%
data
# %%
U, S, V =data.svd()
# project onto first two components
data_2d = data @ V[:2, :].T
# %%
fig = px.scatter(data_2d, x=0, y=1)
fig.show()
# %%
from scipy.linalg import null_space

orthogonal_basis = null_space(regressor.coef_)

# project data onto orthogonal basis
data_orthogonal = data @ orthogonal_basis

data_orthogonal.shape
# %%




projected_data = (hidden_states @ regressor.coef_.T + regressor.intercept_)
fig = px.scatter_3d(projected_data, x=0, y=1, z=2, color=data_orthogonal[:, 0])
fig.layout.scene.camera.projection.type = "orthographic"
# set size of points
fig.update_traces(marker=dict(size=1))
fig.show()
# %%
mse = np.mean((projected_data - Y)**2, axis=1)
mse.shape

# %%
fig = px.scatter(x=mse, y=data_orthogonal[:, 0])
fig.show()
# %%
fig = px.scatter_3d((data @ model.output.weight).detach().cpu().numpy(), x=0, y=1, z=2)
fig.layout.scene.camera.projection.type = "orthographic"
fig.update_traces(marker=dict(size=1))
fig.show()
# %%
# color = [(Y[i, 0], Y[i, 1], Y[i, 2]) for i in range(len(Y))]
color = Y[:, 0]

fig = px.scatter_3d(data.detach().cpu().numpy(), x=0, y=1, z=2, color=color)
fig.layout.scene.camera.projection.type = "orthographic"
fig.update_traces(marker=dict(size=1))
fig.show()
# %%
fig = px.scatter_3d((data @ model.output.weight.T).detach().cpu().numpy(), x=0, y=1, z=2, color=color)
fig.layout.scene.camera.projection.type = "orthographic"
fig.update_traces(marker=dict(size=1))
fig.show()
# %%
fig = px.scatter_3d(torch.softmax(data @ model.output.weight.T, dim=1).detach().cpu().numpy(), x=0, y=1, z=2, color=color)
fig.layout.scene.camera.projection.type = "orthographic"
fig.update_traces(marker=dict(size=1))
fig.show()
# %%
# calculate next token distributions for each sequence
for sequence in all_sequences:
    probs = model(torch.tensor([sequence])).softmax(dim=1).detach().cpu().numpy()
    print(probs)
    break
# %%
