# %%
import numpy as np

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

max_depth = 10
point_size = 10000 / 2 ** max_depth

# Generate belief states
belief_states = generate_belief_states(
    initial_belief,
    tA_np,
    tB_np,
    tC_np,
    max_depth=max_depth
)
print(f"Generated {len(belief_states)} unique belief states")

# Plot the belief state geometry
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(10, 8))
plot_simplex(belief_states, ax, s=point_size)
plt.title("Mess3 Belief State Geometry")
plt.tight_layout()
plt.show()

# %%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os

# Import TransformerLens for the transformer architecture
try:
    from transformer_lens import HookedTransformer, HookedTransformerConfig
    use_transformer_lens = True
    print("Using TransformerLens for transformer implementation")
except ImportError:
    print("TransformerLens not found, using PyTorch implementation")
    use_transformer_lens = False

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Make sure we're using the same transition matrices as in the belief state plotting
# Assuming tA, tB, tC are already defined in the notebook as numpy arrays,
# we'll convert them to PyTorch tensors if they aren't already
if not isinstance(tA, torch.Tensor):
    tA = torch.tensor(tA, dtype=torch.float32)
if not isinstance(tB, torch.Tensor):
    tB = torch.tensor(tB, dtype=torch.float32)
if not isinstance(tC, torch.Tensor):
    tC = torch.tensor(tC, dtype=torch.float32)

# Check if we have a GPU available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps")
print(f"Using device: {device}")

# Data generation functions

def get_stationary_distribution(tA, tB, tC):
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
        stationary = get_stationary_distribution(tA, tB, tC)
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
    stationary = get_stationary_distribution(tA, tB, tC)
    
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

# Let's test the data generation
test_tokens, test_states = generate_batch(tA, tB, tC, batch_size=5, seq_length=10)
print("Sample generated sequences:")
for i in range(5):
    print(f"Sequence {i+1}: {test_tokens[i].tolist()}")
    print(f"States    {i+1}: {test_states[i].tolist()}")
print("\n")

# Define transformer model parameters (exactly as specified in the paper)
context_length = 10
vocab_size = 3  # A, B, C
n_layer = 4
d_model = 64
n_heads = 8
d_head = 8
d_mlp = 256
act_fn = "relu"
normalization_type = "LN"  # Layer Normalization

config = HookedTransformerConfig(
    n_layers=n_layer,
    d_model=d_model,
    n_heads=n_heads,
    d_head=d_head,
    d_mlp=d_mlp,
    act_fn=act_fn,
    normalization_type=normalization_type,
    attention_dir="causal",
    d_vocab=vocab_size,
    n_ctx=context_length
)

model = HookedTransformer(
    cfg=config
)

print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")

# %%
# Set up training parameters
batch_size = 64  # As specified in the paper
learning_rate = 0.01  # As specified in the paper
num_epochs = 1000000  # As specified in the paper

# For demonstration purposes, we'll use a much smaller number of epochs
# Uncomment the next line to train for the full 1,000,000 epochs
# num_epochs_reduced = num_epochs
num_epochs_reduced = 120000  # Reduced for practical purposes

# Set up loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Create a directory to save model checkpoints
os.makedirs("models", exist_ok=True)

# Training loop
def train_transformer(model, tA, tB, tC, criterion, optimizer, batch_size, num_epochs, device=None):
    """Train the transformer model on Mess3 HMM data."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    model.train()
    
    # For tracking progress
    losses = []
    log_interval = 1000  # Log loss every 1000 epochs
    save_interval = 10000  # Save model every 10000 epochs
    
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
        if use_transformer_lens:
            logits = model(inputs)
        else:
            logits = model(inputs)
        
        # Reshape for loss calculation
        batch_size, seq_len, vocab_size = logits.shape
        logits = logits.reshape(batch_size * seq_len, vocab_size)
        targets = targets.reshape(-1)
        
        # Calculate loss
        loss = criterion(logits, targets)
        
        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track and log progress
        if epoch % log_interval == 0:
            losses.append(loss.item())
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        
        # Save model checkpoint
        if epoch % save_interval == 0 and epoch > 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss.item(),
            }, f"models/transformer_mess3_epoch{epoch}.pt")
    
    # Save the final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if len(losses) > 0 else None,
    }, "models/transformer_mess3_final.pt")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, num_epochs, log_interval), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.savefig("models/training_loss.png")
    plt.show()
    
    return model, losses

# Execute the training
print("Starting transformer training...")
trained_model, loss_history = train_transformer(model, tA, tB, tC, criterion, optimizer, batch_size, num_epochs_reduced, device)
print("Training complete!")

# Save the model again (in case the last save interval was missed)
torch.save({
    'model_state_dict': trained_model.state_dict(),
    'loss': loss_history[-1] if loss_history else None,
}, "models/transformer_mess3_final.pt")

print("Model saved to models/transformer_mess3_final.pt")
print("You can now proceed with the belief state analysis as described in the paper.")


# %%
# Belief State Regression Analysis
# This is the analysis part after training the model

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm.notebook import tqdm
import itertools
# Ensure we're using the same device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Step 1: Load the trained model
print("Loading the trained model...")
try:
    checkpoint = torch.load("models/transformer_mess3_final.pt", map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model loaded from checkpoint at epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Final loss: {checkpoint.get('loss', 'unknown')}")
except Exception as e:
    print(f"Error loading model: {e}")
    print("If you just trained the model, it should already be in memory.")

# Set model to evaluation mode
model.eval()

# Step 2: Generate all possible input sequences and their ground truth belief states
print("Generating all possible input sequences...")

def update_belief(belief, observation, tA, tB, tC):
    """
    Update a belief state based on a new observation using Bayes' rule.
    """
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

# Get initial belief state (stationary distribution)
initial_belief = get_stationary_distribution(tA, tB, tC)
print(f"Initial belief state (stationary distribution): {initial_belief.numpy()}")

# Maximum depth for generating sequences (as used in the paper)
max_depth = 8

# Generate all possible sequences up to max_depth
all_sequences = []
for length in range(1, max_depth + 1):
    for seq in itertools.product([0, 1, 2], repeat=length):
        all_sequences.append(list(seq))

print(f"Generated {len(all_sequences)} unique sequences")

# Calculate ground truth belief states for each sequence
belief_states = []
for seq in tqdm(all_sequences, desc="Calculating belief states"):
    belief = initial_belief.clone()
    for token in seq:
        belief = update_belief(belief, token, tA, tB, tC)
    belief_states.append(belief)

# Plot the ground truth belief state geometry
def plot_simplex(belief_states, ax=None, s=2, title=None, colorbar=False, alpha=0.5):
    """
    Plot belief states in a 2-simplex (triangle), including points outside the simplex.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    
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
    scatter = ax.scatter(x, y, c=belief_states.clip(0, 1), s=s, alpha=alpha)
    
    # Set expanded plot limits to show points outside the simplex
    # Calculate the range of the points to determine appropriate limits
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_padding = (x_max - x_min) * 0.1
    y_padding = (y_max - y_min) * 0.1
    
    ax.set_xlim(min(-0.05, x_min - x_padding), max(1.05, x_max + x_padding))
    ax.set_ylim(min(-0.05, y_min - y_padding), max(0.95, y_max + y_padding))
    
    ax.axis('off')
    
    if title:
        ax.set_title(title)
    
    if colorbar:
        plt.colorbar(scatter, ax=ax)
    
    return ax, scatter

# Plot ground truth belief state geometry
fig, ax = plt.subplots(figsize=(10, 8))
plot_simplex([b.numpy() for b in belief_states], ax=ax, s=20, 
             title="Mess3 Ground Truth Belief State Geometry")
plt.tight_layout()
plt.savefig("ground_truth_belief_geometry.png")
plt.show()

# Step 3: Extract residual stream activations for each sequence
print("Extracting residual stream activations...")

# Function to extract residual stream activations
def extract_activations(model, sequences, use_transformer_lens=True):
    """
    Extract residual stream activations for given input sequences.
    
    Args:
        model: The transformer model
        sequences: List of input token sequences
        use_transformer_lens: Whether to use TransformerLens or PyTorch
        
    Returns:
        List of activations from the final layer residual stream
    """
    activations = []
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Extracting activations"):
            # Convert sequence to tensor
            input_tensor = torch.tensor([seq], device=device)
            
            # Get residual stream activation at the last position
            if use_transformer_lens:
                # TransformerLens provides an easy way to get activations
                _, cache = model.run_with_cache(input_tensor)
                # Extract final layer's residual stream (before layer norm)
                activation = cache["resid_pre", n_layer - 1][0, -1].cpu()
            else:
                # For PyTorch, we'd need to register hooks (simplified here)
                # This is a placeholder - you would need proper hooks
                outputs = model(input_tensor)
                activation = outputs[0, -1].cpu()
            
            activations.append(activation)
    
    return activations

# Extract activations (using the same sequences as for belief states)
residual_activations = extract_activations(model, all_sequences, use_transformer_lens=use_transformer_lens)

# Step 4: Perform linear regression to find projection from activations to belief states
print("Performing linear regression...")

# Convert activations to numpy array
X = torch.stack(residual_activations).numpy()

# Convert belief states to numpy array
Y = torch.stack(belief_states).numpy()

# Fit linear regression model
regressor = LinearRegression()
regressor.fit(X, Y)

# Project activations to belief simplex
projected_activations = regressor.predict(X)

# Calculate mean squared error
mse = np.mean(np.sum((projected_activations - Y)**2, axis=1))
print(f"Mean squared error of projection: {mse:.6f}")
print(f"R-squared score: {regressor.score(X, Y):.6f}")

# %%
# Step 5: Visualize the results
print("Visualizing results...")

# Plot comparison between ground truth and projected belief states
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot ground truth
plot_simplex(Y, ax=ax1, s=1, title="Ground Truth Belief States")

# Plot projected activations
plot_simplex(projected_activations, ax=ax2, s=10, title="Projected Residual Stream Activations")

plt.tight_layout()
plt.savefig("belief_state_projection_comparison.png")
plt.show()

# Optional: Visualize regression error
error = np.sum((projected_activations - Y)**2, axis=1)

plt.figure(figsize=(10, 6))
plt.hist(error, bins=50)
plt.xlabel("Squared Error")
plt.ylabel("Frequency")
plt.title("Distribution of Regression Errors")
plt.savefig("regression_error_distribution.png")
plt.show()

# Create a visualization similar to the paper's Figure 5
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot ground truth
plot_simplex(Y, ax=ax1, s=1, title="Ground Truth Belief State Geometry")

# Plot projected activations - REMOVE .clip(0, 1)
plot_simplex(
    projected_activations,
    ax=ax2,
    s=1,
    title="Residual Stream Representation",
    alpha=0.5
)

plt.tight_layout()
plt.savefig("mess3_belief_representation.png")
plt.show()

print("Analysis complete!")
print("The figures show how well the transformer's residual stream linearly represents the belief state geometry.")
print("A close match indicates that the transformer has learned to represent belief states in its residual stream.")

# To closely match the visual style in the paper, create one more specialized visualization
fig, ax = plt.subplots(figsize=(10, 8))

# Convert to barycentric coordinates
corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
x_proj = projected_activations[:, 0] * corners[0, 0] + projected_activations[:, 1] * corners[1, 0] + projected_activations[:, 2] * corners[2, 0]
y_proj = projected_activations[:, 0] * corners[0, 1] + projected_activations[:, 1] * corners[1, 1] + projected_activations[:, 2] * corners[2, 1]

# Draw the simplex outline
for i in range(3):
    ax.plot([corners[i][0], corners[(i+1)%3][0]], 
            [corners[i][1], corners[(i+1)%3][1]], 'k-', alpha=0.5)

# Plot points with coloring by true belief states
scatter = ax.scatter(x_proj, y_proj, c=Y, s=1, alpha=1.0)

# MODIFY these lines to use dynamic limits
x_min, x_max = x_proj.min(), x_proj.max()
y_min, y_max = y_proj.min(), y_proj.max()
x_padding = (x_max - x_min) * 0.1
y_padding = (y_max - y_min) * 0.1

ax.set_xlim(x_min - x_padding, x_max + x_padding)
ax.set_ylim(y_min - y_padding, y_max + y_padding)

ax.axis('off')
ax.set_title("Transformer's Linear Representation of Belief States", fontsize=14)

plt.tight_layout()
plt.savefig("paper_style_visualization.png")
plt.show()

# Optional: Save the regression model and results for later use
import pickle
with open("belief_regression_results.pkl", "wb") as f:
    pickle.dump({
        "regressor": regressor,
        "mse": mse,
        "ground_truth_beliefs": Y,
        "projected_activations": projected_activations,
        "input_sequences": all_sequences
    }, f)

print("Regression model and results saved to belief_regression_results.pkl")
# %%
