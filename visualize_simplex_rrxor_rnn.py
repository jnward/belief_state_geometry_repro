# %%
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

# %%
# Define RRXOR Transition matrices (t0, t1)
t0 = [
    [0, 0.5, 0, 0, 0],
    [0, 0, 0, 0, 0.5],
    [0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0]
]

t1 = [
    [0, 0, 0.5, 0, 0],
    [0, 0, 0, 0.5, 0],
    [0, 0, 0, 0, 0.5],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

print("Total probability mass:", np.sum(t0), np.sum(t1))

# %%
def update_belief(belief, observation, t0, t1):
    """
    Update a belief state based on a new observation using Bayes' rule.
    Returns None if the observation is impossible from the current belief state.
    """
    # Select the appropriate transition matrix
    if observation == 0:
        t_matrix = t0
    elif observation == 1:
        t_matrix = t1
    
    # Apply Bayesian update: η' = (η·T(x))/(η·T(x)·1)
    numerator = np.dot(belief, t_matrix)
    denominator = np.sum(numerator)
    
    # Return None if the observation is impossible
    if denominator < 1e-10:
        return None
    
    updated_belief = numerator / denominator
    
    return updated_belief

# %%
def get_stationary_distribution(t0, t1):
    """
    Calculate the stationary distribution (initial belief state) for the HMM.
    """
    # Combined transition matrix
    T = np.array(t0) + np.array(t1)
    
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
def generate_belief_states(initial_belief, t0, t1, max_depth=8, tolerance=1e-6):
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
            # Try all possible emissions (0 and 1 for RRXOR)
            for emission in [0, 1]:
                new_state = update_belief(state, emission, t0, t1)
                
                # Skip impossible observations
                if new_state is None:
                    continue
                
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
def plot_simplex_projection(belief_states, ax=None, s=2, title=None, c=None, colorbar=False, alpha=0.5):
    """
    Project a 4-simplex (5 states) down to 2D for visualization.
    Uses PCA to find the most informative 2D projection.
    """
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    
    # Convert list of belief states to a numpy array
    if not isinstance(belief_states, np.ndarray):
        belief_states = np.array(belief_states)
    
    # Use PCA to project down to 2D
    pca = PCA(n_components=2)
    belief_states_2d = pca.fit_transform(belief_states)
    
    # Plot the points
    if c is None:
        # Default to a simple color scheme if none provided
        scatter = ax.scatter(
            belief_states_2d[:, 0], 
            belief_states_2d[:, 1],
            s=s, 
            alpha=alpha
        )
    else:
        scatter = ax.scatter(
            belief_states_2d[:, 0], 
            belief_states_2d[:, 1], 
            c=c,
            s=s, 
            alpha=alpha
        )
    
    if title:
        ax.set_title(title)
    
    if colorbar and c is not None:
        plt.colorbar(scatter, ax=ax)
    
    # Add axes and labels
    ax.set_xlabel(f'PC1 (var: {pca.explained_variance_ratio_[0]:.2%})')
    ax.set_ylabel(f'PC2 (var: {pca.explained_variance_ratio_[1]:.2%})')
    ax.grid(alpha=0.3)
    
    return ax, scatter, pca

# %%
# Convert transition matrices to numpy arrays
t0_np = np.array(t0)
t1_np = np.array(t1) 

# Calculate initial belief state (stationary distribution)
initial_belief = get_stationary_distribution(t0_np, t1_np)
print("Initial belief state:", initial_belief)

# Generate belief states
max_depth = 10
point_size = 10000 / 2 ** max_depth

belief_states = generate_belief_states(
    initial_belief,
    t0_np,
    t1_np,
    max_depth=max_depth
)
print(f"Generated {len(belief_states)} unique belief states")

# Plot the belief state geometry with 2D projection
fig, ax = plt.subplots(figsize=(12, 10))
plot_simplex_projection(belief_states, ax, s=point_size, title="RRXOR Belief State Geometry")
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
        
        # RNN layers
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
        Return the hidden states for all layers.
        """
        # Run the model normally, but extract hidden states
        with torch.no_grad():
            # Forward pass through the model
            _, hidden = self.forward(x)  
            # hidden shape: (num_layers, batch_size, hidden_dim)
        
        # Return all layer hidden states
        return hidden

# %%
# Set up data generation functions
def get_stationary_distribution_tensor(t0, t1):
    """Calculate the stationary distribution (initial belief state) for the HMM."""
    # Combined transition matrix
    T = t0 + t1
    
    # Find eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eig(T.T)
    
    # Find the index of eigenvalue closest to 1
    idx = torch.argmin(torch.abs(eigenvalues - 1.0))
    
    # Extract the corresponding eigenvector
    stationary = torch.real(eigenvectors[:, idx])
    
    # Normalize to ensure it sums to 1
    stationary = stationary / torch.sum(stationary)
    
    return stationary

def generate_sequence(t0, t1, length=10, initial_state=None):
    """
    Generate a sequence from the RRXOR HMM.
    
    Args:
        t0, t1: Transition matrices
        length: Length of sequence
        initial_state: Initial state (if None, sample from stationary distribution)
    
    Returns:
        tokens: List of tokens (0 or 1)
        states: List of hidden states
    """
    # Get stationary distribution if initial state not provided
    if initial_state is None:
        stationary = get_stationary_distribution_tensor(t0, t1)
        initial_state = torch.multinomial(stationary, 1).item()
    
    tokens = []
    states = [initial_state]
    current_state = initial_state
    
    for _ in range(length):
        # Get transition probabilities from current state
        t_probs = torch.cat([
            t0[current_state].unsqueeze(0),
            t1[current_state].unsqueeze(0)
        ], dim=0)
        
        # Flatten and normalize to get joint probability of (token, next_state)
        flat_probs = t_probs.flatten()
        flat_probs = flat_probs / flat_probs.sum()
        
        # Sample from joint distribution
        idx = torch.multinomial(flat_probs, 1).item()
        
        # Extract token and next state
        token = idx // 5  # 0 or 1
        next_state = idx % 5
        
        tokens.append(token)
        states.append(next_state)
        current_state = next_state
    
    return tokens, states

def generate_batch(t0, t1, batch_size=64, seq_length=10):
    """Generate a batch of sequences from the RRXOR HMM."""
    batch_tokens = []
    batch_states = []
    
    # Get stationary distribution for initial state sampling
    stationary = get_stationary_distribution_tensor(t0, t1)
    
    for _ in range(batch_size):
        # Sample initial state from stationary distribution
        initial_state = torch.multinomial(stationary, 1).item()
        tokens, states = generate_sequence(t0, t1, seq_length, initial_state)
        batch_tokens.append(tokens)
        batch_states.append(states)
    
    # Convert to PyTorch tensors
    batch_tokens = torch.tensor(batch_tokens)
    batch_states = torch.tensor(batch_states)
    
    return batch_tokens, batch_states

# %%
# Training setup
context_length = 10
vocab_size = 2  # 0, 1 for RRXOR
embedding_dim = 32  # Match with number of states
hidden_dim = 64    # Match with number of states
num_layers = 4    # Multiple layers as paper suggests RRXOR is spread across layers
output_dim = vocab_size

# Convert transition matrices to PyTorch tensors
t0_tensor = torch.tensor(t0, dtype=torch.float32)
t1_tensor = torch.tensor(t1, dtype=torch.float32)

# Create RNN model
model = RNNModel(vocab_size, embedding_dim, hidden_dim, num_layers, output_dim)
print(f"RNN model created with {sum(p.numel() for p in model.parameters())} parameters")

# Check device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
batch_size = 64
learning_rate = 1e-3
num_epochs = 10000

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

# Create directory for models
os.makedirs("models", exist_ok=True)

# %%
# Training function
def train_rnn(model, t0, t1, criterion, optimizer, batch_size, num_epochs, device):
    """Train the RNN model on RRXOR HMM data."""
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
        batch_tokens, _ = generate_batch(t0, t1, batch_size=batch_size, seq_length=context_length)
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
            }, f"models/rnn_rrxor_epoch{epoch}.pt")
    
    # Save the final model
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss.item() if len(losses) > 0 else None,
    }, "models/rnn_rrxor_final.pt")
    
    # Plot loss curve
    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(0, num_epochs, log_interval), losses)
    plt.xlabel('Epoch')
    plt.ylabel('Average Loss')
    plt.title('Training Loss')
    plt.savefig("models/rnn_rrxor_training_loss.png")
    plt.show()
    
    return model, losses

# %%
# Execute training
print("Starting RNN training...")
trained_model, loss_history = train_rnn(model, t0_tensor, t1_tensor, criterion, optimizer, batch_size, num_epochs, device)
print("Training complete!")

# %%
# Hidden State Analysis across all layers
print("Extracting hidden states from all layers for analysis...")

def update_belief_tensor(belief, observation, t0, t1):
    """
    Update a belief state based on observation using Bayes' rule with tensors.
    Returns None if the observation is impossible from the current belief state.
    """
    # Select the appropriate transition matrix
    if observation == 0:
        t_matrix = t0
    elif observation == 1:
        t_matrix = t1
    
    # Apply Bayesian update: η' = (η·T(x))/(η·T(x)·1)
    numerator = torch.matmul(belief, t_matrix)
    denominator = torch.sum(numerator)
    
    # Return None if the observation is impossible
    if denominator < 1e-10:
        return None
    
    updated_belief = numerator / denominator
    
    return updated_belief

def extract_all_layer_hidden_states(model, sequences, device):
    """Extract and concatenate hidden states from all layers for given sequences."""
    model.eval()
    all_layer_hidden_states = []
    outputs = []
    
    with torch.no_grad():
        for seq in tqdm(sequences, desc="Extracting hidden states"):
            # Convert to tensor
            input_tensor = torch.tensor([seq], device=device)
            
            # Run model and get hidden states from all layers
            output, hidden = model(input_tensor)
            
            # Extract hidden state from each layer and concatenate
            # hidden shape: (num_layers, batch_size=1, hidden_dim)
            # We take the last timestep's hidden state
            all_layers_concat = hidden.view(-1).cpu()  # Flatten all layers into one vector
            all_layer_hidden_states.append(all_layers_concat)
            
            # Also save the final output for comparison
            last_output = output[0, -1, :].cpu()
            outputs.append(last_output)
    
    return all_layer_hidden_states, outputs

# Generate all possible input sequences and calculate ground truth belief states
print("Generating all possible sequences...")
import itertools

# Maximum depth for generating sequences
max_depth = 10

# Generate all possible sequences up to max_depth
original_sequences = []
for length in range(1, max_depth + 1):
    for seq in itertools.product([0, 1], repeat=length):
        original_sequences.append(list(seq))

print(f"Generated {len(original_sequences)} unique sequences")

# Calculate ground truth belief states for each sequence
belief_states = []
valid_sequences = []  # Track which sequences are valid

for seq in tqdm(original_sequences, desc="Calculating belief states"):
    belief = torch.tensor(initial_belief, dtype=torch.float32)
    valid_sequence = True
    
    for token in seq:
        updated_belief = update_belief_tensor(belief, token, t0_tensor, t1_tensor)
        if updated_belief is None:
            # This sequence has an impossible transition
            valid_sequence = False
            break
        belief = updated_belief
    
    # Only add valid sequences to our list
    if valid_sequence:
        belief_states.append(belief)
        valid_sequences.append(seq)

# Use only valid sequences for the rest of the analysis
all_sequences = valid_sequences
print(f"Found {len(valid_sequences)} valid belief sequences out of {len(original_sequences)} total sequences")

# Extract hidden states from all layers
hidden_states_all_layers, outputs = extract_all_layer_hidden_states(model, all_sequences, device)
# Convert outputs to softmax probabilities
probs = torch.softmax(torch.stack(outputs), dim=1).numpy()

# %%
# Perform linear regression to find projection from hidden states to belief states
print("Performing linear regression...")

# Convert to numpy arrays
X = torch.stack(hidden_states_all_layers).numpy()
Y = torch.stack(belief_states).numpy()

# Fit linear regression model
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

# Plot comparison between ground truth and projected belief states
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot ground truth (using PCA for visualization)
plot_simplex_projection(Y, ax=ax1, s=5, title="Ground Truth RRXOR Belief States")

# Plot projected hidden states
plot_simplex_projection(projected_hidden, ax=ax2, s=5, title="Projected RNN Hidden States (All Layers)")

plt.tight_layout()
plt.savefig("rnn_rrxor_belief_state_projection_comparison.png")
plt.show()

# Create a visualization similar to the paper's Figure 7C
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Plot ground truth
plot_simplex_projection(Y, ax=ax1, s=5, title="Ground Truth Belief State Geometry")

# For coloring by belief state, use the first component (or any other scalar value)
belief_state_color = Y[:, 0]  # Just use the first dimension for coloring

# Plot projected activations colored by one dimension of the ground truth beliefs
plot_simplex_projection(
    projected_hidden,
    ax=ax2,
    s=5,
    c=belief_state_color,  # Color by the first component of the belief state
    title="RNN Hidden State Representation",
    alpha=0.7,
    colorbar=True
)

plt.tight_layout()
plt.savefig("rnn_rrxor_belief_representation.png")
plt.show()

# %%
# Calculate distances for comparison as in Figure 7D
print("Analyzing pairwise distances...")

def compute_pairwise_distances(vectors):
    """Compute all pairwise Euclidean distances between vectors."""
    n = len(vectors)
    distances = np.zeros((n, n))
    
    for i in range(n):
        for j in range(i+1, n):
            dist = np.linalg.norm(vectors[i] - vectors[j])
            distances[i, j] = dist
            distances[j, i] = dist
    
    return distances

# Get unique belief states by rounding
unique_beliefs = {}
for i, belief in enumerate(Y):
    key = tuple(np.round(belief, 6))
    if key not in unique_beliefs:
        unique_beliefs[key] = i

unique_indices = list(unique_beliefs.values())
unique_Y = Y[unique_indices]
unique_projected = projected_hidden[unique_indices]

# For next-token predictions
unique_probs = probs[unique_indices]

# Compute pairwise distances
gt_distances = compute_pairwise_distances(unique_Y)
model_distances = compute_pairwise_distances(unique_projected)
nexttoken_distances = compute_pairwise_distances(unique_probs)

# Plot scatter of distances
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Belief state vs model distances
from scipy.stats import pearsonr
flat_gt = gt_distances.flatten()
flat_model = model_distances.flatten()
flat_nexttoken = nexttoken_distances.flatten()

# Remove zero distances (diagonal)
non_zero = flat_gt > 0
flat_gt = flat_gt[non_zero]
flat_model = flat_model[non_zero]
flat_nexttoken = flat_nexttoken[non_zero]

# Plot and compute correlation
corr_belief = pearsonr(flat_gt, flat_model)[0]
r2_belief = corr_belief**2
ax1.scatter(flat_gt, flat_model, alpha=0.5, s=2)
ax1.set_title(f"Belief State vs Model Distances (R² = {r2_belief:.2f})")
ax1.set_xlabel("Ground Truth Belief Distance")
ax1.set_ylabel("Model Representation Distance")

# Next-token vs model distances
corr_nexttoken = pearsonr(flat_nexttoken, flat_model)[0]
r2_nexttoken = corr_nexttoken**2
ax2.scatter(flat_nexttoken, flat_model, alpha=0.5, s=2)
ax2.set_title(f"Next-Token vs Model Distances (R² = {r2_nexttoken:.2f})")
ax2.set_xlabel("Next-Token Prediction Distance")
ax2.set_ylabel("Model Representation Distance")

plt.tight_layout()
plt.savefig("rnn_rrxor_distance_correlation.png")
plt.show()

# %%
# Compare regression performance across individual layers vs concatenated
print("Analyzing representation across layers...")

def extract_single_layer_hidden_states(model, sequences, layer_idx, device):
    """Extract hidden states from a specific layer for all input sequences."""
    model.eval()
    layer_hidden_states = []
    
    with torch.no_grad():
        for seq in sequences:
            # Convert to tensor
            input_tensor = torch.tensor([seq], device=device)
            
            # Run model and get all hidden states
            _, hidden = model(input_tensor)
            
            # Extract hidden state from the specified layer
            # hidden shape: (num_layers, batch_size=1, hidden_dim)
            layer_state = hidden[layer_idx, 0, :].cpu()
            layer_hidden_states.append(layer_state)
    
    return layer_hidden_states

# Track MSE and R² for each layer individually and concatenated
layer_mse = []
layer_r2 = []
layer_unexplained_variance = []

# Also track per-state metrics
layer_mse_per_state = []
layer_r2_per_state = []
layer_unexplained_variance_per_state = []

# Analyze each layer separately
for layer_idx in range(num_layers):
    print(f"Analyzing layer {layer_idx}...")
    
    # Extract hidden states for this layer
    layer_hidden = extract_single_layer_hidden_states(
        model, all_sequences, layer_idx, device
    )
    
    # Convert to numpy array
    X_layer = torch.stack(layer_hidden).numpy()
    
    # Fit linear regression for this layer
    layer_regressor = LinearRegression()
    layer_regressor.fit(X_layer, Y)
    
    # Project and calculate MSE
    layer_projected = layer_regressor.predict(X_layer)
    layer_mse_value = np.mean(np.sum((layer_projected - Y)**2, axis=1))
    r2_score = layer_regressor.score(X_layer, Y)
    unexplained_variance = 1 - r2_score
    
    # Calculate metrics for each state dimension separately
    state_mse = np.mean((layer_projected - Y)**2, axis=0)
    
    # Calculate R² for each state dimension separately
    state_r2 = []
    state_unexplained_variance = []
    for i in range(Y.shape[1]):  # For each state dimension
        # Create a regression model for this specific state
        state_regressor = LinearRegression()
        state_regressor.fit(X_layer, Y[:, i])
        state_r2_value = state_regressor.score(X_layer, Y[:, i])
        state_r2.append(state_r2_value)
        state_unexplained_variance.append(1 - state_r2_value)
    
    # Store results
    layer_mse.append(layer_mse_value)
    layer_r2.append(r2_score)
    layer_unexplained_variance.append(unexplained_variance)
    
    layer_mse_per_state.append(state_mse)
    layer_r2_per_state.append(state_r2)
    layer_unexplained_variance_per_state.append(state_unexplained_variance)
    
    print(f"Layer {layer_idx} - MSE: {layer_mse_value:.6f}, R²: {r2_score:.6f}, Unexplained Variance: {unexplained_variance:.6f}")

# Calculate metrics for all layers concatenated
all_r2 = regressor.score(X, Y)
all_unexplained_variance = 1 - all_r2

# Calculate per-state metrics for all layers concatenated
all_state_r2 = []
all_state_unexplained_variance = []
for i in range(Y.shape[1]):  # For each state dimension
    # Create a regression model for this specific state
    state_regressor = LinearRegression()
    state_regressor.fit(X, Y[:, i])
    state_r2_value = state_regressor.score(X, Y[:, i])
    all_state_r2.append(state_r2_value)
    all_state_unexplained_variance.append(1 - state_r2_value)

# Add the concatenated results
layer_mse.append(mse)
layer_r2.append(all_r2)
layer_unexplained_variance.append(all_unexplained_variance)

layer_r2_per_state.append(all_state_r2)
layer_unexplained_variance_per_state.append(all_state_unexplained_variance)
# layer_mse_per_state.append(all_layers_per_state_mse)

# Create a visualization similar to Figure 7E but with unexplained variance
plt.figure(figsize=(10, 6))
x_labels = [f"Layer {i}" for i in range(num_layers)] + ["All Layers"]
plt.bar(x_labels, layer_unexplained_variance)
plt.ylabel("Unexplained Variance (1 - R²)")
plt.title("Belief State Unexplained Variance by Layer")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("rnn_rrxor_layer_unexplained_variance_comparison.png")
plt.show()

# Create a grouped bar chart for unexplained variance per state dimension
plt.figure(figsize=(14, 8))
x = np.arange(len(x_labels))  # Label locations
width = 0.15  # Width of the bars
multiplier = 0

# Convert to numpy array for easier manipulation
layer_unexplained_variance_per_state = np.array(layer_unexplained_variance_per_state)

# Plot each state's unexplained variance as a group
for i in range(5):
    offset = width * multiplier
    plt.bar(x + offset, layer_unexplained_variance_per_state[:, i], width, label=f'State {i}')
    multiplier += 1

# Add labels and legend
plt.ylabel('Unexplained Variance (1 - R²)')
plt.xlabel('Layer')
plt.title('Per-State Belief Unexplained Variance by Layer')
plt.xticks(x + width * 2, x_labels, rotation=45)
plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5)
plt.grid(alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig("rnn_rrxor_per_state_layer_unexplained_variance_comparison.png")
plt.show()

print("Analysis complete!")
print("Saved visualizations to current directory.")

# Save the regression model and results
with open("rnn_rrxor_belief_regression_results.pkl", "wb") as f:
    pickle.dump({
        "regressor": regressor,
        "mse": mse,
        "ground_truth_beliefs": Y,
        "projected_hidden_states": projected_hidden,
        "input_sequences": all_sequences,
        "layer_mse": layer_mse
    }, f)

print("Regression model and results saved to rnn_rrxor_belief_regression_results.pkl")
# %%
