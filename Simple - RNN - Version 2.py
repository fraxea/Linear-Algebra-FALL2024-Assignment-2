import numpy as np


class SimpleRNN:
    def __init__(self, vocab_size, hidden_size=100, seq_length=25, learning_rate=0.1):
        """
        Initializes the SimpleRNN model with the given parameters.

        Arguments:
        vocab_size -- The number of unique characters in the vocabulary.
        hidden_size -- The size of the hidden state vector (default 100).
        seq_length -- The length of input sequences to process at a time (default 25).
        learning_rate -- The learning rate for gradient updates (default 0.1).
        """
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_length = seq_length
        self.learning_rate = learning_rate

        # Initialize the weight matrices with small random values
        self.Wxh = np.random.randn(hidden_size, vocab_size) * 0.01  # Input to hidden
        self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01  # Hidden to hidden (recurrent)
        self.Why = np.random.randn(vocab_size, hidden_size) * 0.01  # Hidden to output
        self.bh = np.zeros((hidden_size, 1))  # Bias for hidden layer
        self.by = np.zeros((vocab_size, 1))  # Bias for output layer

        # Memory for Adagrad updates (stores squared gradients)
        self.memory = {
            "Wxh": np.zeros_like(self.Wxh),
            "Whh": np.zeros_like(self.Whh),
            "Why": np.zeros_like(self.Why),
            "bh": np.zeros_like(self.bh),
            "by": np.zeros_like(self.by)
        }

    def rnn_forward(self, inputs, h_prev):
        """
        Performs the forward pass of the RNN for a sequence of inputs.

        Arguments:
        inputs -- List of indices representing the input sequence (each index corresponds to a character).
        h_prev -- The previous hidden state (for the first step, it's the initial hidden state, usually zeros).

        Returns:
        x -- A dictionary with the input one-hot vectors for each time step.
        h -- A dictionary with the hidden states for each time step.
        """
        x, h = {}, {}
        h[-1] = np.copy(h_prev)  # Initial hidden state

        # Loop over the sequence of inputs and compute the hidden states
        for t in range(len(inputs)):
            x[t] = np.zeros((self.vocab_size, 1))  # One-hot vector for the input at time step t
            x[t][inputs[t]] = 1  # Set the correct index in the one-hot vector to 1
            h[t] = np.tanh(np.dot(self.Wxh, x[t]) + np.dot(self.Whh, h[t - 1]) + self.bh)  # Update hidden state

        return x, h

    def compute_cross_entropy_loss(self, y, targets):
        """
        Computes the cross-entropy loss between the predicted output and the target.

        Arguments:
        y -- The predicted output from the network (before softmax).
        targets -- The true target label (index of the correct character).

        Returns:
        loss -- The computed cross-entropy loss.
        p -- The probabilities of each character in the output (softmax).
        """
        p = np.exp(y) / np.sum(np.exp(y))  # Apply softmax to get probabilities
        loss = -np.log(p[targets, 0])  # Cross-entropy loss (negative log of the target probability)
        return loss, p

    def rnn_backward(self, inputs, targets, activations, h_prev):
        """
        Performs the backward pass through time, computing gradients for all parameters.

        Arguments:
        inputs -- The input sequence as indices (each index corresponds to a character).
        targets -- The target sequence (next character in the sequence).
        activations -- The activations from the forward pass (inputs, hidden states, outputs, and probabilities).
        h_prev -- The previous hidden state (for the first time step).

        Returns:
        gradients -- A dictionary with gradients for all parameters (weights and biases).
        """
        gradients = {param: np.zeros_like(getattr(self, param)) for param in ["Wxh", "Whh", "Why", "bh", "by"]}
        dh_next = np.zeros_like(h_prev)  # Gradient of the hidden state for the next time step

        # Loop over the sequence in reverse order (backpropagation through time)
        for t in reversed(range(len(inputs))):
            dy = np.copy(activations["p"][t])  # Probability vector for output at time step t
            dy[targets[t]] -= 1  # Subtract 1 from the probability at the target index
            gradients["Why"] += np.dot(dy, activations["h"][t].T)  # Gradient for Why
            gradients["by"] += dy  # Gradient for by

            # Compute the gradient of the hidden state
            dh = np.dot(self.Why.T, dy) + dh_next  # Propagate gradient to the hidden state
            dh_raw = (1 - activations["h"][t] * activations["h"][t]) * dh  # Derivative of tanh
            gradients["bh"] += dh_raw  # Gradient for bh
            gradients["Wxh"] += np.dot(dh_raw, activations["x"][t].T)  # Gradient for Wxh
            gradients["Whh"] += np.dot(dh_raw, activations["h"][t - 1].T)  # Gradient for Whh
            dh_next = np.dot(self.Whh.T, dh_raw)  # Propagate the gradient further to the previous hidden state

        # Clip gradients to prevent exploding gradients
        for key in gradients:
            np.clip(gradients[key], -5, 5, out=gradients[key])

        return gradients

    def forward(self, inputs, targets, h_prev):
        """
        Performs the forward pass of the RNN, computing the loss and storing activations.

        Arguments:
        inputs -- The input sequence (list of indices corresponding to characters).
        targets -- The target sequence (next character in the sequence).
        h_prev -- The previous hidden state (usually zeros at the start).

        Returns:
        loss -- The total loss for the sequence.
        activations -- A dictionary containing the inputs, hidden states, outputs, and probabilities.
        h_final -- The final hidden state after processing the entire sequence.
        """
        x, h = self.rnn_forward(inputs, h_prev)  # Get the input activations and hidden states
        y = {}  # Dictionary for output activations
        p = {}  # Dictionary for softmax probabilities
        loss = 0  # Initialize the total loss

        # Loop over each time step to calculate output and loss
        for t in range(len(inputs)):
            y[t] = np.dot(self.Why, h[t]) + self.by  # Compute the output at time step t
            loss_t, p[t] = self.compute_cross_entropy_loss(y[t], targets[t])  # Compute cross-entropy loss
            loss += loss_t  # Add the loss for the current time step to the total loss

        # Return the loss and the activations for the sequence
        activations = {"x": x, "h": h, "y": y, "p": p}
        return loss, activations, h[len(inputs) - 1]

    def backward(self, inputs, targets, activations, h_prev):
        """
        Performs the backward pass and calculates the gradients for all parameters.

        Arguments:
        inputs -- The input sequence as indices.
        targets -- The target sequence (next character in the sequence).
        activations -- The activations obtained from the forward pass.
        h_prev -- The previous hidden state.

        Returns:
        gradients -- The gradients for all parameters (weights and biases).
        """
        return self.rnn_backward(inputs, targets, activations, h_prev)

    def update_parameters(self, gradients):
        """
        Updates the model parameters using Adagrad (adaptive gradient descent).

        Arguments:
        gradients -- The gradients computed during the backward pass.
        """
        for param in ["Wxh", "Whh", "Why", "bh", "by"]:
            mem = self.memory[param]  # Memory for each parameter (stores squared gradients)
            grad = gradients[param]  # Gradient for the parameter
            mem += grad * grad  # Update the memory with the squared gradient
            getattr(self, param)[:] += -self.learning_rate * grad / np.sqrt(mem + 1e-8)  # Update the parameter

    def sample(self, h, seed_idx, length):
        """
        Generates a sequence of characters starting from a seed index and hidden state.

        Arguments:
        h -- The initial hidden state.
        seed_idx -- The index of the starting character.
        length -- The number of characters to generate.

        Returns:
        generated_sequence -- A string containing the generated characters.
        """
        x = np.zeros((self.vocab_size, 1))  # One-hot vector for the seed character
        x[seed_idx] = 1  # Set the input value to the seed index
        indices = []  # List to store the generated character indices

        # Generate the sequence character by character
        for _ in range(length):
            h = np.tanh(np.dot(self.Wxh, x) + np.dot(self.Whh, h) + self.bh)  # Compute the new hidden state
            y = np.dot(self.Why, h) + self.by  # Compute the output
            p = np.exp(y) / np.sum(np.exp(y))  # Apply softmax to get the probability distribution
            idx = np.random.choice(range(self.vocab_size), p=p.flatten())  # Sample the next character
            indices.append(idx)  # Store the generated character index
            x = np.zeros((self.vocab_size, 1))  # Reset the input vector
            x[idx] = 1  # Set the new character as the input

        # Convert indices back to characters
        generated_sequence = ''.join([idx_to_char[idx] for idx in indices])
        return generated_sequence


def load_data(file_path):
    with open(file_path, 'r') as file:
        data = file.read()
    vocab = list(set(data))
    char_to_idx = {ch: i for i, ch in enumerate(vocab)}
    idx_to_char = {i: ch for i, ch in enumerate(vocab)}
    return data, vocab, char_to_idx, idx_to_char


# def train_rnn(model, data, char_to_idx, idx_to_char, iterations=10000):
#     data_size = len(data)
#     h_prev = np.zeros((model.hidden_size, 1))
#     pointer = 0
#     smooth_loss = -np.log(1.0 / model.vocab_size) * model.seq_length
#
#     for iteration in range(iterations):
#         if pointer + model.seq_length + 1 >= data_size:
#             h_prev = np.zeros((model.hidden_size, 1))
#             pointer = 0
#
#         inputs = [char_to_idx[ch] for ch in data[pointer:pointer + model.seq_length]]
#         targets = [char_to_idx[ch] for ch in data[pointer + 1:pointer + model.seq_length + 1]]
#
#         loss, activations, h_prev = model.forward(inputs, targets, h_prev)
#         gradients = model.backward(inputs, targets, activations, h_prev)
#         model.update_parameters(gradients)
#
#         smooth_loss = smooth_loss * 0.999 + loss * 0.001
#         if iteration % 100 == 0:
#             print(f"Iteration {iteration}, Smooth Loss: {smooth_loss:.4f}")
#             sample = model.sample(h_prev, inputs[0], 200)
#             print(f"Sampled text:\n{sample}\n")
#
#         pointer += model.seq_length


def train_rnn(model, data, char_to_idx, idx_to_char, iterations=10000):
    """
    Trains the SimpleRNN model using the provided data and updates the model's parameters.

    Arguments:
    model -- The SimpleRNN model to train.
    data -- The input text data as a string.
    char_to_idx -- A dictionary mapping characters to indices.
    idx_to_char -- A dictionary mapping indices to characters.
    iterations -- The number of iterations to train the model (default 10000).

    Returns:
    loss_history -- A list of loss values at each iteration for tracking progress.
    """
    h_prev = np.zeros((model.hidden_size, 1))  # Initialize the hidden state (zero at the start)
    loss_history = []  # List to store the loss history for visualization

    # Training loop
    for iteration in range(iterations):
        # Generate random indices for a mini-batch of training data
        start_idx = np.random.randint(0, len(data) - model.seq_length)  # Random start index
        end_idx = start_idx + model.seq_length  # Calculate the end index
        inputs = [char_to_idx[ch] for ch in data[start_idx:end_idx]]  # Convert characters to indices
        targets = [char_to_idx[ch] for ch in
                   data[start_idx + 1:end_idx + 1]]  # The next character in the sequence is the target

        # Perform the forward pass and calculate the loss
        loss, activations, h_prev = model.forward(inputs, targets, h_prev)

        # Perform the backward pass and compute the gradients
        gradients = model.backward(inputs, targets, activations, h_prev)

        # Update the model parameters using the computed gradients
        model.update_parameters(gradients)

        # Store and print the loss for every 100 iterations
        if iteration % 100 == 0:
            loss_history.append(loss)
            print(f"Iteration {iteration}, Loss: {loss}")

    return loss_history

# --- Main Execution ---
data, vocab, char_to_idx, idx_to_char = load_data('Harry_Potter_all_books_preprocessed.txt')
rnn_model = SimpleRNN(len(vocab))
train_rnn(rnn_model, data, char_to_idx, idx_to_char)
