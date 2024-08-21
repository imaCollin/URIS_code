#Uris-PAN Yuguang-main.py
# Import necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch_geometric.nn import GCNConv, GATConv
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
from torch.utils.tensorboard import SummaryWriter

# A.2.1 Data Preprocessing and Augmentation
def preprocess_data(data, adjacency_matrix, external_factors=None):
    """
    Standardize the input data and convert it to tensors.
    
    Parameters:
    - data: numpy array, time series data of shape (time_steps, num_features).
    - adjacency_matrix: numpy array, adjacency matrix of shape (num_nodes, num_nodes) for GCN.
    - external_factors: numpy array, external factors (e.g., macroeconomic data) of shape (time_steps, num_external_features).
    
    Returns:
    - data_tensor: Tensor representation of the standardized time series data.
    - adj_tensor: Tensor representation of the adjacency matrix.
    - external_tensor: Tensor representation of the standardized external factors.
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data)
    
    # If external factors are provided, standardize and convert them to tensors
    if external_factors is not None:
        external_scaled = scaler.fit_transform(external_factors)
        external_tensor = torch.tensor(external_scaled, dtype=torch.float32)
    else:
        external_tensor = None

    data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
    adj_tensor = torch.tensor(adjacency_matrix, dtype=torch.float32)
    return data_tensor, adj_tensor, external_tensor

def augment_data(data, method='noise'):
    """
    Data augmentation method to increase model robustness.
    
    Parameters:
    - data: numpy array, original time series data of shape (time_steps, num_features).
    - method: Augmentation method (default is 'noise'), can choose 'noise' or 'mixup'.
    
    Returns:
    - augmented_data: Augmented dataset.
    """
    if method == 'noise':
        noise_factor = 0.05  # Noise factor
        noise = np.random.randn(*data.shape) * noise_factor
        augmented_data = data + noise
    elif method == 'mixup':
        alpha = 0.2
        lam = np.random.beta(alpha, alpha)
        index = np.random.permutation(data.shape[0])
        augmented_data = lam * data + (1 - lam) * data[index]
    else:
        raise ValueError(f"Unknown augmentation method: {method}")

    return augmented_data

# A.2.2 Advanced GCN-LSTM Model with Attention Mechanism
class TransformerLSTM_Model(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_size, num_layers, output_size=1, external_size=None, use_gat=False, dropout=0.2):
        """
        Initialize the advanced GCN-LSTM model with an attention mechanism.
        
        Parameters:
        - num_nodes: Number of nodes in the graph (e.g., number of companies).
        - num_features: Number of features per time step (e.g., number of companies).
        - hidden_size: Number of units in the LSTM hidden layer.
        - num_layers: Number of LSTM layers.
        - output_size: Size of the output layer for multitask learning.
        - external_size: Number of external factor features, if any.
        - use_gat: Whether to use Graph Attention Network (GAT) instead of GCN.
        - dropout: Dropout rate to prevent overfitting.
        """
        super(TransformerLSTM_Model, self).__init__()
        self.num_nodes = num_nodes
        self.use_gat = use_gat
        
        # Use GCN or GAT
        if use_gat:
            self.gcn = GATConv(num_features, hidden_size)
        else:
            self.gcn = GCNConv(num_features, hidden_size)
            
        # Transformer encoder layer for extracting complex temporal features
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, dim_feedforward=hidden_size * 4, dropout=dropout), 
            num_layers=num_layers
        )
        
        # LSTM layer for further processing of temporal features
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # If there are external factors, define an additional fully connected layer to process them
        if external_size is not None:
            self.external_fc = nn.Linear(external_size, hidden_size)
        
        # Fully connected layer to map the output of the Transformer and LSTM to the final prediction
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x, adj, external=None):
        """
        Forward propagation of the model.
        
        Parameters:
        - x: Input time series data of shape (time_steps, num_features).
        - adj: Adjacency matrix for the GCN layer, shape (num_nodes, num_nodes).
        - external: Input external factors, shape (time_steps, num_external_features).
        
        Returns:
        - out: Model prediction.
        """
        # Apply GCN/GAT layer to capture spatial dependencies
        x = self.gcn(x, adj)
        x = torch.relu(x)  # Apply ReLU activation
        
        # Reshape data to (batch_size, num_nodes, hidden_size) to fit Transformer input
        x = x.view(-1, self.num_nodes, x.size(1))
        
        # Apply Transformer encoder layer
        x = self.transformer_encoder(x)
        
        # Apply LSTM layer to capture temporal dependencies
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        
        # If there are external factors, combine them with the LSTM output
        if external is not None:
            external_out = self.external_fc(external)
            lstm_out = lstm_out + external_out
        
        # Apply the fully connected layer to generate the final output
        out = self.fc(lstm_out)
        return out

# A.2.3 Dynamic Market Simulation and Adaptive Architecture
def update_adjacency_matrix(adj_matrix, dynamic_factor):
    """
    Dynamically update the adjacency matrix to simulate changes in relationships between companies.
    
    Parameters:
    - adj_matrix: Initial adjacency matrix.
    - dynamic_factor: Dynamic adjustment factor to simulate changes in relationship strength.
    
    Returns:
    - updated_adj_matrix: Updated adjacency matrix.
    """
    noise = np.random.normal(0, dynamic_factor, adj_matrix.shape)
    updated_adj_matrix = adj_matrix + noise
    updated_adj_matrix = np.clip(updated_adj_matrix, 0, 1)  # Ensure matrix values are within [0, 1]
    return updated_adj_matrix

def simulate_dynamic_market(data_tensor, adj_tensor, num_simulations=10, dynamic_factor=0.1, adaptive=False):
    """
    Simulate a dynamic market environment and evaluate model performance under changing market conditions.
    
    Parameters:
    - data_tensor: Preprocessed data tensor.
    - adj_tensor: Initial adjacency matrix tensor.
    - num_simulations: Number of simulations to run.
    - dynamic_factor: Dynamic factor used to adjust the adjacency matrix.
    - adaptive: Whether to enable adaptive architecture adjustments.
    
    Returns:
    - simulation_results: List of model outputs for each simulation.
    """
    simulation_results = []
    for i in range(num_simulations):
        # Dynamically update the adjacency matrix
        adj_tensor = update_adjacency_matrix(adj_tensor.numpy(), dynamic_factor)
        adj_tensor = torch.tensor(adj_tensor, dtype=torch.float32)
        
        # Reinitialize and train the model, applying adaptive architecture adjustments
        if adaptive:
            hidden_size = random.choice([32, 64, 128])
            num_layers = random.choice([1, 2, 3])
        else:
            hidden_size = 64
            num_layers = 2
        
        model = TransformerLSTM_Model(num_nodes=adj_tensor.shape[0], num_features=data_tensor.shape[1], hidden_size=hidden_size, num_layers=num_layers, output_size=1, use_gat=True)
        model.eval()  # Set to evaluation mode
        
        # Get the model output
        with torch.no_grad():
            output = model(data_tensor, adj_tensor)
            simulation_results.append(output.numpy())
    
    return simulation_results

# A.2.4 GANs for Data Augmentation
class GANGenerator(nn.Module):
    def __init__(self, input_dim, output_dim):
        """
        Initialize the GAN generator network.
        
        Parameters:
        - input_dim: Dimension of the input noise.
        - output_dim: Dimension of the generated data.
        """
        super(GANGenerator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Tanh()
        )

    def forward(self, x):
        """
        Forward propagation to generate data.
        
        Parameters:
        - x: Input random noise tensor.
        
        Returns:
        - Generated data tensor.
        """
        return self.fc(x)

class GANDiscriminator(nn.Module):
    def __init__(self, input_dim):
        """
        Initialize the GAN discriminator network.
        
        Parameters:
        - input_dim: Dimension of the input data.
        """
        super(GANDiscriminator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        Forward propagation for classification.
        
        Parameters:
        - x: Input data tensor.
        
        Returns:
        - Classification probability by the discriminator.
        """
        return self.fc(x)

def train_gan(generator, discriminator, data_tensor, num_epochs=1000, batch_size=64, noise_dim=100, lr=0.0002):
    """
    Train a GAN model to generate time series data.
    
    Parameters:
    - generator: GAN generator network.
    - discriminator: GAN discriminator network.
    - data_tensor: Real data tensor.
    - num_epochs: Number of training epochs.
    - batch_size: Batch size.
    - noise_dim: Dimension of the random noise.
    - lr: Learning rate.
    
    Returns:
    - generator: Trained generator model.
    """
    criterion = nn.BCELoss()
    optimizer_g = optim.Adam(generator.parameters(), lr=lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=lr)

    for epoch in range(num_epochs):
        for _ in range(len(data_tensor) // batch_size):
            # Train the discriminator
            real_data = data_tensor[torch.randint(0, len(data_tensor), (batch_size,))]
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            
            # Discriminator loss
            real_labels = torch.ones(batch_size, 1)
            fake_labels = torch.zeros(batch_size, 1)
            d_loss_real = criterion(discriminator(real_data), real_labels)
            d_loss_fake = BXgs2AAWD7gF2WUNhva7byzkpR4QbvM3YHoCJzrGebnb fake_labels)
            d_loss = d_loss_real + d_loss_fake
            
            optimizer_d.zero_grad()
            d_loss.backward()
            optimizer_d.step()

            # Train the generator
            noise = torch.randn(batch_size, noise_dim)
            fake_data = generator(noise)
            g_loss = criterion(discriminator(fake_data), real_labels)  # Generator wants the discriminator to think the generated data is real
            
            optimizer_g.zero_grad()
            g_loss.backward()
            optimizer_g.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}")
    
    return generator

# A.2.5 Reinforcement Learning-Controlled Real-Time Prediction
class RealTimePredictorRL:
    def __init__(self, model, initial_data, adj_matrix, scaler, num_nodes, num_features, rl_agent):
        """
        Initialize the reinforcement learning-controlled real-time predictor.
        
        Parameters:
        - model: Trained GCN-LSTM model.
        - initial_data: Initial historical data for the model's initial prediction.
        - adj_matrix: Adjacency matrix for the GCN layer.
        - scaler: Data scaler for preprocessing real-time data.
        - num_nodes: Number of nodes in the graph (e.g., number of companies).
        - num_features: Number of features per time step (e.g., number of companies).
        - rl_agent: Reinforcement learning agent for dynamically adjusting model parameters.
        """
        self.model = model
        self.data = initial_data
        self.adj_matrix = adj_matrix
        self.scaler = scaler
        self.num_nodes = num_nodes
        self.num_features = num_features
        self.rl_agent = rl_agent
    
    def predict(self, new_data):
        """
        Perform real-time prediction using new data and dynamically adjust model parameters with reinforcement learning.
        
        Parameters:
        - new_data: New incoming time series data.
        
        Returns:
        - prediction: Real-time model prediction.
        """
        # Update data
        self.data = np.vstack([self.data, new_data])[-100:]  # Keep the most recent 100 time steps
        
        # Data standardization
        data_scaled = self.scaler.transform(self.data)
        data_tensor = torch.tensor(data_scaled, dtype=torch.float32)
        adj_tensor = torch.tensor(self.adj_matrix, dtype=torch.float32)
        
        # Reinforcement learning agent chooses an action (e.g., adjusting learning rate, LSTM layers, etc.)
        action = self.rl_agent.select_action(self.data)
        self._adjust_model_parameters(action)
        
        # Model prediction
        self.model.eval()
        with torch.no_grad():
            prediction = self.model(data_tensor, adj_tensor)
        
        return prediction.numpy()
    
    def _adjust_model_parameters(self, action):
        """
        Adjust model parameters based on the reinforcement learning agent's decision.
        
        Parameters:
        - action: Action selected by the reinforcement learning agent.
        """
        # Example: Adjust learning rate or LSTM layers based on action
        if action == 'increase_lr':
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] *= 1.1
        elif action == 'decrease_lr':
            for param_group in self.model.optimizer.param_groups:
                param_group['lr'] *= 0.9
        elif action == 'increase_layers':
            self.model.lstm.num_layers += 1
        elif action == 'decrease_layers':
            if self.model.lstm.num_layers > 1:
                self.model.lstm.num_layers -= 1

# A.2.6 Multi-Market Multitask Learning
class MultiTaskGCN_LSTM_Model(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_size, num_layers, output_sizes, external_size=None, use_gat=False, dropout=0.2):
        """
        Initialize a multitask learning GCN-LSTM model for cross-market forecasting.
        
        Parameters:
        - num_nodes: Number of nodes in the graph (e.g., number of companies).
        - num_features: Number of features per time step (e.g., number of companies).
        - hidden_size: Number of units in the LSTM hidden layer.
        - num_layers: Number of LSTM layers.
        - output_sizes: List of output sizes for different market forecasting tasks.
        - external_size: Number of external factor features, if any.
        - use_gat: Whether to use Graph Attention Network (GAT) instead of GCN.
        - dropout: Dropout rate to prevent overfitting.
        """
        super(MultiTaskGCN_LSTM_Model, self).__init__()
        self.num_nodes = num_nodes
        self.use_gat = use_gat
        
        # Use GCN or GAT
        if use_gat:
            self.gcn = GATConv(num_features, hidden_size)
        else:
            self.gcn = GCNConv(num_features, hidden_size)
            
        # LSTM layer to extract temporal features from the time series
        self.lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # If there are external factors, define an additional fully connected layer to process them
        if external_size is not None:
            self.external_fc = nn.Linear(external_size, hidden_size)
        
        # Multitask output layer, one output layer per task
        self.fc_tasks = nn.ModuleList([nn.Linear(hidden_size, output_size) for output_size in output_sizes])

    def forward(self, x, adj, external=None):
        """
        Forward propagation of the model.
        
        Parameters:
        - x: Input time series data of shape (time_steps, num_features).
        - adj: Adjacency matrix for the GCN layer, shape (num_nodes, num_nodes).
        - external: Input external factors, shape (time_steps, num_external_features).
        
        Returns:
        - outputs: List of prediction results for each task.
        """
        # Apply GCN/GAT layer to capture spatial dependencies
        x = self.gcn(x, adj)
        x = torch.relu(x)  # Apply ReLU activation
        
        # Reshape data to (batch_size, num_nodes, hidden_size) to fit LSTM input
        x = x.view(-1, self.num_nodes, x.size(1))
        
        # Apply LSTM layer to capture temporal dependencies
        lstm_out, _ = self.lstm(x)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        
        # If there are external factors, combine them with the LSTM output
        if external is not None:
            external_out = self.external_fc(external)
            lstm_out = lstm_out + external_out
        
        # Apply multitask output layers to generate predictions
        outputs = [fc(lstm_out) for fc in self.fc_tasks]
        return outputs

# A.2.7 Enhanced Model Interpretability with SHAP
def explain_model_with_shap(model, data_tensor, adj_tensor, feature_names):
    """
    Use SHAP to analyze model interpretability and generate feature importance plots.
    
    Parameters:
    - model: Trained GCN-LSTM model.
    - data_tensor: Input data tensor.
    - adj_tensor: Adjacency matrix tensor.
    - feature_names: List of feature names to explain each feature's contribution to the prediction.
    
    Returns:
    - shap_values: SHAP values to explain each feature's contribution to the prediction.
    """
    model.eval()
    explainer = shap.DeepExplainer(model, [data_tensor, adj_tensor])
    shap_values = explainer.shap_values([data_tensor, adj_tensor])
    
    # Visualize SHAP values
    shap.summary_plot(shap_values[0], data_tensor.numpy(), feature_names=feature_names)
    return shap_values

# Example feature names
feature_names = [f'Company {i+1}' for i in range(data_tensor.shape[1])]

# Generate SHAP explanations
shap_values = explain_model_with_shap(model, data_tensor, adj_tensor, feature_names)

# A.2.8 Automated Trading System Integration and Real-Time Monitoring
class AutomatedTradingSystem:
    def __init__(self, predictor, trading_api, stop_loss=0.02, take_profit=0.05):
        """
        Initialize the automated trading system, integrating real-time prediction and trade execution.
        
        Parameters:
        - predictor: Real-time predictor object.
        - trading_api: Trading API interface for executing buy/sell operations.
        - stop_loss: Stop-loss threshold.
        - take_profit: Take-profit threshold.
        """
        self.predictor = predictor
        self.trading_api = trading_api
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.position = None  # Current position state
        self.entry_price = None  # Entry price
    
    def on_new_data(self, new_data):
        """
        Execute the appropriate trading strategy when new market data is received.
        
        Parameters:
        - new_data: New incoming market data.
        """
        prediction = self.predictor.predict(new_data)
        
        if self.position is None:
            # If no position is open, execute a buy operation based on the prediction
            if prediction > new_data[-1]:
                self.position = 'long'
                self.entry_price = new_data[-1]
                self.trading_api.buy()
        else:
            # If a position is open, check if stop-loss or take-profit conditions are met
            current_price = new_data[-1]
            if self.position == 'long':
                if current_price <= self.entry_price * (1 - self.stop_loss):
                    self.trading_api.sell()
                    self.position = None
                elif current_price >= self.entry_price * (1 + self.take_profit):
                    self.trading_api.sell()
                    self.position = None

    def monitor_market(self, data_stream):
        """
        Monitor the market data stream and execute the appropriate trading strategy.
        
        Parameters:
        - data_stream: Real-time market data stream.
        """
        for new_data in data_stream:
            self.on_new_data(new_data)