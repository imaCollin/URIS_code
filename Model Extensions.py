#Model Extensions
# A.5.2.1 Hierarchical Multitask Learning
class HierarchicalMultiTaskModel(nn.Module):
    def __init__(self, num_nodes, num_features, hidden_size, num_layers, output_sizes, external_size=None, use_gat=False, dropout=0.2):
        """
        Initialize a hierarchical multitask learning model.
        
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
        super(HierarchicalMultiTaskModel, self).__init__()
        self.num_nodes = num_nodes
        self.use_gat = use_gat
        
        # Use GCN or GAT
        if use_gat:
            self.gcn = GATConv(num_features, hidden_size)
        else:
            self.gcn = GCNConv(num_features, hidden_size)
            
        # Shared LSTM layer
        self.shared_lstm = nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        
        # Task-specific layers
        self.task_specific_layers = nn.ModuleList([nn.LSTM(input_size=hidden_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout) for _ in output_sizes])
        
        # If there are external factors, define an additional fully connected layer to process them
        if external_size is not None:
            self.external_fc = nn.Linear(external_size, hidden_size)
        
        # Multitask output layers, one output layer per task
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
        
        # Shared LSTM layer
        shared_output, _ = self.shared_lstm(x)
        shared_output = shared_output[:, -1, :]  # Take the last time step output of the shared layer
        
        # Task-specific layers
        outputs = []
        for i, task_lstm in enumerate(self.task_specific_layers):
            task_output, _ = task_lstm(shared_output.unsqueeze(1))
            task_output = task_output[:, -1, :]  # Take the last time step output of the task-specific layer
            
            # If there are external factors, combine them with the task output
            if external is not None:
                external_out = self.external_fc(external)
                task_output = task_output + external_out
            
            # Apply task-specific output layer to generate predictions
            outputs.append(self.fc_tasks[i](task_output))
        
        return outputs

# A.5.2.2 Domain Adaptation
class DomainAdaptationModel(nn.Module):
    def __init__(self, base_model, domain_classifier, lambda_grad=0.1):
        """
        Initialize a domain adaptation model.
        
        Parameters:
        - base_model: Base model for feature extraction.
        - domain_classifier: Domain classifier to identify which domain the data is from.
        - lambda_grad: Coefficient for the gradient reversal layer to control the strength of domain adaptation.
        """
        super(DomainAdaptationModel, self).__init__()
        self.base_model = base_model
        self.domain_classifier = domain_classifier
        self.lambda_grad = lambda_grad

    def forward(self, x, adj):
        # Extract features
        features = self.base_model(x, adj)
        
        # Reverse gradient for domain classifier
        reverse_features = GradientReversalFunction.apply(features, self.lambda_grad)
        domain_output = self.domain_classifier(reverse_features)
        
        return features, domain_output

class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambda_grad):
        ctx.lambda_grad = lambda_grad
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.lambda_grad
        return output, None

class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        """
        Initialize the domain classifier.
        
        Parameters:
        - input_dim: Dimension of the input features.
        - hidden_dim: Number of units in the hidden layer.
        """
        super(DomainClassifier, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)  # Assuming there are two domains
        )

    def forward(self, x):
        return self.fc(x)