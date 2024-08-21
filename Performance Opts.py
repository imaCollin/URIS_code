# Performance Opt
from torch.cuda.amp import autocast, GradScaler
import optuna

# A.5.1.1 Mixed Precision Training
def train_with_mixed_precision(model, train_loader, val_loader, adj_tensor, num_epochs=100, initial_lr=0.001):
    """
    Train the model using mixed precision to improve training efficiency and reduce memory usage.
    
    Parameters:
    - model: GCN-LSTM model.
    - train_loader: Training data loader.
    - val_loader: Validation data loader.
    - adj_tensor: Tensor representation of the adjacency matrix.
    - num_epochs: Number of training epochs.
    - initial_lr: Initial learning rate.
    
    Returns:
    - model: Trained model.
    """
    optimizer = optim.Adam(model.parameters(), lr=initial_lr)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    criterion = nn.MSELoss()
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            with autocast():  # Use automatic mixed precision
                output = model(batch[0], adj_tensor)
                loss = criterion(output, batch[0][:, -1, :])

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
        
        # Validation loss
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch[0], adj_tensor)
                loss = criterion(output, batch[0][:, -1, :])
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {total_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")
        
        # Adjust learning rate
        scheduler.step(val_loss)
    
    return model

# A.5.1.2 Knowledge Distillation
class DistilledModel(nn.Module):
    def __init__(self, student_model, teacher_model, alpha=0.5, temperature=2.0):
        """
        Initialize a knowledge distillation model.
        
        Parameters:
        - student_model: The smaller, faster student model.
        - teacher_model: The larger, more accurate teacher model.
        - alpha: Weight factor in the distillation loss.
        - temperature: Temperature for smoothing the teacher model's output.
        """
        super(DistilledModel, self).__init__()
        self.student = student_model
        self.teacher = teacher_model
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, x, adj):
        student_output = self.student(x, adj)
        with torch.no_grad():
            teacher_output = self.teacher(x, adj)

        # Distillation loss
        distillation_loss = nn.KLDivLoss()(nn.functional.log_softmax(student_output / self.temperature, dim=1),
                                           nn.functional.softmax(teacher_output / self.temperature, dim=1)) * (self.alpha * self.temperature * self.temperature)
        loss = distillation_loss + nn.MSELoss()(student_output, teacher_output) * (1. - self.alpha)
        return student_output, loss

# A.5.1.3 Hyperparameter Optimization with Optuna
def optimize_hyperparameters(train_loader, val_loader, adj_tensor):
    def objective(trial):
        # Suggest hyperparameters
        hidden_size = trial.suggest_int('hidden_size', 32, 256)
        num_layers = trial.suggest_int('num_layers', 1, 3)
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        dropout_rate = trial.suggest_uniform('dropout_rate', 0.0, 0.5)

        model = TransformerLSTM_Model(num_nodes=adj_tensor.shape[0], 
                                      num_features=train_loader.dataset.tensors[0].shape[2],
                                      hidden_size=hidden_size, 
                                      num_layers=num_layers, 
                                      dropout=dropout_rate)

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        # Training loop
        for epoch in range(10):  # Shorter training for hyperparameter search
            model.train()
            for batch in train_loader:
                optimizer.zero_grad()
                output = model(batch[0], adj_tensor)
                loss = criterion(output, batch[0][:, -1, :])
                loss.backward()
                optimizer.step()

        # Validation loop
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                output = model(batch[0], adj_tensor)
                loss = criterion(output, batch[0][:, -1, :])
                val_loss += loss.item()

        return val_loss / len(val_loader)
    
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    
    return study.best_params

# A.5.1.4 Ensemble Learning
class EnsembleModel(nn.Module):
    def __init__(self, models):
        """
        Initialize an ensemble model that combines predictions from multiple models.
        
        Parameters:
        - models: List of models to be combined.
        """
        super(EnsembleModel, self).__init__()
        self.models = nn.ModuleList(models)

    def forward(self, x, adj):
        outputs = [model(x, adj) for model in self.models]
        mean_output = torch.mean(torch.stack(outputs), dim=0)  # Average the predictions
        return mean_output