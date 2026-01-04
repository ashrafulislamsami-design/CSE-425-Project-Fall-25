import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearVAE(nn.Module):
    def __init__(self, input_dim=26, hidden_dim=64, latent_dim=4):
        """
        Args:
            input_dim (int): Number of input features (26 for our MFCC stats).
            hidden_dim (int): Size of the hidden layer (neurons).
            latent_dim (int): Size of the bottleneck (compressed representation).
                              We use 4 to keep enough info, but small enough to cluster.
        """
        super(LinearVAE, self).__init__()

        # ==================
        # 1. ENCODER
        # ==================
        # Compresses Input -> Hidden
        self.encoder_h = nn.Linear(input_dim, hidden_dim)
        
        # Hidden -> Mean (mu) and Log-Variance (logvar)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

        # ==================
        # 2. DECODER
        # ==================
        # Expands Latent (z) -> Hidden
        self.decoder_h = nn.Linear(latent_dim, hidden_dim)
        
        # Hidden -> Reconstructed Input
        self.decoder_output = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        """
        The "Reparameterization Trick".
        Instead of sampling strictly from N(mu, sigma), we sample noise (epsilon)
        and shift it by mu and scale by sigma. This allows backpropagation.
        """
        if self.training:
            std = torch.exp(0.5 * logvar) # Convert log-variance to standard deviation
            eps = torch.randn_like(std)   # Random noise
            return mu + eps * std
        else:
            # During testing/inference, we just want the most likely vector (the mean)
            return mu

    def forward(self, x):
        # 1. Encode
        h = F.relu(self.encoder_h(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)

        # 2. Re-parameterize (Sample z)
        z = self.reparameterize(mu, logvar)

        # 3. Decode
        h_decode = F.relu(self.decoder_h(z))
        reconstruction = self.decoder_output(h_decode)

        return reconstruction, mu, logvar

# Test Block
if __name__ == "__main__":
    # Simulate a batch of 5 songs, each with 26 features
    dummy_input = torch.randn(5, 26)
    
    # Initialize model
    model = LinearVAE(input_dim=26, hidden_dim=64, latent_dim=4)
    
    # Run model
    recon, mu, logvar = model(dummy_input)
    
    print("\nTest passed if shapes match:")
    print(f"Input Shape:      {dummy_input.shape}")
    print(f"Reconstruction:   {recon.shape} (Should be [5, 26])")
    print(f"Latent Mean (mu): {mu.shape}    (Should be [5, 4])")