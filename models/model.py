# SW-MAIN/deeplearning/models/model.py

import torch
import torch.nn as nn
from deeplearning.models.lightweight_cnn_v5 import ImprovedLightweightCNN_v5

class CNNAE_LSTM_Transformer(nn.Module):
    """
    5차 개선 모델:
    - CNN 백본: ImprovedLightweightCNN_v5 (Bottleneck + SEBlock)
    - GRU
    - AutoEncoder
    - Transformer
    """
    def __init__(self, 
                 ae_latent_dim=128,
                 gru_hidden_dim=256,
                 lstm_num_layers=2,
                 transformer_d_model=256,
                 transformer_nhead=4,
                 transformer_num_layers=1,
                 num_classes=2,
                 cnn_feature_dim=256):
        super().__init__()
        
        self.cnn = ImprovedLightweightCNN_v5(in_channels=3, feature_dim=cnn_feature_dim)
        self.cnn_output_dim = cnn_feature_dim
        
        self.ae_encoder = nn.Sequential(
            nn.Linear(self.cnn_output_dim, 512),
            nn.ReLU(),
            nn.Linear(512, ae_latent_dim),
            nn.ReLU()
        )
        self.ae_decoder = nn.Sequential(
            nn.Linear(ae_latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, self.cnn_output_dim)
        )
        
        self.gru = nn.GRU(
            input_size=ae_latent_dim,
            hidden_size=gru_hidden_dim,
            num_layers=lstm_num_layers,
            dropout=0.2,
            batch_first=True
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=transformer_nhead,
            dim_feedforward=transformer_d_model*4,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_num_layers)
        
        self.fc_cls = nn.Linear(transformer_d_model, num_classes)

    def forward(self, x):
        B, seq, C, H, W = x.shape
        x = x.view(B*seq, C, H, W)
        feat_orig = self.cnn(x)  # [B*seq, 256]

        latent = self.ae_encoder(feat_orig)  # [B*seq, ae_latent_dim]
        feat_recon = self.ae_decoder(latent) # [B*seq, 256]

        latent_seq = latent.view(B, seq, -1)
        gru_out, _ = self.gru(latent_seq)     # [B, seq, gru_hidden_dim]

        trans_out = self.transformer(gru_out) # [B, seq, transformer_d_model]
        final_out = trans_out[:, -1, :]       # [B, transformer_d_model]
        logits = self.fc_cls(final_out)
        
        return logits, feat_orig, feat_recon

if __name__ == "__main__":
    model = CNNAE_LSTM_Transformer()
    dummy_input = torch.randn(2, 30, 3, 224, 224)
    logits, feat_orig, feat_recon = model(dummy_input)
    print("Logits shape:", logits.shape)
    print("feat_orig shape:", feat_orig.shape)
    print("feat_recon shape:", feat_recon.shape)
