import torch
import torch.nn as nn

class HybridModel(nn.Module):
    def __init__(self, lcnn_model, spectrogram_model):
        super(HybridModel, self).__init__()
        self.lcnn_model = lcnn_model
        self.spectrogram_model = spectrogram_model
        
        # Fusion network
        self.fusion = nn.Sequential(
            nn.Linear(1024, 512),  # 512 from LCNN + 512 from Spectrogram
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 5),  # 5 feature scores
            nn.Sigmoid()
        )
        
        self.final_classifier = nn.Sequential(
            nn.Linear(5, 1),
            nn.Sigmoid()
        )
    
    def forward(self, waveform, spectrogram):
        # Get features from both models
        _, lcnn_features = self.lcnn_model(waveform)
        _, spec_features = self.spectrogram_model(spectrogram)
        
        # Concatenate features
        combined_features = torch.cat((lcnn_features, spec_features), dim=1)
        
        # Get feature scores through fusion network
        feature_scores = self.fusion(combined_features)
        
        # Final classification
        prediction = self.final_classifier(feature_scores)
        
        return prediction, feature_scores