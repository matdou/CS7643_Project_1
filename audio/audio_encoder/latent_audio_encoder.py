import torch
import torch.nn as nn
from transformers import Wav2Vec2Model, Wav2Vec2Processor
import numpy as np

class LatentAudioEncoder(nn.Module):
    def __init__(self, model_name="facebook/wav2vec2-base"):
        super().__init__()
        self.processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self"
        )
        # Load model in float32 to match input dtype
        self.model = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-large-960h-lv60-self",
            torch_dtype=torch.float32
        )

    def forward(self, waveform, sampling_rate=16000):
        """
        Args:
            waveform: Can be:
                - torch.Tensor of shape [num_samples] or [batch, num_samples]
                - numpy array of shape [num_samples]
                - list of numpy arrays
        """
        print(f"[LATENT] Input type: {type(waveform)}")
        
        # Convert to numpy if torch tensor
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().numpy()
            
            # Ensure proper shape: squeeze to 1D if needed
            while waveform.ndim > 1:
                waveform = waveform.squeeze(0)
                
        elif isinstance(waveform, np.ndarray):
            # Ensure 1D
            while waveform.ndim > 1:
                waveform = waveform.squeeze(0)

        if waveform.size == 0:
            print("[LATENT] Empty waveform detected — returning zero embedding.")
            return torch.zeros(1, 1024, device=next(self.model.parameters()).device)
        elif waveform.shape[0] < 320:  # less than ~20ms of audio at 16kHz
            pad_len = 320 - waveform.shape[0]
            print(f" [LATENT] Waveform too short ({waveform.shape[0]} samples). Padding to {320}.")
            waveform = np.pad(waveform, (0, pad_len))
        
        
        # Process audio - processor expects raw audio array or list of arrays
        inputs = self.processor(
            waveform,  # Pass the 1D numpy array directly
            sampling_rate=sampling_rate, 
            return_tensors="pt", 
            padding=True
        )
        
        print(f"[LATENT] Processor output shape: {inputs['input_values'].shape}")

        # Move model inputs to correct device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pool over time dimension to get fixed-size embedding
        embedding = outputs.last_hidden_state.mean(dim=1)
        print(f"[LATENT] Output embedding shape: {embedding.shape}")
        
        return embedding