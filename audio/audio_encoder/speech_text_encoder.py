import torch
import torch.nn as nn
from transformers import WhisperProcessor, WhisperForConditionalGeneration, AutoTokenizer, AutoModel
import numpy as np

class SpeechTextEncoder(nn.Module):
    def __init__(self,
                 asr_model="openai/whisper-base",
                 text_model="sentence-transformers/all-MiniLM-L6-v2",
                 translate_to_english=True):
        super().__init__()
        self.translate_to_english = translate_to_english
        self.whisper_processor = WhisperProcessor.from_pretrained(asr_model)
        self.whisper = WhisperForConditionalGeneration.from_pretrained(asr_model)
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model)
        self.text_model = AutoModel.from_pretrained(text_model)

    def forward(self, waveform, sampling_rate=16000):
        """
        Args:
            waveform: Can be:
                - torch.Tensor of shape [num_samples] or [batch, num_samples]
                - numpy array of shape [num_samples]
        """
        
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
        
        
        # Step 1: Process audio for Whisper
        input_feat = self.whisper_processor(
            waveform, 
            sampling_rate=sampling_rate, 
            return_tensors="pt"
        )
        
        #print(f"[SPEECH] Whisper input features shape: {input_feat['input_features'].shape}")
        
        # Move to same device as model
        input_feat = {k: v.to(self.whisper.device) for k, v in input_feat.items()}

        # Step 2: Generate text (transcription or translation)
        generate_opts = {"task": "translate"} if self.translate_to_english else {}
        
        with torch.no_grad():
            transcription_ids = self.whisper.generate(
                input_feat["input_features"], **generate_opts
            )

        # Decode to text
        transcript = self.whisper_processor.batch_decode(
            transcription_ids, skip_special_tokens=True
        )[0].strip()
        
        #print(f"[SPEECH] Transcript: '{transcript}'")

        # Step 3: Text embedding (using English embeddings)
        text_inputs = self.text_tokenizer(
            transcript, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=512
        )
        
        # Move to same device as text model
        text_inputs = {k: v.to(self.text_model.device) for k, v in text_inputs.items()}
        
        with torch.no_grad():
            outputs = self.text_model(**text_inputs)
        
        text_emb = outputs.last_hidden_state.mean(dim=1)
        
        #print(f"[SPEECH] Text embedding shape: {text_emb.shape}")

        return text_emb, transcript