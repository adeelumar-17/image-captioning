# =============================================================================
# CHUNK 10: STREAMLIT APP DEPLOYMENT
# =============================================================================
# This cell creates an interactive Streamlit app for the image captioning model.
# Upload an image → Get AI-generated caption
# Supports both greedy and beam search decoding.
# Model files (.pth and .pkl) should be in the same directory as this script.
# =============================================================================

import streamlit as st
import torch
import pickle
import numpy as np
from PIL import Image
from torchvision import models, transforms
import torch.nn as nn
import torch.nn.functional as F

# =============================
# FILE CONFIGURATION
# =============================
# Replace with your Hugging Face repository ID
# Format: "username/repo-name"
HF_REPO_ID = "YOUR_USERNAME/image-captioning-model"  # <-- CHANGE THIS

# File names on Hugging Face Hub
MODEL_FILENAME = "best_caption_model.pth"
VOCAB_FILENAME = "vocabulary.pkl"


# =============================
# REDEFINE MODEL CLASSES
# =============================
# (These need to be defined here for the app to work independently)

class Encoder(nn.Module):
    def __init__(self, feature_dim=2048, hidden_size=512, num_layers=2, dropout=0.5):
        super(Encoder, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.fc = nn.Linear(feature_dim, hidden_size)
        self.bn = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.init_h = nn.Linear(hidden_size, hidden_size * num_layers)
        self.init_c = nn.Linear(hidden_size, hidden_size * num_layers)
    
    def forward(self, features):
        batch_size = features.size(0)
        x = self.fc(features)
        x = self.bn(x)
        x = torch.relu(x)
        x = self.dropout(x)
        hidden = self.init_h(x).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        cell = self.init_c(x).view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5):
        super(Decoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size, padding_idx=0)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, dropout=dropout if num_layers > 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, captions, hidden, cell):
        embeddings = self.dropout(self.embedding(captions))
        lstm_out, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
        return self.fc(lstm_out), hidden, cell
    
    def generate_step(self, word_idx, hidden, cell):
        if word_idx.dim() == 1:
            word_idx = word_idx.unsqueeze(1)
        embeddings = self.embedding(word_idx)
        lstm_out, (hidden, cell) = self.lstm(embeddings, (hidden, cell))
        return self.fc(lstm_out.squeeze(1)), hidden, cell


class ImageCaptioningModel(nn.Module):
    def __init__(self, vocab_size, feature_dim=2048, embed_size=256, hidden_size=512, num_layers=2, dropout=0.5):
        super(ImageCaptioningModel, self).__init__()
        self.encoder = Encoder(feature_dim, hidden_size, num_layers, dropout)
        self.decoder = Decoder(vocab_size, embed_size, hidden_size, num_layers, dropout)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
    
    def forward(self, features, captions):
        hidden, cell = self.encoder(features)
        outputs, _, _ = self.decoder(captions[:, :-1], hidden, cell)
        return outputs


# =============================
# LOAD MODELS AND VOCABULARY
# =============================

@st.cache_resource
def load_models():
    """Load all models with caching for performance."""
    # Device (use CPU for Streamlit Cloud compatibility)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load vocabulary from local file
    with open(VOCAB_PATH, 'rb') as f:
        vocab = pickle.load(f)
    
    # Load ResNet50 for feature extraction
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    resnet = nn.Sequential(*list(resnet.children())[:-1])
    resnet = resnet.to(device)
    resnet.eval()
    
    # Image transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    
    # Load caption model
    caption_model = ImageCaptioningModel(
        vocab_size=len(vocab),
        feature_dim=2048,
        embed_size=256,
        hidden_size=512,
        num_layers=2,
        dropout=0.5
    ).to(device)
    
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    caption_model.load_state_dict(checkpoint['model_state_dict'])
    caption_model.eval()
    
    return device, vocab, resnet, transform, caption_model


# =============================
# INFERENCE FUNCTIONS
# =============================

def extract_features(image, resnet, transform, device):
    """Extract features from PIL image using ResNet50."""
    img_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        features = resnet(img_tensor).view(1, -1)
    return features


def generate_caption_app(model, features, vocab, device, method='beam', beam_width=5, max_len=50):
    """Generate caption for the app."""
    model.eval()
    
    with torch.no_grad():
        hidden, cell = model.encoder(features)
        
        if method == 'greedy':
            word_idx = torch.tensor([vocab.start_idx], device=device)
            words = []
            for _ in range(max_len):
                output, hidden, cell = model.decoder.generate_step(word_idx, hidden, cell)
                word_idx = output.argmax(dim=1)
                if word_idx.item() == vocab.end_idx:
                    break
                words.append(vocab.idx2word.get(word_idx.item(), '<unk>'))
            return ' '.join(words)
        
        else:  # Beam search
            beams = [(0.0, [vocab.start_idx], hidden, cell)]
            completed = []
            
            for _ in range(max_len):
                candidates = []
                for score, seq, h, c in beams:
                    if seq[-1] == vocab.end_idx:
                        completed.append((score, seq))
                        continue
                    word_idx = torch.tensor([seq[-1]], device=device)
                    output, new_h, new_c = model.decoder.generate_step(word_idx, h, c)
                    log_probs = F.log_softmax(output, dim=1).squeeze(0)
                    top_probs, top_idx = log_probs.topk(beam_width)
                    for prob, idx in zip(top_probs.tolist(), top_idx.tolist()):
                        candidates.append((score + prob, seq + [idx], new_h, new_c))
                beams = sorted(candidates, key=lambda x: x[0], reverse=True)[:beam_width]
                if all(b[1][-1] == vocab.end_idx for b in beams):
                    break
            
            completed.extend([(s, seq) for s, seq, _, _ in beams])
            best = max(completed, key=lambda x: x[0] / len(x[1]))
            words = [vocab.idx2word.get(idx, '<unk>') for idx in best[1][1:] 
                     if idx not in [vocab.start_idx, vocab.end_idx, vocab.pad_idx]]
            return ' '.join(words)


# =============================
# STREAMLIT APP INTERFACE
# =============================

def main():
    # Page configuration
    st.set_page_config(
        page_title="Neural Storyteller",
        page_icon="🖼️",
        layout="centered"
    )
    
    # Title and description
    st.title("🖼️ Neural Storyteller")
    st.markdown("### Image Captioning with Seq2Seq")
    st.markdown("""
    Upload an image and get an AI-generated caption!
    
    This model uses a **Seq2Seq architecture** with:
    - **Encoder**: ResNet50 for image feature extraction
    - **Decoder**: LSTM for caption generation
    """)
    
    st.divider()
    
    # Load models
    with st.spinner("Loading models..."):
        device, vocab, resnet, transform, caption_model = load_models()
    
    # Sidebar for settings
    st.sidebar.header("⚙️ Settings")
    method = st.sidebar.radio(
        "Decoding Method",
        ["Beam Search", "Greedy"],
        help="Beam Search produces better quality but is slower"
    )
    
    beam_width = 5
    if method == "Beam Search":
        beam_width = st.sidebar.slider(
            "Beam Width",
            min_value=1,
            max_value=10,
            value=5,
            help="Higher values explore more options but are slower"
        )
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...",
        type=["jpg", "jpeg", "png"],
        help="Upload an image to generate a caption"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_column_width=True)
        
        with col2:
            # Generate caption button
            if st.button("🚀 Generate Caption", type="primary", use_container_width=True):
                with st.spinner("Generating caption..."):
                    # Extract features
                    features = extract_features(image, resnet, transform, device)
                    
                    # Generate caption
                    caption = generate_caption_app(
                        caption_model, features, vocab, device,
                        method=method.lower().replace(" ", ""),
                        beam_width=beam_width
                    )
                
                # Display result
                st.success("Caption generated!")
                st.markdown("### 📝 Generated Caption:")
                st.markdown(f"> **{caption}**")
                
                # Show method used
                st.caption(f"Generated using {method}" + 
                          (f" (width={beam_width})" if method == "Beam Search" else ""))
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
        <small>Built with ❤️ using PyTorch and Streamlit | 
        Trained on Flickr30k Dataset</small>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
