import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np

# Definindo o dispositivo (CPU ou GPU) - Foco no uso eficiente
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================================
# 1. FUNÇÃO: CARREGAR O MODELO (FEATURE EXTRACTOR)
# ==============================================================
@st.cache_resource # Cacheia o modelo para não recarregar a cada interação
def load_feature_extractor():
    # Usamos o MobileNetV2 (pequeno e eficiente) pré-treinado no ImageNet
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # ⚠️ A chave do FSL é remover a última camada (classificador)
    # para usar a penúltima camada como nosso Feature Extractor (embedding)
    model.classifier = nn.Identity() # Remove a camada de classificação
    
    # Coloca o modelo em modo de avaliação (importante para modelos pré-treinados)
    model.eval()
    model.to(DEVICE)
    st.info(f"🧠 Modelo MobileNetV2 carregado com sucesso no {DEVICE.type}!")
    return model

# ==============================================================
# 2. FUNÇÃO: PROCESSAR IMAGEM E GERAR EMBEDDING (VETOR)
# ==============================================================
def preprocess_and_embed(image: Image.Image, model: nn.Module) -> np.ndarray:
    # ⚙️ Transformações necessárias para MobileNet
    transform = transforms.Compose([
        transforms.Resize((224, 224)), # Redimensiona (padrão 224x224 para MobileNet)
        transforms.ToTensor(),         # Converte para tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Normalização ImageNet
    ])

    # 1. Pré-processamento
    input_tensor = transform(image).unsqueeze(0).to(DEVICE) # Adiciona a dimensão do lote (batch)

    # 2. Geração do Embedding
    with torch.no_grad():
        embedding_tensor = model(input_tensor)
        
    # Converte o tensor para um array NumPy para facilitar os cálculos de distância
    return embedding_tensor.cpu().numpy().flatten()

# ==============================================================
# 3. FUNÇÃO: CALCULAR A DISTÂNCIA DE SIMILARIDADE
# ==============================================================
def calculate_similarity(embedding_a: np.ndarray, embedding_b: np.ndarray) -> float:
    # 📏 Distância Coseno é a métrica padrão para similaridade de embeddings
    # Quanto mais próximo de 1, mais similares são.
    
    dot_product = np.dot(embedding_a, embedding_b)
    norm_a = np.linalg.norm(embedding_a)
    norm_b = np.linalg.norm(embedding_b)
    
    if norm_a == 0 or norm_b == 0:
        return 0.0 # Evita divisão por zero
        
    cosine_similarity = dot_product / (norm_a * norm_b)
    return float(cosine_similarity) # Retorna um float simples
