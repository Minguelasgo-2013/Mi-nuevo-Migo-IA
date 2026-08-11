import torch
import torch.nn as nn

# Valores equilibrados: Inteligente pero rápido en cualquier CPU
BLOQUE_CONTEXTO = 64  
DIM_EMBEDDING = 128   
NUM_CABEZAS = 4       
NUM_CAPAS = 3         # Bajamos a 3 capas para quitarle peso al procesador


class BloqueAtencion(nn.Module):
    def __init__(self, dim_emb, num_cabezas):
        super().__init__()
        self.atencion = nn.MultiheadAttention(embed_dim=dim_emb, num_heads=num_cabezas, batch_first=True)
        self.norma1 = nn.LayerNorm(dim_emb)
        self.norma2 = nn.LayerNorm(dim_emb)
        
        self.feed_forward = nn.Sequential(
            nn.Linear(dim_emb, 4 * dim_emb),
            nn.ReLU(),
            nn.Linear(4 * dim_emb, dim_emb)
        )

    def forward(self, x):
        T = x.size(1)
        # 🔧 CORRECCIÓN: máscara causal para que cada posición solo pueda
        # atender a sí misma y a las posiciones anteriores, nunca a las futuras.
        # Sin esto, el modelo "hace trampa" durante el entrenamiento y aprende
        # muy mal a generar texto nuevo.
        mascara_causal = torch.triu(
            torch.full((T, T), float('-inf'), device=x.device),
            diagonal=1
        )
        atendido, _ = self.atencion(x, x, x, attn_mask=mascara_causal)
        x = self.norma1(x + atendido)
        procesado = self.feed_forward(x)
        x = self.norma2(x + procesado)
        return x

class MiGeminiDesdeCero(nn.Module):
    def __init__(self, tamano_vocabulario):
        super().__init__()
        self.embedding_tokens = nn.Embedding(tamano_vocabulario, DIM_EMBEDDING)
        self.embedding_posicional = nn.Embedding(BLOQUE_CONTEXTO, DIM_EMBEDDING)
        self.capas = nn.Sequential(*[BloqueAtencion(DIM_EMBEDDING, NUM_CABEZAS) for _ in range(NUM_CAPAS)])
        self.salida_lineal = nn.Linear(DIM_EMBEDDING, tamano_vocabulario)

    def forward(self, indices_entrada):
        B, T = indices_entrada.shape
        emb_tok = self.embedding_tokens(indices_entrada)
        emb_pos = self.embedding_posicional(torch.arange(T, device=indices_entrada.device))
        x = emb_tok + emb_pos
        x = self.capas(x)
        logits = self.salida_lineal(x)
        return logits