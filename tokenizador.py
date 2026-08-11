import os
import torch

class TokenizadorLocal:
    def __init__(self, ruta_texto_pendrive="datos_entrenamiento.txt"):
        if not os.path.exists(ruta_texto_pendrive):
            raise FileNotFoundError(f"❌ Error: No se encuentra el archivo '{ruta_texto_pendrive}'. ¡Ejecuta primero convertir_datos.py!")
            
        with open(ruta_texto_pendrive, 'r', encoding='utf-8') as f:
            self.texto = f.read()
            
        self.caracteres = sorted(list(set(self.texto)))
        self.tamano_vocabulario = len(self.caracteres)
        self.char_a_int = { ch:i for i,ch in enumerate(self.caracteres) }
        self.int_a_char = { i:ch for i,ch in enumerate(self.caracteres) }
        
    def codificar(self, texto_plano):
        return [self.char_a_int[c] for c in texto_plano if c in self.char_a_int]
        
    def decodificar(self, lista_numeros):
        return ''.join([self.int_a_char[i] for i in lista_numeros])

def obtener_lote_entrenamiento(datos_codificados, tamano_bloque, tamano_lote):
    ix = torch.randint(len(datos_codificados) - tamano_bloque, (tamano_lote,))
    x = torch.stack([datos_codificados[i:i+tamano_bloque] for i in ix])
    y = torch.stack([datos_codificados[i+1:i+tamano_bloque+1] for i in ix])
    return x, y
