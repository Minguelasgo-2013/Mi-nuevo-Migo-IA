import os
import re
import torch

# 🔧 CAMBIO IMPORTANTE: tokenización por PALABRAS en vez de por CARACTERES.
# Antes, el modelo tenía que aprender a la vez cómo se deletrea cada palabra
# Y la gramática Y el significado, letra por letra — la tarea más difícil
# posible para un modelo tan pequeño. Ahora cada palabra (y cada signo de
# puntuación) es una sola "ficha", así que el modelo se puede concentrar en
# aprender a combinar palabras con sentido.

# Reconoce, en este orden: las etiquetas especiales completas, luego
# secuencias de letras/números (palabras), luego cualquier signo de
# puntuación suelto.
PATRON_TOKENS = re.compile(r"<\|usuario\|>|<\|asistente\|>|\w+|[^\w\s]", re.UNICODE)

TOKEN_DESCONOCIDO = "<UNK>"

# Para reconstruir el texto con espacios en el lugar correcto
SIN_ESPACIO_ANTES = set(list(",.;:!?)]}»") + ["’", "”"])
SIN_ESPACIO_DESPUES = set(list("([{«") + ["‘", "“"])


class TokenizadorLocal:
    def __init__(self, ruta_texto_pendrive="datos_entrenamiento.txt"):
        if not os.path.exists(ruta_texto_pendrive):
            raise FileNotFoundError(f"❌ Error: No se encuentra el archivo '{ruta_texto_pendrive}'. ¡Ejecuta primero convertir_datos.py!")

        with open(ruta_texto_pendrive, 'r', encoding='utf-8') as f:
            self.texto = f.read()

        palabras = self._tokenizar(self.texto)

        self.palabras_unicas = sorted(set(palabras))
        self.palabras_unicas.append(TOKEN_DESCONOCIDO)  # para palabras nunca vistas

        self.tamano_vocabulario = len(self.palabras_unicas)
        self.token_a_int = {t: i for i, t in enumerate(self.palabras_unicas)}
        self.int_a_token = {i: t for i, t in enumerate(self.palabras_unicas)}

    def _tokenizar(self, texto_plano):
        return PATRON_TOKENS.findall(texto_plano)

    def codificar(self, texto_plano):
        # Acepta tanto un string (lo tokeniza) como una lista ya tokenizada
        tokens = self._tokenizar(texto_plano) if isinstance(texto_plano, str) else texto_plano
        id_desconocido = self.token_a_int[TOKEN_DESCONOCIDO]
        return [self.token_a_int.get(t, id_desconocido) for t in tokens]

    def decodificar(self, lista_numeros):
        tokens = [self.int_a_token[i] for i in lista_numeros if i in self.int_a_token]
        return self._unir_tokens(tokens)

    def _unir_tokens(self, tokens):
        # Reconstruye el texto poniendo espacios entre palabras, pero sin
        # espacio antes de comas, puntos, signos de cierre, etc.
        texto = ""
        for i, tok in enumerate(tokens):
            if tok == TOKEN_DESCONOCIDO:
                continue
            if not texto:
                texto = tok
                continue
            if tok in SIN_ESPACIO_ANTES:
                texto += tok
            elif texto[-1] in SIN_ESPACIO_DESPUES:
                texto += tok
            else:
                texto += " " + tok
        return texto


def obtener_lote_entrenamiento(datos_codificados, tamano_bloque, tamano_lote):
    ix = torch.randint(len(datos_codificados) - tamano_bloque, (tamano_lote,))
    x = torch.stack([datos_codificados[i:i+tamano_bloque] for i in ix])
    y = torch.stack([datos_codificados[i+1:i+tamano_bloque+1] for i in ix])
    return x, y