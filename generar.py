import os
import torch
import torch.nn.functional as F
# Importamos tu arquitectura y tu tokenizador personalizados
from modelo import MiGeminiDesdeCero
from tokenizador import TokenizadorLocal

# ==========================================
# 1. CONFIGURACIÓN Y RUTAS (Ajusta según tu PC)
# ==========================================
# Las mismas rutas que definiste en tu archivo main.py
RUTA_TEXTO_PENDRIVE = "datos_entrenamiento.txt"
RUTA_PESOS_PENDRIVE = "mi_gemini_pesos.pth"

# Parámetros para controlar cómo habla la IA
MAX_TOKENS_RESPUESTA = 150  # Límite de caracteres en su respuesta
TEMPERATURA = 0.7           # Creatividad (1.0 = muy creativo/caótico, 0.2 = muy serio/repetitivo)

# Configurar hardware
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

# ==========================================
# 2. CARGAR TRADUCTOR Y CEREBRO ENTRENADO
# ==========================================
if not os.path.exists(RUTA_PESOS_PENDRIVE):
    print(f"❌ Error: No se encontró el archivo '{RUTA_PESOS_PENDRIVE}'.")
    print("Debes ejecutar primero 'main.py' para entrenar al modelo y guardar sus pesos.")
    exit()

# Cargar el tokenizador para traducir letras
tokenizador = TokenizadorLocal(RUTA_TEXTO_PENDRIVE)
vocab_size = tokenizador.tamano_vocabulario

# Reconstruir la estructura vacía del cerebro
modelo = MiGeminiDesdeCero(tamano_vocabulario=vocab_size)
modelo = modelo.to(dispositivo)

# Cargar los conocimientos (pesos matemáticos) desde tu pendrive
print("🧠 Cargando los conocimientos de la IA desde el pendrive...")
checkpoint = torch.load(RUTA_PESOS_PENDRIVE, map_location=dispositivo)
if isinstance(checkpoint, dict) and "modelo" in checkpoint:
    modelo.load_state_dict(checkpoint["modelo"])  # formato nuevo
    print(f"   (Entrenado hasta el paso {checkpoint.get('paso', '?')})")
else:
    modelo.load_state_dict(checkpoint)  # formato antiguo
modelo.eval() # Activar modo de evaluación (apaga el dropout)

print("\n🤖 ¡IA cargada con éxito! Escribe 'salir' para cerrar el chat.")
print("-" * 50)

# ==========================================
# 3. INTERFAZ DE DIÁLOGO INTERACTIVA
# ==========================================
while True:
    # Leer lo que escribe el usuario por consola
    entrada_usuario = input("\n👤 Tú: ")
    if entrada_usuario.lower() == 'salir':
        print("¡Adiós! Cerrando interfaz de chat.")
        break

    if not entrada_usuario.strip():
        continue

    # Formatear la frase con las etiquetas con las que fue entrenada
    frase_formateada = f"<|usuario|> {entrada_usuario}\n<|asistente|>"

    # Traducir el texto a números y pasarlo a un Tensor de PyTorch
    contexto_numeros = tokenizador.codificar(frase_formateada)
    x = torch.tensor([contexto_numeros], dtype=torch.long, device=dispositivo)

    print("🤖 IA: ", end="", flush=True)

    # Bucle de generación letra por letra (Auto-regresivo)
    with torch.no_grad():
        for _ in range(MAX_TOKENS_RESPUESTA):
            # Recortar el contexto si supera el tamaño de bloque que la IA puede recordar
            x_cond = x[:, -64:] # 64 es el BLOQUE_CONTEXTO definido originalmente

            # Predecir las probabilidades de la siguiente letra
            logits = modelo(x_cond)
            logits = logits[:, -1, :] / TEMPERATURA # Aplicar temperatura para regular creatividad

            # Convertir probabilidades en la siguiente letra elegida
            probs = F.softmax(logits, dim=-1)
            siguiente_token = torch.multinomial(probs, num_samples=1)

            # Detenerse si la IA decide escribir una nueva etiqueta de usuario (fin de respuesta)
            letra_decodificada = tokenizador.decodificar([siguiente_token.item()])
            if "<|" in letra_decodificada:
                break

            # Imprimir la letra en la pantalla al instante
            print(letra_decodificada, end="", flush=True)

            # Añadir la letra recién generada al historial para calcular la siguiente
            x = torch.cat((x, siguiente_token), dim=1)

    print() # Salto de línea final al terminar la respuesta