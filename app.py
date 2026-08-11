import streamlit as nn_web
import torch
import torch.nn.functional as F
import os
import time

# Importamos las piezas reales de tu proyecto
from modelo import MiGeminiDesdeCero
from tokenizador import TokenizadorLocal

# Configuración visual de la pestaña de Google
nn_web.set_page_config(page_title="Mi Gemini Local", page_icon="🤖", layout="centered")

# Configurar rutas locales
RUTA_TEXTO = "datos_entrenamiento.txt"
RUTA_PESOS = "mi_gemini_pesos.pth"
dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'

# Función especial para cargar el cerebro una sola vez y que la web vaya rápida
@nn_web.cache_resource
def cargar_cerebro_ia():
    if not os.path.exists(RUTA_TEXTO):
        return None, None

    tokenizador = TokenizadorLocal(RUTA_TEXTO)
    vocab_size = tokenizador.tamano_vocabulario

    modelo = MiGeminiDesdeCero(tamano_vocabulario=vocab_size).to(dispositivo)

    # Si ya entrenaste un poco con main.py, cargamos tus pesos guardados
    if os.path.exists(RUTA_PESOS):
        checkpoint = torch.load(RUTA_PESOS, map_location=dispositivo)
        if isinstance(checkpoint, dict) and "modelo" in checkpoint:
            modelo.load_state_dict(checkpoint["modelo"])  # formato nuevo
        else:
            modelo.load_state_dict(checkpoint)  # formato antiguo
    modelo.eval()
    return modelo, tokenizador

with nn_web.spinner("🧠 Cargando el cerebro de la IA..."):
    modelo, tokenizador = cargar_cerebro_ia()

# ==========================================
# PANEL LATERAL: controles y estado
# ==========================================
with nn_web.sidebar:
    nn_web.header("⚙️ Ajustes")

    temperatura = nn_web.slider(
        "🎨 Creatividad (temperatura)",
        min_value=0.1, max_value=1.5, value=0.7, step=0.1,
        help="Valores bajos = respuestas más serias y repetitivas. Valores altos = más creativas y caóticas."
    )

    max_tokens = nn_web.slider(
        "✏️ Longitud máxima de respuesta",
        min_value=20, max_value=300, value=100, step=10
    )

    nn_web.divider()
    nn_web.subheader("📊 Estado del modelo")

    if modelo is None or tokenizador is None:
        nn_web.error("Sin datos de entrenamiento")
    elif not os.path.exists(RUTA_PESOS):
        nn_web.warning("Modelo sin entrenar (pesos no encontrados)")
    else:
        nn_web.success("Modelo cargado y listo")
        _checkpoint_info = torch.load(RUTA_PESOS, map_location=dispositivo)
        if isinstance(_checkpoint_info, dict) and "paso" in _checkpoint_info:
            nn_web.markdown(f"**Entrenado hasta el paso:** `{_checkpoint_info['paso']}`")

    nn_web.markdown(f"**Dispositivo:** `{dispositivo.upper()}`")
    if tokenizador is not None:
        nn_web.markdown(f"**Vocabulario:** `{tokenizador.tamano_vocabulario}` caracteres")

    nn_web.divider()
    if nn_web.button("🗑️ Borrar conversación", use_container_width=True):
        nn_web.session_state.mensajes = []
        nn_web.rerun()

# ==========================================
# CABECERA PRINCIPAL
# ==========================================
nn_web.title("🤖 Mi Gemini Conectado")
nn_web.subheader("Ejecutando lógica Transformer real desde tu carpeta")

# Avatares para diferenciar quién habla
AVATARES = {"user": "👤", "assistant": "🤖"}

# Crear el historial de la conversación en la web si no existe
if "mensajes" not in nn_web.session_state:
    nn_web.session_state.mensajes = []

# Mostrar los mensajes anteriores en la pantalla web
for mensaje in nn_web.session_state.mensajes:
    with nn_web.chat_message(mensaje["rol"], avatar=AVATARES.get(mensaje["rol"])):
        nn_web.markdown(mensaje["texto"])

# Caja de texto abajo para que el usuario escriba
if entrada_usuario := nn_web.chat_input("Escribe un mensaje a tu IA..."):

    with nn_web.chat_message("user", avatar=AVATARES["user"]):
        nn_web.markdown(entrada_usuario)
    nn_web.session_state.mensajes.append({"rol": "user", "texto": entrada_usuario})

    with nn_web.chat_message("assistant", avatar=AVATARES["assistant"]):
        contenedor_respuesta = nn_web.empty()

        # Si el usuario no ha generado los datos de entrenamiento todavía
        if modelo is None or tokenizador is None:
            respuesta_error = "❌ Error: No encuentro el archivo 'datos_entrenamiento.txt'. Por favor, ejecuta primero el convertidor de datos (`convertir_datos.py`)."
            contenedor_respuesta.markdown(respuesta_error)
            nn_web.session_state.mensajes.append({"rol": "assistant", "texto": respuesta_error})
        elif not os.path.exists(RUTA_PESOS):
            respuesta_error = "⚠️ Todavía no he sido entrenado (no encuentro `mi_gemini_pesos.pth`). Ejecuta `entrenar.bat` primero y luego vuelve a intentarlo."
            contenedor_respuesta.markdown(respuesta_error)
            nn_web.session_state.mensajes.append({"rol": "assistant", "texto": respuesta_error})
        else:
            # 🧠 AQUÍ COMIENZA LA LÓGICA DE TU TRANSFORMER REAL
            frase_formateada = f"<|usuario|> {entrada_usuario}\n<|asistente|>"
            contexto_numeros = tokenizador.codificar(frase_formateada)
            x = torch.tensor([contexto_numeros], dtype=torch.long, device=dispositivo)

            respuesta_completa = ""

            with torch.no_grad():
                for _ in range(max_tokens):
                    x_cond = x[:, -64:]  # BLOQUE_CONTEXTO
                    logits = modelo(x_cond)
                    logits = logits[:, -1, :] / temperatura  # Temperatura (creatividad)

                    probs = F.softmax(logits, dim=-1)
                    siguiente_token = torch.multinomial(probs, num_samples=1)

                    letra_decodificada = tokenizador.decodificar([siguiente_token.item()])

                    # Si la IA decide terminar la frase o abrir otra etiqueta, frena
                    if "<|" in letra_decodificada:
                        break

                    respuesta_completa += letra_decodificada
                    contenedor_respuesta.markdown(respuesta_completa + "▌")
                    x = torch.cat((x, siguiente_token), dim=1)
                    time.sleep(0.01)  # Simulación de fluidez visual

            contenedor_respuesta.markdown(respuesta_completa if respuesta_completa.strip() else "*(La IA generó un silencio, necesita más entrenamiento)*")
            nn_web.session_state.mensajes.append({"rol": "assistant", "texto": respuesta_completa})