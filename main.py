import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from modelo import MiGeminiDesdeCero
from tokenizador import TokenizadorLocal, obtener_lote_entrenamiento

# Asegurar carpeta de trabajo correcta
carpeta_ia = os.path.dirname(os.path.abspath(__file__))
os.chdir(carpeta_ia)

# (Se quitó el forzado de todos los núcleos: en algunos procesadores con
# Windows puede causar que el entrenamiento se congele sin dar error.)

RUTA_TEXTO = "datos_entrenamiento.txt" 
RUTA_PESOS = "mi_gemini_pesos.pth"

TAMANO_LOTE = 32          
# Cambia esta línea que está cerca del principio de tu main.py
BLOQUE_CONTEXTO = 64     # Cambia el 128 que pusimos antes por este 64
PASOS_ENTRENAMIENTO = 10000 
TASA_APRENDIZAJE = 5e-4   # Ajustada para un aprendizaje más fino
FRECUENCIA_GUARDADO = 500 

dispositivo = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"--- Entrenando usando el dispositivo: {dispositivo.upper()} ---")

if not os.path.exists(RUTA_TEXTO):
    print(f"❌ Error: Falta '{RUTA_TEXTO}'.")
    exit()

tokenizador = TokenizadorLocal(RUTA_TEXTO)
vocab_size = tokenizador.tamano_vocabulario

texto_codificado = tokenizador.codificar(tokenizador.texto)
datos_tensores = torch.tensor(texto_codificado, dtype=torch.long).to(dispositivo)

n = int(0.9 * len(datos_tensores))
datos_entrenamiento = datos_tensores[:n]

modelo = MiGeminiDesdeCero(tamano_vocabulario=vocab_size).to(dispositivo)
optimizar = torch.optim.AdamW(modelo.parameters(), lr=TASA_APRENDIZAJE)

paso_inicial = 0

if os.path.exists(RUTA_PESOS):
    print("¡Cargando conocimientos previos de tu carpeta!")
    try:
        checkpoint = torch.load(RUTA_PESOS, map_location=dispositivo)
        if isinstance(checkpoint, dict) and "modelo" in checkpoint:
            # Formato nuevo: guarda el modelo Y el paso en el que se quedó
            modelo.load_state_dict(checkpoint["modelo"])
            paso_inicial = checkpoint.get("paso", 0) + 1
            print(f"✅ Continuando el entrenamiento desde el paso {paso_inicial}.")
        else:
            # Formato antiguo: solo el modelo, sin número de paso guardado
            modelo.load_state_dict(checkpoint)
            print("✅ Pesos cargados (formato antiguo, se reinicia el contador de pasos).")
    except Exception as e:
        print(f"⚠️ No se pudieron cargar los pesos anteriores: {e}")
        print("Empezando entrenamiento desde cero con un cerebro nuevo.")

print("Iniciando el entrenamiento real de tu IA...")
print(f"(Se guarda el progreso automáticamente cada {FRECUENCIA_GUARDADO} pasos. Puedes cerrar con Ctrl+C cuando quieras y no perderás el avance guardado.)\n")
modelo.train()

tiempo_inicio = time.time()

for paso in range(paso_inicial, PASOS_ENTRENAMIENTO):
    xb, yb = obtener_lote_entrenamiento(datos_entrenamiento, BLOQUE_CONTEXTO, TAMANO_LOTE)
    xb, yb = xb.to(dispositivo), yb.to(dispositivo)
    
    logits = modelo(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    yb = yb.view(B*T)
    error = F.cross_entropy(logits, yb)
    
    optimizar.zero_grad(set_to_none=True)
    error.backward()
    optimizar.step()
    
    if paso % 100 == 0:
        transcurrido = time.time() - tiempo_inicio
        pasos_hechos_ahora = paso - paso_inicial + 1
        segundos_por_paso = transcurrido / pasos_hechos_ahora
        pasos_restantes = PASOS_ENTRENAMIENTO - paso - 1
        minutos_restantes = (segundos_por_paso * pasos_restantes) / 60
        print(f"Paso {paso}/{PASOS_ENTRENAMIENTO} | Error real: {error.item():.4f} | Tiempo estimado restante: {minutos_restantes:.1f} min")
        
    if paso % FRECUENCIA_GUARDADO == 0 and paso > 0:
        torch.save({"modelo": modelo.state_dict(), "paso": paso}, RUTA_PESOS)

torch.save({"modelo": modelo.state_dict(), "paso": PASOS_ENTRENAMIENTO - 1}, RUTA_PESOS)
print("¡Entrenamiento finalizado con éxito!")