import json
import os
import re
import sys

# 🔧 Caracteres permitidos: alfabeto español, números, puntuación común
# y las etiquetas especiales de formato. Todo lo demás (emojis, símbolos
# de otros idiomas/alfabetos, caracteres raros de codificación) se elimina.
# Esto reduce muchísimo el tamaño del vocabulario y hace que un modelo
# pequeño aprenda español "de verdad" mucho más rápido.
PATRON_PERMITIDO = re.compile(
    r"[^a-zA-Z0-9áéíóúÁÉÍÓÚñÑüÜ\s\.,;:!¡?¿'\"()\-\n<>|]"
)

def limpiar_texto(texto):
    # Quita cualquier carácter que no esté en la lista permitida
    texto = PATRON_PERMITIDO.sub("", texto)
    # Colapsa espacios/tabulaciones repetidos en uno solo (pero conserva saltos de línea)
    texto = re.sub(r"[ \t]+", " ", texto)
    return texto.strip()

# 🛠️ TRUCO DEFENSIVO: Obligar a Windows a trabajar DENTRO de tu carpeta MiGemini
carpeta_de_tu_ia = os.path.dirname(os.path.abspath(__file__))
os.chdir(carpeta_de_tu_ia)

archivo_json = "alpaca_data_cleaned_spanish.json"
archivo_salida = "datos_entrenamiento.txt"

print("==================================================")
print(f"📁 CARPETA DETECTADA POR WINDOWS: {carpeta_de_tu_ia}")
print("==================================================")
print(f"🔎 Buscando el archivo '{archivo_json}'...")

if not os.path.exists(archivo_json):
    print("\n❌ ==================== ERROR CRÍTICO ====================")
    print(f"No se encuentra el archivo '{archivo_json}' en esta carpeta.")
    print(f"Ruta donde lo busca Python: {os.path.join(carpeta_de_tu_ia, archivo_json)}")
    print("Asegúrate de mover el archivo JSON descargado AQUÍ dentro.")
    print("=========================================================\n")
    sys.exit(1)

print("✅ Archivo JSON encontrado con éxito.")
print("🔄 Leyendo y transformando más de 50.000 conversaciones...")

with open(archivo_json, "r", encoding="utf-8") as f:
    datos = json.load(f)

print(f"📦 Total de diálogos detectados en el archivo: {len(datos)}")

with open(archivo_salida, "w", encoding="utf-8") as f_salida:
    for elemento in datos:
        pregunta = elemento.get("instruction", "")
        contexto_extra = elemento.get("input", "")
        respuesta = elemento.get("output", "")
        
        if contexto_extra:
            pregunta = f"{pregunta}\nContexto: {contexto_extra}"

        texto_formateado = f"<|usuario|> {limpiar_texto(pregunta)}\n<|asistente|> {limpiar_texto(respuesta)}\n\n"
        f_salida.write(texto_formateado)

print(f"🎉 ¡CONVERSIÓN COMPLETADA! Generado '{archivo_salida}' correctamente.\n")