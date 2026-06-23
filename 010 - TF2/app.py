import os
import gradio as gr
from transformers import pipeline

print("Configurando las capas de la aplicación...")

# =====================================================================
# 1. DATA LAYER (Capa de Datos): Carga del modelo preentrenado
# =====================================================================
print("[Data Layer] Cargando modelo preentrenado...")
clasificador_modelo = pipeline(
    "image-classification",
    model="watersplash/waste-classification"
)
print("✓ [Data Layer] Modelo cargado con éxito en memoria.")

# El modelo cargado devuelve las clases de cada material en inglés. Las traducimos para mostrar la respuesta en castellano.
TRADUCCIONES = {
    "battery": "Pila/Batería",
    "biological": "Orgánico",
    "brown-glass": "Vidrio color ámbar",
    "cardboard": "Cartón",
    "clothes": "Ropa",
    "green-glass": "Vidrio verde",
    "metal": "Metal",
    "paper": "Papel",
    "plastic": "Plástico",
    "shoes": "Calzado",
    "trash": "Basura general",
    "white-glass": "Vidrio sin color",
}

# Recomendación práctica de manejo y/o reciclaje según la categoría detectada
RECOMENDACIONES = {
    "Pila/Batería": "⚠️ Es un residuo peligroso. No lo tires al tacho de basura común: llevalo a un punto de recolección especial.",
    "Orgánico": "🌱 Es ideal para el compost. Separalo por completo del resto de los residuos.",
    "Vidrio color ámbar": "♻️ Enjuagá el envase y retirá la tapa metálica antes de llevarlo a reciclar.",
    "Cartón": "♻️ Desarmá las cajas y retirá los excedentes de cintas, ganchos metálicos, stickers o restos de plástico antes de reciclar.",
    "Ropa": "👕 Si está en buen estado, donala. También podés buscar un punto de reciclaje de textil.",
    "Vidrio verde": "♻️ Enjuagá el envase y retirá la tapa antes de llevarlo a reciclar.",
    "Metal": "♻️ Enjuagá latas de comida o bebida y aplastalas para ahorrar espacio. Las latas rígidas pueden reutilizarse.",
    "Papel": "♻️ Reciclalo si está limpio y seco. Evitá papel sucio, plastificado o con grasa.",
    "Plástico": "♻️ Lavá y secá el envase. Si podés, separalo por tipo de plástico (PET, HDPE, PVC, LDPE, PP, PS, otros).",
    "Calzado": "👟 Si está en buen estado, donalo. También podés buscar un punto de reciclaje de calzado/textil para dejarlo.",
    "Basura general": "🗑️ No es reciclable: va al cesto de residuos comunes.",
    "Vidrio sin color": "♻️ Enjuagá el envase y retirá la tapa antes de llevarlo a reciclar.",
}

# =====================================================================
# 2. BUSINESS LOGIC LAYER (Capa de Lógica de Negocio): Validación y control
# =====================================================================
def clasificar_residuo(imagen):
    """
    Valida los datos y ejecuta el flujo.
    """
    if imagen is None:
        print("✗ [Business Layer] Intento de clasificación sin imagen.")
        return {"Error": "Por favor, suban una imagen válida."}, "Subí una imagen para ver la recomendación."

    print("[Business Layer] Imagen recibida. Ejecutando preprocesamiento e inferencia...")

    try:
        # Inferencia directa mediante DataLayer
        resultados = clasificador_modelo(imagen)

        # Formateamos y traducimos el resultado al castellano
        salida = {
            TRADUCCIONES.get(r["label"].lower(), r["label"]): float(r["score"]) # sin el lower, no tendríamos recomendaciones
            for r in resultados
        }

        # Tomamos la categoría con mayor probabilidad para buscar su recomendación
        categoria_top = max(salida, key=salida.get)
        recomendacion = RECOMENDACIONES.get(
            categoria_top,
            "No hay una recomendación específica para esta categoría todavía."
        )

        print(f"✓ [Business Layer] Clasificación completada: {categoria_top}.")
        return salida, recomendacion

    except Exception as error:
        print(f"✗ [Business Layer] Falla en la inferencia: {error}")
        return {"Error de inferencia": str(error)}, "Ocurrió un error al generar la recomendación."

# =====================================================================
# 3. PRESENTATION LAYER (Capa de Presentación): Interfaz de Gradio
# =====================================================================
print("[Presentation Layer] Construyendo interfaz web...")

demo = gr.Interface(
    fn=clasificar_residuo,
    inputs=gr.Image(type="pil", label="Subí una foto del residuo"),
    outputs=[
        gr.Label(num_top_classes=5, label="Predicción"),
        gr.Textbox(label="Qué hacer con este residuo", lines=3),
    ],
    title="Clasificador de Residuos",
    description=(
        "Subí una imagen y el modelo va a predecir a qué categoría de residuo pertenece "
        "(Pila/Batería, Orgánico, Cartón, Vidrio, Metal, Papel, Plástico, Ropa, Calzado, Basura general) "
        "y te va a sugerir cómo manejarlo."
    ),
)

print("✓ [Presentation Layer] UI inicializada correctamente.")

# =====================================================================
# 4. LANZAMIENTO DE LA APLICACIÓN
# =====================================================================
if __name__ == "__main__":
    print("✦ Iniciando el servidor de Gradio...")
    demo.launch()
