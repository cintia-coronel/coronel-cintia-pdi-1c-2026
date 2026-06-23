---
title: Clasificador de Residuos
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 4.44.1
app_file: app.py
pinned: false
license: mit
---

# Clasificador de Residuos

Aplicación web que clasifica imágenes de residuos en distintas categorías (plástico, papel, metal, vidrio, cartón, etc.) 
utilizando un modelo Vision Transformer (ViT) preentrenado, disponible en Hugging Face Hub: https://huggingface.co/watersplash/waste-classification.

## Cómo funciona

1. El usuario sube una foto de un residuo.
2. El modelo predice a qué categoría pertenece.
3. Se muestran las 5 categorías más probables junto con su nivel de confianza, y una recomendación de manejo para la categoría más probable.

## Arquitectura

La aplicación sigue un patrón de 3 capas:

- **Data Layer**: carga del modelo preentrenado (`pipeline` de `transformers`).
- **Business Logic Layer**: valida la imagen recibida, ejecuta la inferencia y formatea/traduce el resultado.
- **Presentation Layer**: interfaz interactiva construida con Gradio (`gr.Interface`).
