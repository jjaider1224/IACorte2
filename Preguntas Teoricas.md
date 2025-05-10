## Preguntas Teóricas

### ¿Cuáles son las diferencias fundamentales entre los modelos encoder-only, decoder-only y encoder-decoder en el contexto de los chatbots conversacionales? Explique qué tipo de modelo sería más adecuado para cada caso de uso y por qué.

- **Encoder-only**: Utilizan solo la parte de codificación de un modelo transformer (como BERT). Son ideales para tareas de comprensión del lenguaje, como clasificación, análisis de sentimientos o búsqueda semántica, pero no generan texto.  
  **Uso adecuado**: sistemas que necesitan entender el lenguaje pero no producirlo.

- **Decoder-only**: Usan solo la parte de decodificación (como GPT). Están optimizados para tareas de generación de texto autoregresiva, como respuestas conversacionales o redacción creativa.  
  **Uso adecuado**: chatbots generativos como asistentes virtuales.

- **Encoder-decoder**: Combinan ambas partes (como T5 o BART). El encoder interpreta la entrada y el decoder genera la salida. Son útiles para tareas que requieren comprensión y generación, como traducción o resumen.  
  **Uso adecuado**: sistemas donde la entrada requiere procesamiento complejo antes de generar una respuesta precisa.

### Explique el concepto de "temperatura" en la generación de texto con LLMs. ¿Cómo afecta al comportamiento del chatbot y qué consideraciones debemos tener al ajustar este parámetro para diferentes aplicaciones?

La **temperatura** es un parámetro que controla la aleatoriedad en la generación de texto. Valores bajos (cerca de 0) hacen que el modelo sea más **determinista y conservador**, eligiendo las palabras más probables. Valores altos (hasta 1 o más) lo hacen **más creativo e impredecible**.

- **Baja temperatura (~0.2–0.5)**: útil para respuestas técnicas, precisas o donde se requiere consistencia.
- **Alta temperatura (~0.7–1.0)**: adecuada para tareas creativas o exploratorias, como redacción artística o brainstorming.

**Consideraciones**: ajustar según la tarea. Demasiada creatividad en contextos técnicos puede inducir errores; demasiada rigidez en contextos creativos puede limitar la utilidad.

### Describa las técnicas principales para reducir el problema de "alucinaciones" en chatbots basados en LLMs. ¿Qué estrategias podemos implementar a nivel de inferencia y a nivel de prompt engineering para mejorar la precisión factual de las respuestas?

**A nivel de inferencia**:
- **RAG (Retrieval-Augmented Generation)**: incorporar un motor de búsqueda para enriquecer al modelo con información actual o específica.
- **Post-verificación**: usar modelos auxiliares o reglas para validar hechos antes de mostrar la respuesta.
- **Ensembles o verificación cruzada**: combinar múltiples respuestas para evaluar consistencia.

**A nivel de prompt engineering**:
- **Instrucciones explícitas**: guiar al modelo a ser preciso, citar fuentes o limitarse a hechos conocidos.
- **Contexto relevante**: incluir información precisa y estructurada en el prompt.
- **Chain-of-thought prompting**: inducir al modelo a razonar paso a paso antes de responder.

Estas técnicas ayudan a mejorar la precisión y confianza en la generación de respuestas.
