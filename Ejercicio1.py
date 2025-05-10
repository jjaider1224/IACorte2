import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os

# Configurar caché local de modelos en Colab
os.environ['TRANSFORMERS_CACHE'] = '/content/transformers_cache'
os.makedirs(os.environ['TRANSFORMERS_CACHE'], exist_ok=True)

def cargar_modelo(nombre_modelo):
    """
    Carga un modelo preentrenado y su tokenizador.
    """
    tokenizador = AutoTokenizer.from_pretrained(nombre_modelo)
    modelo = AutoModelForCausalLM.from_pretrained(nombre_modelo)

    if torch.cuda.is_available():
        modelo.half()

    modelo.eval()
    return modelo, tokenizador

def verificar_dispositivo():
    """
    Verifica el dispositivo disponible.
    """
    if torch.cuda.is_available():
        dispositivo = torch.device("cuda")
        print(f"GPU disponible: {torch.cuda.get_device_name(0)}")
    else:
        dispositivo = torch.device("cpu")
        print("GPU no disponible, usando CPU.")
    return dispositivo

def main():
    dispositivo = verificar_dispositivo()
    print(f"Utilizando dispositivo: {dispositivo}")

    nombre_modelo = "distilgpt2"
    modelo, tokenizador = cargar_modelo(nombre_modelo)
    modelo.to(dispositivo)

    # Entrada de prueba
    entrada = "Barcelona F.C."
    entradas = tokenizador.encode(entrada, return_tensors="pt").to(dispositivo)

    # Configuración de generación
    pad_token_id = tokenizador.eos_token_id
    salida = modelo.generate(
        entradas,
        max_length=120,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        repetition_penalty=1.1,
        pad_token_id=pad_token_id
    )

    texto_generado = tokenizador.decode(salida[0], skip_special_tokens=True)
    print("Texto generado:")
    print(texto_generado)

if __name__ == "__main__":
    main()
