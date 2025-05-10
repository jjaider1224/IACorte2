import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM
import os
from huggingface_hub import login
import torch

class ChatbotPersonalizado:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.modelo = None
        self.tokenizador = None
        self.modelo_cargado = False
        
    def autenticar_huggingface(self, token=None):
        """Autenticación segura con Hugging Face"""
        try:
            if token:
                login(token=token)
                print("Autenticado en Hugging Face Hub")
                return True
            else:
                print("No se proporcionó token para Hugging Face Hub")
                return False
        except Exception as e:
            print(f"Error en autenticación: {e}")
            return False

    def cargar_modelo_predeterminado(self):
        """Carga un modelo pequeño predeterminado para pruebas"""
        try:
            print("Cargando modelo predeterminado 'distilgpt2'...")
            self.tokenizador = AutoTokenizer.from_pretrained("distilgpt2")
            self.modelo = AutoModelForCausalLM.from_pretrained(
                "distilgpt2",
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            self.modelo_cargado = True
            print("Modelo predeterminado cargado exitosamente")
            return True
        except Exception as e:
            print(f"Error al cargar modelo predeterminado: {e}")
            return False

    def cargar_modelo_personalizado(self, ruta_modelo, hf_token=None):
        """Intenta cargar un modelo personalizado con manejo robusto de errores"""
        if not ruta_modelo or ruta_modelo == "tu_modelo":
            print("No se especificó un modelo válido. Usando modelo predeterminado.")
            return self.cargar_modelo_predeterminado()
            
        try:
            print(f"Intentando cargar modelo desde: {ruta_modelo}")
            
            # Verificar si es una ruta local
            if os.path.exists(ruta_modelo):
                print("Cargando modelo local...")
                self.tokenizador = AutoTokenizer.from_pretrained(ruta_modelo)
                self.modelo = AutoModelForCausalLM.from_pretrained(
                    ruta_modelo,
                    device_map="auto",
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
                )
                self.modelo_cargado = True
                return True
            
            # Si no es local, intentar desde Hugging Face Hub
            print("Intentando cargar desde Hugging Face Hub...")
            if hf_token:
                self.autenticar_huggingface(hf_token)
                
            self.tokenizador = AutoTokenizer.from_pretrained(ruta_modelo)
            self.modelo = AutoModelForCausalLM.from_pretrained(
                ruta_modelo,
                device_map="auto",
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                token=hf_token
            )
            self.modelo_cargado = True
            return True
            
        except Exception as e:
            print(f"Error al cargar modelo personalizado: {e}")
            print("Intentando cargar modelo predeterminado como respaldo...")
            return self.cargar_modelo_predeterminado()

    def generar_respuesta(self, mensaje, historial=[]):
        """Genera respuesta con manejo de errores"""
        if not self.modelo_cargado:
            return "Error: Modelo no cargado correctamente", historial
            
        try:
            inputs = self.tokenizador(mensaje, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.modelo.generate(
                    **inputs,
                    max_new_tokens=100,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizador.eos_token_id
                )
                
            respuesta = self.tokenizador.decode(outputs[0], skip_special_tokens=True)
            historial.append((mensaje, respuesta))
            return respuesta, historial
            
        except Exception as e:
            return f"Error: {str(e)}", historial

def crear_interfaz(chatbot):
    """Interfaz simplificada pero funcional"""
    with gr.Blocks() as interfaz:
        gr.Markdown("# Chatbot de Prueba")
        
        chatbot_ui = gr.Chatbot(label="Conversación")
        mensaje = gr.Textbox(label="Tu mensaje")
        btn_enviar = gr.Button("Enviar")
        
        def responder(mensaje, historial):
            respuesta, nuevo_historial = chatbot.generar_respuesta(mensaje, historial)
            return "", nuevo_historial
            
        btn_enviar.click(
            responder,
            inputs=[mensaje, chatbot_ui],
            outputs=[mensaje, chatbot_ui]
        )
        
        mensaje.submit(
            responder,
            inputs=[mensaje, chatbot_ui],
            outputs=[mensaje, chatbot_ui]
        )
        
    return interfaz

def main():
    # Configuración - ¡ACTUALIZA ESTOS VALORES!
    MODELO_PERSONALIZADO = None  # Usa None para el modelo predeterminado o tu ruta/modelo
    HF_TOKEN = None  # Tu token de Hugging Face si es necesario
    
    chatbot = ChatbotPersonalizado()
    
    # Primero intenta cargar el modelo personalizado si se especificó
    if MODELO_PERSONALIZADO:
        if not chatbot.cargar_modelo_personalizado(MODELO_PERSONALIZADO, HF_TOKEN):
            print("No se pudo cargar el modelo personalizado. Usando predeterminado.")
    
    # Si no se especificó modelo personalizado, carga el predeterminado
    if not chatbot.modelo_cargado:
        chatbot.cargar_modelo_predeterminado()
    
    # Crear y lanzar interfaz
    interfaz = crear_interfaz(chatbot)
    interfaz.launch(share=False)

if __name__ == "__main__":
    main()
