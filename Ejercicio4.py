# Versión optimizada para CPU con manejo de modelos grandes
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import time
import psutil
import gc
from tabulate import tabulate

class ChatbotOptimizadoCPU:
    def __init__(self):
        self.estilos = {
            "titulo": "\033[1;36m",
            "subtitulo": "\033[1;34m",
            "exito": "\033[1;32m",
            "advertencia": "\033[1;33m",
            "error": "\033[1;31m",
            "reset": "\033[0m"
        }
        self.dispositivo = "cpu"
        self.modelos_optimizados = {
            "1": ("distilgpt2", "GPT-2 Pequeño (Inglés)", 82),
            "2": ("DeepESP/gpt2-spanish", "GPT-2 Español Pequeño", 124),
            "3": ("bertin-project/bertin-gpt-j-6B-alternative", "Bertin Alternativo", 512)  # Versión más ligera
        }
        self.configurar_entorno()

    def configurar_entorno(self):
        print(f"{self.estilos['titulo']}\n⚙️ Configuración del Entorno:{self.estilos['reset']}")
        print(f"- Dispositivo: {self.estilos['advertencia']}{self.dispositivo.upper()}{self.estilos['reset']}")
        print(f"- Memoria RAM disponible: {psutil.virtual_memory().available / (1024**3):.1f} GB")

    def cargar_modelo_seguro(self, modelo_id, max_memoria_gb=2):
        modelo_info = self.modelos_optimizados.get(modelo_id, self.modelos_optimizados["1"])
        
        print(f"\n{self.estilos['subtitulo']}📦 Cargando {modelo_info[1]} (~{modelo_info[2]}MB){self.estilos['reset']}")
        
        try:
            # Configuración para CPU con control de memoria
            tokenizador = AutoTokenizer.from_pretrained(modelo_info[0])
            
            modelo = AutoModelForCausalLM.from_pretrained(
                modelo_info[0],
                torch_dtype=torch.float32,
                low_cpu_mem_usage=True,
                device_map="cpu",
                max_memory={0: f"{max_memoria_gb}GB"} if max_memoria_gb else None
            )
            
            return modelo, tokenizador
        except Exception as e:
            print(f"{self.estilos['error']}❌ Error: {e}{self.estilos['reset']}")
            print(f"{self.estilos['advertencia']}Probando con modelo más pequeño...{self.estilos['reset']}")
            return self.cargar_modelo_seguro("1")  # Fallback a modelo pequeño

    def generar_texto(self, modelo, tokenizador, texto, max_longitud=50):
        try:
            inputs = tokenizador(texto, return_tensors="pt").to(self.dispositivo)
            
            # Medición de rendimiento
            inicio = time.time()
            mem_inicio = psutil.Process().memory_info().rss
            
            with torch.no_grad():
                outputs = modelo.generate(
                    **inputs,
                    max_new_tokens=max_longitud,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizador.eos_token_id
                )
            
            texto_generado = tokenizador.decode(outputs[0], skip_special_tokens=True)
            tiempo = time.time() - inicio
            mem_usada = (psutil.Process().memory_info().rss - mem_inicio) / (1024 ** 2)
            
            return {
                "texto": texto_generado,
                "tiempo": tiempo,
                "memoria": mem_usada,
                "velocidad": outputs.shape[1] / tiempo if tiempo > 0 else 0
            }
        except Exception as e:
            print(f"{self.estilos['error']}Error en generación: {e}{self.estilos['reset']}")
            return None

    def mostrar_menu(self):
        print(f"\n{self.estilos['titulo']}🤖 CHATBOT OPTIMIZADO PARA CPU{self.estilos['reset']}")
        print(f"{self.estilos['subtitulo']}Modelos disponibles:{self.estilos['reset']}")
        for k, v in self.modelos_optimizados.items():
            print(f"{k}. {v[1]} ({v[0]})")

    def ejecutar(self):
        self.mostrar_menu()
        opcion = input("\nSeleccione modelo (1-3): ") or "1"
        texto = input("Ingrese su mensaje: ") or "La inteligencia artificial es"
        
        modelo, tokenizador = self.cargar_modelo_seguro(opcion)
        
        if modelo:
            resultado = self.generar_texto(modelo, tokenizador, texto)
            
            if resultado:
                print(f"\n{self.estilos['exito']}💡 Respuesta:{self.estilos['reset']}")
                print(resultado["texto"])
                
                print(f"\n{self.estilos['subtitulo']}📊 Métricas:{self.estilos['reset']}")
                print(f"- Tiempo: {resultado['tiempo']:.2f}s")
                print(f"- Memoria usada: {resultado['memoria']:.2f}MB")
                print(f"- Velocidad: {resultado['velocidad']:.2f} tokens/s")
            
            # Liberar memoria
            del modelo
            gc.collect()

if __name__ == "__main__":
    chatbot = ChatbotOptimizadoCPU()
    chatbot.ejecutar()
