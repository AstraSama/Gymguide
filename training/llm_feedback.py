import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()  # Carrega as variáveis do .env

api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    print("❌ Erro: GOOGLE_API_KEY não encontrada no .env")
else:
    print("🔐 GOOGLE_API_KEY carregada com sucesso.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-pro")


def gerar_feedback(dados):
    print("🎯 Função gerar_feedback() foi chamada!")
    
    prompt = f"""
Você é um treinador de academia virtual. O usuário executou flexões no modo "{dados['mode']}". 
Com base nos dados abaixo, forneça um feedback técnico:

- Classificação final: {dados['classificacao']}
- Repetições realizadas: {dados['reps']}
- Ângulo médio do cotovelo: {dados['mean']:.2f} graus
- Amplitude do movimento (variação do ângulo): {dados['amplitude']:.2f} graus

**Objetivo:**
- Se a classificação for "Ruim", identifique os principais erros (pouca amplitude, ângulo inadequado etc) e explique como corrigir.
- Se a classificação for "Bom", parabenize e destaque o que foi feito corretamente.
- Sempre dê 1 ou 2 dicas práticas para melhorar a execução.

Dado o modo {dados['mode']}, ajuste as dicas conforme o ângulo (modo lateral) ou posição do tronco/quadril (modo frontal).
"""

    try:
        response = model.generate_content(prompt)
        print("✅ Resposta recebida do Gemini.")
        return response.text.strip()
    except Exception as e:
        print(f"❌ Erro ao gerar feedback com Gemini: {e}")
        return "⚠️ Erro ao gerar feedback."

