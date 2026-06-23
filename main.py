# This is a new Python file
import re 
import unicodedata
import os
import joblib

#fazendo uma função para limpar o texto
def limpeza_do_texto (texto):
#garantindo que o texto não esteja vazio/nulo
    if not isinstance(texto, str):
        return ""
    
    #separa o caractere de acentuação e depois remove os acentos
    texto_sem_acento = ''.join(c for c in unicodedata.normalize('NFKD', texto) if unicodedata.category(c) != 'Mn') 

    #convertendo para minúsculo
    texto_limpo = texto_sem_acento.lower()

    #mantendo apenas letras e espaços
    texto_limpo = re.sub(r'[^a-z\s]', '', texto_limpo)

    #removendo espaços extras
    texto_limpo = re.sub(r'\s+', ' ', texto_limpo).strip()

    return texto_limpo

DIR_ATUAL = os.path.dirname(os.path.abspath(__file__))



caminho_modelo = os.path.join(DIR_ATUAL, "melhor_modelo.joblib")
modelo = joblib.load(caminho_modelo)
# Hotfix para compatibilidade de versão do scikit-learn:
modelo.steps[-1][1].multi_class = "auto"

def prever_sentimentos(texto):
    texto_limpo = limpeza_do_texto(texto)
    previsao = modelo.predict([texto_limpo])[0]
    probs = modelo.predict_proba([texto_limpo])[0]
    confianca = probs[previsao]

    mapeamento = {0: "negativo", 1: "positivo"}
    sentimento_text = mapeamento[previsao]
    return {
        "sentimento": sentimento_text,
        "confianca": float(confianca)
    } 

if __name__ == "__main__":
    teste_ruim = "A bateria do celular não dura nada, péssima compra"
    teste_bom = "Chegou super rápido, excelente produto!"
    
    print(f"Teste 1: '{teste_ruim}' -> Sentimento: {prever_sentimentos(teste_ruim)}")
    print(f"Teste 2: '{teste_bom}' -> Sentimento: {prever_sentimentos(teste_bom)}")

