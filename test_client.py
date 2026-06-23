from fastapi.testclient import TestClient
from app import app

cliente = TestClient(app)

def test_health():
    resposta = cliente.get("/health") 
    assert resposta.status_code == 200

    assert resposta.json() == {"status":"ok"}

def test_predict():
    resposta = cliente.post("/predict", json={"texto":"excelente produto"})
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["sentimento"] == "positivo"
    assert "confianca" in dados 

def test_predict_muito_curto():
    # Enviando texto menor que 3 caracteres para forçar erro 422
    resposta = cliente.post("/predict", json={"texto": "Oi"})
    assert resposta.status_code == 422

def test_model_info():
    resposta = cliente.get("/model-info")
    assert resposta.status_code == 200
    dados = resposta.json()
    assert dados["Nome"] == "Classificador de Sentimentos de E-commerce"
    assert dados["Versão"] == "2.0" # ou "Versão", dependendo de como você definiu a chave
