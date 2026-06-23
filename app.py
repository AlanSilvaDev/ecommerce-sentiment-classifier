from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from main import prever_sentimentos

app = FastAPI(
    title="Classificador de Sentimentos",
    description="API e Interface para classificação de sentimentos em avaliações de e-commerce",
    version="2.0"
)

# Adiciona suporte a CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent

class EntradaTexto(BaseModel):
    texto: str = Field(min_length=3, max_length=1000)

@app.post("/predict")
def predict(entrada: EntradaTexto):
    return prever_sentimentos(entrada.texto)

@app.get("/health")
def healt():
    return {'status': 'ok'}

@app.get("/model-info")
def model_info():
    return {
        "Nome": "Classificador de Sentimentos de E-commerce",
        "Versão": "2.0",
        "Algoritmo": "TF-IDF + StandardScaler + Regressão Logística (Pipeline)",
        "acuracia_treino": 0.8115,
        "classes": ["negativo", "positivo"]
    }
    
@app.get("/", response_class=HTMLResponse)
def home():
    with open(BASE_DIR / "index.html", "r", encoding="utf-8") as f:
        return f.read()

