# 📊 Classificador de Sentimentos de E-Commerce v2.0

Este projeto é uma aplicação de Machine Learning que classifica sentimentos (positivo/negativo) em avaliações de produtos de e-commerce. A solução engloba desde a modelagem e persistência do modelo de inteligência artificial até uma API robusta com FastAPI e uma interface web interativa de alta fidelidade (glassmorphism).

---

## 💡 Propósito e Evolução do Projeto

Este é um **projeto de aprendizado** que representa uma evolução direta de um estudo anterior.

- **Evolução do Projeto 1.0:** Na primeira etapa deste estudo, o trabalho consistia apenas em importar, limpar a base de dados e treinar uma rede/modelo básico em um Jupyter Notebook, sem qualquer interação externa.
- **A Versão 2.0 (Atual):** Evoluiu para uma solução completa (de ponta a ponta), onde o modelo treinado foi encapsulado em uma API FastAPI estruturada, testada de forma automatizada e acoplada a uma interface web real.
- **Interface Gerada por IA (com minha validação):** O design e a estrutura inicial da interface (HTML/CSS) foram gerados com auxílio de IA, mas validados, ajustados, revisados e conectados à API manualmente por mim.
- **Fundação para o Futuro:** O projeto serve como base e arquitetura de referência. À medida que eu aprender novas técnicas, poderei reaproveitar esta estrutura de deploy para modelos maiores e mais complexos.

---


## 🚀 Funcionalidades

- **Inteligência Artificial:** Pipeline composto por vetorização TF-IDF, normalização (`StandardScaler`) e classificação com `Regressão Logística`.
- **API de Alta Performance (FastAPI):**
  - Endpoint `/predict` para predições em tempo real.
  - Endpoint `/model-info` para exibição de metadados do modelo.
  - Validação de dados de entrada via `Pydantic` (bloqueia textos menores que 3 ou maiores que 1000 caracteres).
- **Interface Web Premium:** Single Page Application (SPA) construída com HTML5, CSS3 e Javascript puro, contendo efeitos modernos de glassmorphism, barra de confiança dinâmica e feedbacks interativos de validação/erro.
- **Suíte de Testes Automatizados:** Testes automatizados implementados com `pytest` e `TestClient` cobrindo sucesso, erros de validação e endpoints informativos.

---

## 📁 Estrutura do Projeto

```text
├── 2.0 clas/                  # Código-fonte da aplicação
│   ├── main.py                # Lógica principal de pré-processamento e inferência do modelo
│   ├── app.py                 # Rotas da API FastAPI e suporte a CORS
│   ├── index.html             # Interface web interativa
│   ├── test_client.py         # Testes automatizados com pytest
│   └── pipeline_classificador_sentimentos.pkl # Pipeline do modelo treinado persistido
│
├── requirements.txt           # Dependências do projeto
```

---

## 🛠️ Pré-requisitos e Instalação

### 1. Clonar o repositório
```bash
git clone https://github.com/alanSilvaDev/Classifica-o-de-Sentimentos-de-um-E-commerce.git
cd Classifica-o-de-Sentimentos-de-um-E-commerce
```

### 2. Configurar o ambiente virtual (opcional, mas recomendado)
```bash
python -m venv venv
# No Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# No Linux/macOS:
source venv/bin/activate
```

### 3. Instalar as dependências
```bash
pip install -r requirements.txt
```
> *Nota: Se você não tiver o arquivo `requirements.txt` criado, as dependências principais são: `fastapi`, `uvicorn`, `pydantic`, `scikit-learn`, `pytest`, `requests` e `pandas`.*

---

## 🖥️ Como Executar o Projeto

### Iniciar o servidor da API
Para rodar a aplicação localmente e habilitar o recarregamento automático (reload) a partir da pasta raiz do projeto:

```bash
python -m uvicorn --app-dir "2.0 clas" app:app --reload --port 8000
```

Assim que o servidor iniciar:
- **Interface Web:** Acesse **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** para testar as classificações de forma visual.
- **Documentação Interativa (Swagger):** Acesse **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)** para inspecionar e interagir com os endpoints individualmente.

---

## 🧪 Como Executar os Testes Automatizados

Garantir a integridade da aplicação é fundamental. Execute os testes unitários com o `pytest`:

```bash
python -m pytest "2.0 clas/test_client.py"
```

---

## 🔌 API Endpoints

### 1. `POST /predict`
Recebe um texto e retorna o sentimento classificado com a probabilidade (confiança).

- **Corpo da Requisição (JSON):**
```json
{
  "texto": "Excelente produto, super recomendo!"
}
```
- **Resposta de Sucesso (200 OK):**
```json
{
  "sentimento": "positivo",
  "confianca": 0.895
}
```
- **Resposta de Erro de Validação (422 Unprocessable Entity):**
*Ocorre caso o texto possua menos de 3 caracteres.*

### 2. `GET /model-info`
Retorna as especificações do classificador.

- **Resposta de Sucesso (200 OK):**
```json
{
  "Nome": "Classificador de Sentimentos de E-commerce",
  "Versão": "2.0",
  "Algoritmo": "TF-IDF + StandardScaler + Regressão Logística (Pipeline)",
  "acuracia_treino": 0.8115,
  "classes": ["negativo", "positivo"]
}
```

### 3. `GET /health`
Verifica se a API está online.
- **Resposta de Sucesso (200 OK):** `{"status": "ok"}`

---

## 📈 Detalhes, Simplicidade e Limitações do Modelo

O modelo treinado possui uma arquitetura clássica e intencionalmente simples (TF-IDF + StandardScaler + Regressão Logística). Ele obteve uma **acurácia de 81.15%** nos dados de validação.

### ⚠️ Limitações Conhecidas:
Como se trata de um modelo linear clássico baseado em frequências de palavras (saco de palavras):
- **Sarcasmo e Ironia:** Ele tem dificuldade em classificar textos onde o sentimento real é oposto ao significado literal das palavras (ex: *"Excelente produto, quebrou na primeira hora"* pode ser classificado incorretamente como positivo devido à palavra *"Excelente"*).
- **Semântica e Contexto:** Por não utilizar redes neurais profundas de Processamento de Linguagem Natural (PLN) ou modelos do estado da arte (como Transformers), ele não entende a ordem complexa e a semântica de frases mais ambíguas.
- **Dataset Reduzido:** Foi treinado em um dataset simples de avaliações de e-commerce, o que significa que abreviações ou gírias fora do conjunto de treinamento podem causar erros de predição.

### 🧠 Como Funciona o Pipeline Atual:
- **Pré-processamento:** Remoção de stopwords, limpeza de pontuações e vetorização com TF-IDF.
- **Normalização:** Ajuste de escala dos recursos numéricos.
- **Classificação:** Regressão Logística, ideal para classificação binária rápida e altamente explicável.
- **Persistência:** Empacotamento de todas as etapas de processamento e do modelo em um único arquivo `.pkl` para garantir consistência e evitar vazamento de dados (*data leakage*).

---

## 🎓 Autor

Desenvolvido com dedicação para demonstração de portfólio.
- **GitHub:** [alanSilvaDev](https://github.com/alanSilvaDev)
