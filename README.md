Markdown
# Avaliação A1 - IA UNITINS: Pipeline de Treinamento com MLflow e PyTorch

## Sobre o Projeto
Este repositório contém a resolução da Avaliação 1 (A1) da disciplina de Inteligência Artificial da UNITINS. O projeto implementa um pipeline de treinamento observável para uma rede neural (MLP) utilizando PyTorch, com rastreabilidade total garantida pelo MLflow.

O objetivo do modelo é prever a aprovação de alunos em Matemática (classificação binária) utilizando o dataset **Student Performance** (UCI Machine Learning Repository). O pipeline previne vazamento de dados (*data leakage*) realizando o particionamento antes da transformação dos dados.

Foram documentados dois experimentos para fins de comparação:
- **Experimento A (Referência):** Modelo base sem regularização, sujeito a *overfitting*.
- **Experimento B (Com Regularização):** Inserção de Dropout (0.3) e Weight Decay (L2), comprovando a melhoria na capacidade de generalização da rede.

## Requisitos
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) ou Anaconda instalado.

## Como Executar

**1. Clone o repositório e acesse a pasta**
```bash
git clone [https://github.com/Blilinho/AI_UNITINS_A1.git](https://github.com/Blilinho/AI_UNITINS_A1.git)
cd AI_UNITINS_A1
2. Crie e ative o ambiente virtual isolado

Bash
conda create -n ia_ambiente python=3.11 -y
conda activate ia_ambiente
3. Instale as bibliotecas necessárias

Bash
pip install torch torchvision torchaudio scikit-learn matplotlib seaborn mlflow pandas numpy
4. Execute o treinamento

Bash
python prova_a1.py
O script executará automaticamente os dois experimentos, calculará as métricas e salvará os artefatos (como a Matriz de Confusão e os pesos do modelo).

5. Visualize os resultados no painel de observabilidade

Bash
mlflow ui
Abra o link gerado (geralmente http://127.0.0.1:5000) no navegador para comparar as métricas das rodadas e baixar os artefatos.
