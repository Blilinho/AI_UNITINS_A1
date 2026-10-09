import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Carregar os Dados
# Lê o arquivo separando as colunas corretamente por ponto e vírgula
df = pd.read_csv('student-mat.csv', sep=';')

# Criar a variável alvo binária: Aprovado (G3 >= 10) vira 1, Reprovado vira 0
df['aprovado'] = (df['G3'] >= 10).astype(int)

# Remover as notas (G1, G2, G3) para que a rede tente prever APENAS com base no perfil do aluno
X = df.drop(columns=['G1', 'G2', 'G3', 'aprovado'])
y = df['aprovado'].values

# 2. Pré-processamento CORRETO (Split ANTES do Scaler para evitar vazamento de dados)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Separar colunas numéricas e categóricas (texto)
num_cols = X.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X.select_dtypes(include=['object']).columns

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols)
    ])

# Ajustar o processador APENAS no treino e transformar os outros
X_train_processed = preprocessor.fit_transform(X_train)
X_val_processed = preprocessor.transform(X_val)
X_test_processed = preprocessor.transform(X_test)

# Converter tudo para Tensores do PyTorch
X_train_t = torch.tensor(X_train_processed, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_val_t = torch.tensor(X_val_processed, dtype=torch.float32)
y_val_t = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
X_test_t = torch.tensor(X_test_processed, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 3. Definir o Modelo Neural (MLP)
class StudentMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout_rate):
        super(StudentMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, int(hidden_dim/2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(int(hidden_dim/2), 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.net(x)

# 4. Pipeline de Automação com MLflow
def treinar_modelo(nome_experimento, hidden_dim, lr, dropout, weight_decay, epochs=100):
    mlflow.set_experiment(nome_experimento)
    
    # Inicia o rastreio (Tracking) desta Run
    with mlflow.start_run(run_name=f"Taxa_{lr}_Dropout_{dropout}_L2_{weight_decay}"):
        
        # Salva a configuração usada
        mlflow.log_params({
            "hidden_dim": hidden_dim,
            "learning_rate": lr,
            "dropout": dropout,
            "weight_decay": weight_decay,
            "epochs": epochs
        })
        
        input_dim = X_train_t.shape[1]
        model = StudentMLP(input_dim, hidden_dim, dropout)
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        print("Treinando...")
        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_train_t)
            loss = criterion(y_pred, y_train_t)
            loss.backward()
            optimizer.step()
            
            # Validação
            model.eval()
            with torch.no_grad():
                y_val_pred = model(X_val_t)
                val_loss = criterion(y_val_pred, y_val_t)
                val_preds_labels = (y_val_pred >= 0.5).float()
                val_acc = accuracy_score(y_val_t.numpy(), val_preds_labels.numpy())
            
            # A cada 10 épocas, salva as métricas no MLflow para gerar a curva
            if epoch % 10 == 0:
                mlflow.log_metrics({"train_loss": loss.item(), "val_loss": val_loss.item(), "val_acc": val_acc}, step=epoch)
                
        # Gerar evidências finais (F1 e Matriz de Confusão)
        val_f1 = f1_score(y_val_t.numpy(), val_preds_labels.numpy())
        mlflow.log_metric("val_f1_final", val_f1)
        
        # Salva o gráfico da Matriz como um Artefato
        cm = confusion_matrix(y_val_t.numpy(), val_preds_labels.numpy())
        plt.figure(figsize=(5,4))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.xlabel('Previsto')
        plt.ylabel('Real')
        plt.title('Matriz de Confusão')
        plt.savefig("matriz_confusao.png")
        mlflow.log_artifact("matriz_confusao.png")
        plt.close()
        
        # Salva o arquivo do modelo na nuvem local
        mlflow.pytorch.log_model(model, "modelo_final", input_example=X_train_t.numpy()) 
        
        print(f"-> Run Concluída! Acurácia Final: {val_acc:.4f} | F1 Score: {val_f1:.4f}\n")

# 5. EXECUÇÃO DOS DOIS EXPERIMENTOS
# Pergunta objetiva 1: Como o modelo se sai em sua forma base (sem regularização)?
print("Iniciando Experimento A (Referência)...")
treinar_modelo("Prevendo_Aprovacao_Alunos", hidden_dim=64, lr=0.01, dropout=0.0, weight_decay=0.0, epochs=150)

# Pergunta objetiva 2: Adicionar Regularização (Dropout e L2) melhora a capacidade de generalização?
print("Iniciando Experimento B (Com Regularização)...")
treinar_modelo("Prevendo_Aprovacao_Alunos", hidden_dim=64, lr=0.01, dropout=0.3, weight_decay=1e-4, epochs=150)