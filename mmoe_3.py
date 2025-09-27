import torch
from torch import nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import joblib

# ==============================================================================
# 2. 定义“零件”：被主模型依赖的工具类
# ==============================================================================

class EmbeddingLayer(nn.Module):
    """
    创建所有稀疏特征的Embedding层。
    """
    def __init__(self, enc_dict, embedding_dim):
        super(EmbeddingLayer, self).__init__()
        self.enc_dict = enc_dict
        self.embedding_dim = embedding_dim
        
        self.embedding_layers = nn.ModuleDict()
        for feat, vocab_size in self.enc_dict.items():
            self.embedding_layers[feat] = nn.Embedding(vocab_size, embedding_dim)

    def forward(self, data_dict):
        embeddings = []
        for feat, layer in self.embedding_layers.items():
            if feat in data_dict:
                embeddings.append(layer(data_dict[feat]))
        return embeddings

class MLP_Layer(nn.Module):
    """
    一个标准的多层感知机 (Multi-Layer Perceptron)。
    """
    def __init__(self, input_dim, output_dim=None, hidden_units=[], dropout_rates=0.0):
        super(MLP_Layer, self).__init__()
        layers = []
        dims = [input_dim] + hidden_units
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.ReLU())
            if isinstance(dropout_rates, list):
                layers.append(nn.Dropout(dropout_rates[i]))
            else:
                layers.append(nn.Dropout(dropout_rates))
        
        if output_dim is not None:
            if len(hidden_units) > 0:
                layers.append(nn.Linear(hidden_units[-1], output_dim))
            else:
                layers.append(nn.Linear(input_dim, output_dim))

        self.mlp = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.mlp(x)

# ==============================================================================
# 3. 定义“组装机”：主模型 MMOE
# ==============================================================================

class MMOE(nn.Module):
    def __init__(self,
                 enc_dict,
                 dense_feature_cols,
                 num_task=1,
                 n_expert=3,
                 embedding_dim=40,
                 mmoe_hidden_dim=128,
                 hidden_dim=[128, 64],
                 dropouts=[0.2, 0.2]):
        super(MMOE, self).__init__()
        
        # 定义层和参数
        self.embedding_layer = EmbeddingLayer(enc_dict=enc_dict, embedding_dim=embedding_dim)
        
        num_sparse_fea, num_dense_fea = len(enc_dict), len(dense_feature_cols)
        total_input_dim = num_sparse_fea * embedding_dim + num_dense_fea

        # Experts
        self.experts = nn.ModuleList([
            MLP_Layer(total_input_dim, mmoe_hidden_dim) for _ in range(n_expert)
        ])
        
        # Gates
        self.gates = nn.ModuleList([
            nn.Linear(total_input_dim, n_expert) for _ in range(num_task)
        ])

        # Towers
        self.towers = nn.ModuleList([
            MLP_Layer(mmoe_hidden_dim, output_dim=hidden_dim[-1], hidden_units=hidden_dim[:-1], dropout_rates=dropouts) 
            for _ in range(num_task)
        ])
        
        # 每个任务的输出层
        self.predict_layers = nn.ModuleList([
            nn.Linear(hidden_dim[-1], 1) for _ in range(num_task)
        ])

    def forward(self, data):
        # 特征处理
        feature_embedding = self.embedding_layer(data)
        sparse_input = torch.cat(feature_embedding, dim=1)
        dense_input = data['dense_features']
        hidden = torch.cat([sparse_input, dense_input], dim=1)

        # 专家网络输出
        experts_out = torch.stack([expert(hidden) for expert in self.experts], dim=1)

        # 门控网络输出
        gates_out = [torch.softmax(gate(hidden), dim=-1) for gate in self.gates]

        # 加权求和
        task_inputs = []
        for gate_output in gates_out:
            expanded_gate = gate_output.unsqueeze(-1)
            weighted_expert_output = experts_out * expanded_gate
            task_input = torch.sum(weighted_expert_output, dim=1)
            task_inputs.append(task_input)

        # 任务塔
        task_outputs = []
        for i in range(len(task_inputs)):
            tower_output = self.towers[i](task_inputs[i])
            pred = torch.sigmoid(self.predict_layers[i](tower_output))
            task_outputs.append(pred)
        
        return task_outputs

# ==============================================================================
# 4. 定义数据处理类
# ==============================================================================

class RecSysDataset(Dataset):
    def __init__(self, df, sparse_cols, dense_cols, label_col):
        self.df = df
        self.sparse_cols = sparse_cols
        self.dense_cols = dense_cols
        self.label_col = label_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx].to_dict()
        sparse_vals = {col: torch.tensor(row[col], dtype=torch.long) for col in self.sparse_cols}
        dense_vals = torch.tensor([row[col] for col in self.dense_cols], dtype=torch.float32)
        label = torch.tensor(row[self.label_col], dtype=torch.float32)
        
        feature_dict = {**sparse_vals, 'dense_features': dense_vals}
        
        return feature_dict, label

# ==============================================================================
# 5. 主执行模块 (仅在直接运行时执行)
# ==============================================================================

if __name__ == '__main__':
    
    # --- 数据准备 ---
    print("--> Step 1: Preparing data...")
    data = pd.read_csv("final_features_for_mmoe.csv")

    sparse_features = ['session_id', 'item_id']
    dense_features = ['session_len', 'session_nunique_items'] + [f'embed_{i}' for i in range(64)]
    label_col = 'label'

    print("   -> Preprocessing features...")
    encoders = {}
    for feat in sparse_features:
        lbe = LabelEncoder()
        data[feat] = lbe.fit_transform(data[feat])
        encoders[feat] = lbe

    mms = MinMaxScaler(feature_range=(0, 1))
    data[dense_features] = mms.fit_transform(data[dense_features])
    enc_dict = {feat: data[feat].nunique() for feat in sparse_features}
    
    print("--> Step 1.5: Saving artifacts...")
    joblib.dump(encoders, 'label_encoders.pkl')
    joblib.dump(mms, 'min_max_scaler.pkl')
    joblib.dump(enc_dict, 'enc_dict.pkl')
    print("   ✅ Artifacts saved successfully!")

    # --- 创建 DataLoader ---
    train_size = int(0.8 * len(data))
    train_df, valid_df = data.iloc[:train_size], data.iloc[train_size:]
    train_dataset = RecSysDataset(train_df, sparse_features, dense_features, label_col)
    valid_dataset = RecSysDataset(valid_df, sparse_features, dense_features, label_col)
    train_loader = DataLoader(train_dataset, batch_size=1024, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=1024, shuffle=False)
    print("Data preparation finished.")

    # --- 超参数和设备设置 ---
    print("--> Step 2: Setting up hyperparameters...")
    LEARNING_RATE = 0.001
    EPOCHS = 5
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # --- 模型、优化器、损失函数实例化 ---
    print("--> Step 3: Instantiating model, optimizer, and loss function...")
    model = MMOE(
        num_task=1,
        n_expert=3,
        embedding_dim=16,
        mmoe_hidden_dim=64,
        hidden_dim=[64],
        dropouts=[0.2],
        enc_dict=enc_dict,
        dense_feature_cols=dense_features
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.BCELoss()

    # --- 训练和验证循环 ---
    print("--> Step 4: Starting training loop...")
    for epoch in range(EPOCHS):
        model.train()
        total_train_loss = 0
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Training]")
        for features, labels in train_bar:
            features = {k: v.to(DEVICE) for k, v in features.items()}
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            task_outputs = model(features)
            loss = criterion(task_outputs[0].squeeze(-1), labels)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for features, labels in tqdm(valid_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Validation]"):
                features = {k: v.to(DEVICE) for k, v in features.items()}
                task_outputs = model(features)
                all_preds.extend(task_outputs[0].squeeze().cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        auc = roc_auc_score(all_labels, all_preds)
        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {avg_train_loss:.4f}, Validation AUC: {auc:.4f}")

    # --- 模型保存 ---
    print("Training finished.")
    torch.save(model.state_dict(), "mmoe_model.pth")
    print("Model saved to mmoe_model.pth")