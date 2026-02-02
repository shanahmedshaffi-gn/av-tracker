import os
import numpy as np

folder = "../data/embeddings"

print(f"🔍 Auditando pasta: {folder}")
print("-" * 40)

if not os.path.exists(folder):
    print("❌ Pasta não existe!")
    exit()

files = [f for f in os.listdir(folder) if f.endswith('.npy')]

if not files:
    print("❌ A pasta está vazia!")

for f in files:
    path = os.path.join(folder, f)
    try:
        data = np.load(path)
        
        # Verifica se é um vetor válido
        is_zeros = not np.any(data)
        shape_ok = data.shape == (192,)
        
        status = "✅ OK"
        if is_zeros: status = "❌ ZERADO (Erro)"
        if not shape_ok: status = f"❌ SHAPE ERRADO {data.shape}"
        
        print(f"{f:<30} | {status} | Max Value: {data.max():.4f}")
        
    except Exception as e:
        print(f"{f:<30} | ❌ CORROMPIDO ({e})")