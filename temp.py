import matplotlib.pyplot as plt
import json
import os

# 1. Setup Directories
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

model_names = ['gru', 'lstm', 'transformer']
colors = {'gru': 'blue', 'lstm': 'red', 'transformer': 'green'}

plt.figure(figsize=(12, 7))

# 2. Loop and Plot
for model_name in model_names:
    file_path = os.path.join(RESULTS_DIR, f"{model_name}_history.json")
    
    if not os.path.exists(file_path):
        print(f"Warning: {file_path} not found. Skipping.")
        continue
        
    with open(file_path, 'r') as f:
        history = json.load(f)
    
    epochs = range(1, len(history['loss']) + 1)
    color = colors[model_name]

    # Plot Training Loss
    plt.plot(epochs, history['loss'], 
             label=f'{model_name.upper()} Train', 
             color=color, linestyle='-')
    
    # Plot Validation Loss
    plt.plot(epochs, history['val_loss'], 
             label=f'{model_name.upper()} Val', 
             color=color, linestyle='--')

# 3. Formatting
plt.title('Training vs. Validation Loss Comparison (All Models)', fontsize=14)
plt.xlabel('Epoch', fontsize=12)
plt.ylabel('Loss (Combined)', fontsize=12)
plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1)) # Moved legend outside slightly
plt.grid(True, alpha=0.3)
plt.tight_layout()

# 4. Save and Show
save_path = os.path.join(PLOTS_DIR, 'training_curves.png')
plt.savefig(save_path, dpi=300)
print(f"Plot saved successfully to: {save_path}")

plt.show()