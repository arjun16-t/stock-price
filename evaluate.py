import numpy as np
import json
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    accuracy_score, f1_score, confusion_matrix,
    ConfusionMatrixDisplay
)
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

MODELS_DIR  = 'saved_models'
RESULTS_DIR = 'results'
PLOTS_DIR = os.path.join(RESULTS_DIR, 'plots')
os.makedirs(PLOTS_DIR, exist_ok=True)

# ── Metric Helpers ────────────────────────────────────────────────────────────

def evaluate_price(y_true_return, y_pred_return, last_closes) -> dict:
    """
    Compute price regression metrics.
    Converts % return predictions back to rupee prices first.
    """
    actual_prices = last_closes * (1 + y_true_return)
    pred_prices = last_closes * (1 + y_pred_return)

    mae = mean_absolute_error(actual_prices, pred_prices)
    mse = mean_squared_error(actual_prices, pred_prices)
    mape = np.mean(np.abs((actual_prices - pred_prices) / actual_prices)) * 100
    r2 = 1 - (np.sum((actual_prices - pred_prices)**2) / np.sum((actual_prices - np.mean(actual_prices))**2))

    return {
        "mae": mae,
        "rmse": np.sqrt(mse),
        "mape": mape,
        "r2": r2
    }


def evaluate_direction(y_true, y_pred_prob):
    """
    Compute direction classification metrics.
    """
    y_pred = (y_pred_prob > 0.5).astype(int)
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    conf_matrix = confusion_matrix(y_true, y_pred)

    return {
        "accuracy": accuracy,
        "f1": f1,
        "conf_matrix": conf_matrix
    }


def baseline_metrics(y_true_return, last_closes):
    """
    Naive baseline: predict tomorrow's close = today's close (0% return).
    """
    y_pred_return = np.zeros_like(y_true_return)
    price = evaluate_price(y_true_return, y_pred_return, last_closes)
    for metrics in price:
        print(f'{metrics}: {price[metrics]}')
    
    return price


# ── Plot Helpers ──────────────────────────────────────────────────────────────

def plot_training_curves(model_names: list):
    """
    Plot train vs val loss curves for all models on one figure.
    """
    colors = {'gru': 'blue', 'lstm': 'red', 'transformer': 'green'}
    plt.figure(figsize=(12, 7))
    
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
    
    plt.title('Training vs. Validation Loss Comparison')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss (Combined)', fontsize=12)
    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_path = os.path.join(PLOTS_DIR, 'training_curves.png')
    plt.savefig(save_path, dpi=300)
    print(f"Plot saved successfully to: {save_path}")

    plt.show()


def plot_predictions(y_true_prices, y_pred_prices, model_name: str):
    """
    Plot actual vs predicted closing prices on test set.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(y_true_prices[:100], label='Actual', color='blue', linewidth=1.5)
    plt.plot(y_pred_prices[:100], label='Predicted', color='red', linestyle='--', linewidth=1.5)

    plt.title(f'Actual vs. Predicted Price: {model_name.upper()} Model', fontsize=14)
    plt.xlabel('Trading Days', fontsize=12)
    plt.ylabel('Price (₹)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    save_path = os.path.join(PLOTS_DIR, f"{model_name}_predictions.png")
    plt.savefig(save_path, dpi=300)
    print(f"Prediction plot saved to: {save_path}")

    plt.show()


def plot_confusion_matrix(cm, model_name: str):
    """
    Plot confusion matrix for direction classification.
    """
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, 
                                  display_labels=['Down', 'Up'])
    
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap=plt.cm.Blues, values_format='d')

    plt.title(f'Confusion Matrix: {model_name.upper()} Direction Prediction', fontsize=14)

    save_path = os.path.join(PLOTS_DIR, f"{model_name}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300)
    print(f"Confusion matrix saved to: {save_path}")

    plt.show()


# ── Main Evaluation ───────────────────────────────────────────────────────────

def evaluate_model(model_name: str, splits: tuple):
    """
    Full evaluation pipeline for one model.
    """
    from tensorflow.keras.models import load_model      # type: ignore

    print(f"\n--- Evaluating Model: {model_name.upper()} ---")
    (_, _, X_test,
    _, _, y_price_test,
    _, _, y_dir_test,
    _, _, last_closes_test) = splits

    model_path = f'{MODELS_DIR}/{model_name}.keras'
    model = load_model(model_path, compile=False)

    predictions = model.predict(X_test)

    if isinstance(predictions, dict):
        pred_returns = predictions['price'].flatten()
        pred_dir_prob = predictions['direction'].flatten()
    else:
        pred_returns = predictions[0].flatten()
        pred_dir_prob = predictions[1].flatten()

    y_price_test = y_price_test.flatten()
    y_dir_test = y_dir_test.flatten()

    price_metrics = evaluate_price(y_price_test, pred_returns, last_closes_test)
    dir_metrics = evaluate_direction(y_dir_test, pred_dir_prob)

    print(f"{'Metric':<20} | {'Value':<10}")
    print("-" * 35)
    print(f"{'Price MAE (₹)':<20} | {price_metrics['mae']:.2f}")
    print(f"{'Price RMSE (₹)':<20} | {price_metrics['rmse']:.2f}")
    print(f"{'Price R2 Score':<20} | {price_metrics['r2']:.4f}")
    print(f"{'Direction Accuracy':<20} | {dir_metrics['accuracy']:.4%}")
    print(f"{'Direction F1':<20} | {dir_metrics['f1']:.4f}")

    actual_prices = last_closes_test * (1 + y_price_test)
    predicted_prices = last_closes_test * (1 + pred_returns)
    
    plot_predictions(actual_prices, predicted_prices, model_name)
    plot_confusion_matrix(dir_metrics['conf_matrix'], model_name)

    return {"price": price_metrics, "direction": dir_metrics}


def compare_all_models(model_names: list, splits: tuple):
    """
    Run evaluate_model() for all 3 models and print comparison table.

    TODO:
    1. Loop over model_names, collect metrics for each
    2. Print a comparison table:
        Model       | MAE(₹) | RMSE(₹) | MAPE(%) | R²   | Dir Acc | F1
        ----------------------------------------------------------------
        gru         | ...    | ...     | ...     | ...  | ...     | ...
        lstm        | ...    | ...     | ...     | ...  | ...     | ...
        transformer | ...    | ...     | ...     | ...  | ...     | ...
    3. Also call baseline_metrics() and print it alongside for comparison
    """
    all_metrics = {}

    for name in model_names:
        metrics = evaluate_model(name, splits)
        all_metrics[name] = metrics

    (_, _, _,
    _, _, y_price_test,
    _, _, _,
    _, _, last_closes_test) = splits
    
    print("\n" + "="*80)
    print("BASELINE PERFORMANCE (Predicting 0% Return)")
    print("-"*80)
    baseline = baseline_metrics(y_price_test, last_closes_test)
    
    print("\n" + "="*80)
    print(f"{'Model':<12} | {'MAE(₹)':<8} | {'RMSE(₹)':<8} | {'MAPE(%)':<8} | {'R²':<7} | {'Dir Acc':<8} | {'F1':<6}")
    print("-"*80)
    
    for name in model_names:
        m = all_metrics[name]
        print(f"{name:<12} | "
              f"{m['price']['mae']:<8.2f} | "
              f"{m['price']['rmse']:<8.2f} | "
              f"{m['price']['mape']:<8.2f} | "
              f"{m['price']['r2']:<7.4f} | "
              f"{m['direction']['accuracy']:<8.2%} | "
              f"{m['direction']['f1']:<6.4f}")
    
    
    print(f"{'baseline':<12} | "
          f"{baseline['mae']:<8.2f} | "
          f"{baseline['rmse']:<8.2f} | "
          f"{baseline['mape']:<8.2f} | "
          f"{baseline['r2']:<7.4f} | "
          f"{'50.00%':<8} | {'N/A':<6}")
    print("="*80)

    return all_metrics


if __name__ == "__main__":
    from train import prepare_data

    print("Fetching data for evaluation...")
    splits = prepare_data()
    
    compare_all_models(['gru', 'lstm', 'transformer'], splits)
    plot_training_curves(['gru', 'lstm', 'transformer'])