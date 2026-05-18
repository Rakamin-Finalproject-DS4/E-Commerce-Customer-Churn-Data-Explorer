import nbformat
from pathlib import Path

notebook_path = Path('notebook/ecommerce_prediction.ipynb')
nb = nbformat.read(notebook_path, as_version=4)
new_lines = [
    "def evaluate_model(name, model, X_test, y_test):",
    "    y_pred = model.predict(X_test)",
    "    acc = accuracy_score(y_test, y_pred)",
    "    prec = precision_score(y_test, y_pred)",
    "    rec = recall_score(y_test, y_pred)",
    "    f1 = f1_score(y_test, y_pred)",
    "    print(f'### {name}')",
    "    print('Accuracy:', round(acc, 4))",
    "    print('Precision:', round(prec, 4))",
    "    print('Recall:', round(rec, 4))",
    "    print('F1-Score:', round(f1, 4))",
    "    print('\\nClassification Report:\\n', classification_report(y_test, y_pred, digits=4))",
    "    print('Confusion Matrix:\\n', confusion_matrix(y_test, y_pred))",
    "    return {'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}",
    "",
    "results_rf = evaluate_model('Random Forest', rf, X_test_trans, y_test)",
    "results_xgb = evaluate_model('XGBoost', xgb, X_test_trans, y_test)",
]
for cell in nb.cells:
    if cell.cell_type == 'code' and 'def evaluate_model' in cell.source:
        cell.source = '\n'.join(new_lines) + '\n'
        break
nbformat.write(nb, notebook_path)
print('Updated evaluation cell source with safe assembly.')
