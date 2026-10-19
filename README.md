# ML Projects

A collection of standalone machine learning notebooks covering classical ML,
CNNs, and a transformer fine-tune — mostly classification tasks across a mix
of standard and custom datasets.

| Notebook | Task | Approach |
|---|---|---|
| `BrainTumor.ipynb` | Brain tumor classification on tabular data | Multiple classifiers compared (incl. Random Forest), evaluated on accuracy/F1/precision/recall |
| `CNNModelMaking_.ipynb` | Diabetic retinopathy detection (binary) | CNN vs. SVC comparison on a retinopathy dataset |
| `Clg_project_2.ipynb` | 5-class image classification | Fine-tunes a Vision Transformer (ViT) with PyTorch + Hugging Face Transformers |
| `Linear_regression.ipynb` | Housing price prediction | Linear regression with cross-validation (`scikit-learn`) |
| `Logistic_regression.ipynb` | Breast cancer classification | Logistic regression on the sklearn breast cancer dataset |
| `Maratha_warfield.ipynb` | Historical Maratha weaponry classification | Classical ML models on a custom dataset, with model comparison |
| `Marathi_ornaments.ipynb` | Traditional Marathi ornament classification | Classical ML + KNN comparison on a custom dataset |

## Stack

Python, pandas, NumPy, scikit-learn, PyTorch, Hugging Face Transformers,
seaborn/matplotlib for visualization.

## Running

Each notebook is self-contained — open it in Jupyter and run top to bottom.
Dataset paths are currently hardcoded to local file locations; swap them for
your own copy of the relevant dataset before running.
