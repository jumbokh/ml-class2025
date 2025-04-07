import nbformat
import json

with open("Dimensionality reduction (PCA, tSNE)/iris-pca.ipynb", "r", encoding="utf-8") as f:
    nb = nbformat.read(f, as_version=4)
    
print(json.dumps(nb, indent=2, ensure_ascii=False))
