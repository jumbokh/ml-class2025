import zipfile

try:
    with zipfile.ZipFile('Dimensionality reduction (PCA, tSNE)/train.csv.zip', 'r') as z:
        print("Files in zip archive:")
        for file in z.namelist():
            print(f"- {file}")
            
        # 檢查列名
        with z.open(z.namelist()[0]) as f:
            header = f.readline().decode('utf-8').strip()
            print("\nColumn names:")
            print(header)
except Exception as e:
    print(f"Error: {e}")
