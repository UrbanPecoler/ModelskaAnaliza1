import pandas as pd

# CELOTNA TABELA ŽIVIL
df = pd.read_excel("Zivila.xlsx")
latex_table = df.to_latex(index=False, float_format="%.2f") 
# print(latex_table)