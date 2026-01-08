import pandas as pd

def ensure_length(data, length):
    data = list(data)
    if len(data) == 0: return [0.0] * length 
    if len(data) < length: return data + [data[-1]] * (length - len(data))
    else: return data[:length]
    
def format_sci_latex(x):
    if isinstance(x, str): return x
    if x == 0: return "0"
    s = "{:.2e}".format(x) 
    base, exponent = s.split('e')
    return fr"${base} \times 10^{{{int(exponent)}}}$"

def save_table_to_txt(df, filename, precision=4, formatting="exponential"):
    if formatting == "exponential":
        formatter = "{:.4e}".format
    else:
        formatter = "{:.4f}".format
        
    latex_str = df.to_latex(float_format=formatter)
    
    with open(filename, "w", encoding="utf-8") as f:
        f.write(latex_str)
    print(f"[Saved] Table saved to: {filename}")