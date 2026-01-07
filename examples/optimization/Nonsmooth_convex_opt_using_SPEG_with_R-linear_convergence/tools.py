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