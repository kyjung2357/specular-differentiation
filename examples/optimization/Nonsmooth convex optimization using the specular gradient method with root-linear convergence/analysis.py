import specular
from specular.optimization.classical_solver import Adam, BFGS, gradient_descent_method

import numpy as np
import pandas as pd
from tqdm import tqdm  
import matplotlib.pyplot as plt
from IPython.display import clear_output, display
import matplotlib
import sys
import os
import torch
from concurrent.futures import ProcessPoolExecutor 

# 스타일 설정
matplotlib.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams["font.family"] = "Times New Roman"

machine_epsilon = np.finfo(float).eps

# -----------------------------------------------------------
# [1] Auxiliary functions
# -----------------------------------------------------------
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

# -----------------------------------------------------------
# [2] A single function
# -----------------------------------------------------------
def run_single_experiment(args):
    seed, f, f_torch, max_iter = args
    
    # 시드 고정
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1) 

    history = {}
    x_0_val = np.random.randn(1)
    
    # ==== Specular Methods ====
    # (StepSize 클래스는 specular 패키지 안에 있다고 가정)
    step_size1 = specular.StepSize(name='geometric_series', parameters=[1, 0.5])
    
    # ISPEG
    # specular.gradient_method가 OptimizationResult를 반환한다고 가정
    _, res = specular.gradient_method(
        f=f, x_0=x_0_val, step_size=step_size1, form='implicit', 
        max_iter=max_iter, print_bar=False
    ).history()
    history["ISPEG"] = ensure_length(res, max_iter)

    # SPEG (geometric series)
    _, res = specular.gradient_method(
        f=f, x_0=x_0_val, step_size=step_size1, 
        max_iter=max_iter, print_bar=False
    ).history()
    history["SPEG geo"] = ensure_length(res, max_iter)

    # SPEG (square summable not summable)
    step_size2 = specular.StepSize(name='square_summable_not_summable', parameters=[2, 0])
    _, res = specular.gradient_method(
        f=f, x_0=x_0_val, step_size=step_size2, 
        max_iter=max_iter, print_bar=False
    ).history()
    history["SPEG sq"] = ensure_length(res, max_iter)

    # ==== Classical Methods ====
    # [수정됨] 이제 이 함수들도 OptimizationResult를 반환하므로 .history()를 호출합니다.
    
    # Gradient Descent (geometric series)
    _, res = gradient_descent_method(
        f_torch=f_torch, x_0=x_0_val, step_size=step_size1, max_iter=max_iter
    ).history()
    history["GD geo"] = ensure_length(res, max_iter)

    # Gradient Descent (square summable not summable)
    _, res = gradient_descent_method(
        f_torch=f_torch, x_0=x_0_val, step_size=step_size2, max_iter=max_iter
    ).history()
    history["GD sq"] = ensure_length(res, max_iter)

    # Adam (step_size로 float를 넘겨도 내부에서 처리됨)
    _, res = Adam(
        f_torch=f_torch, x_0=x_0_val, step_size=0.01, max_iter=max_iter
    ).history()
    history["Adam"] = ensure_length(res, max_iter)

    # BFGS
    _, res = BFGS(
        f_np=f, x_0=x_0_val, max_iter=max_iter
    ).history()
    history["BFGS"] = ensure_length(res, max_iter)
    
    return history

# -----------------------------------------------------------
# [3] Main function
# -----------------------------------------------------------
def repeat_experiment(f, f_torch, num_runs, max_iter, latex_code=False, save_name=False):
    histories = {"ISPEG": [], "SPEG geo": [], "SPEG sq": [], "GD geo": [], "GD sq": [], "Adam": [], "BFGS": []}

    seeds = range(num_runs)
    tasks = [(seed, f, f_torch, max_iter) for seed in seeds]

    # CPU 코어 수 확인 및 병렬 처리
    num_workers = min(os.cpu_count(), num_runs) # type: ignore
    print(f"Starting parallel execution with {num_workers} cores...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(run_single_experiment, tasks), total=num_runs, desc="Running Experiments"))

    for single_history in results:
        for method, data in single_history.items():
            histories[method].append(data)

    # ==== Visualization ====
    plt.figure(figsize=(7, 3))
    x_axis = range(1, max_iter + 1)

    colors = {
        'ISPEG': 'blue', 'SPEG geo': 'red', 'SPEG sq': 'orange', 
        'GD geo': 'darkgreen', 'GD sq': 'limegreen', 'Adam': 'brown', 'BFGS': 'black'
    }
    linestyles = {
        'ISPEG': '-', 'SPEG geo': '-', 'SPEG sq': '-', 
        'GD geo': '-', 'GD sq': '-', 'Adam': '-', 'BFGS': '-'
    }

    for name, data in histories.items():
        arr = np.array(data)
        # 통계 계산
        med = np.median(arr, axis=0)
        q25 = np.percentile(arr, 25, axis=0)
        q75 = np.percentile(arr, 75, axis=0)
        
        c = colors.get(name, 'black')
        ls = linestyles.get(name, '-')
        
        plt.plot(x_axis, med, label=f'{name}', color=c, linestyle=ls)
        plt.fill_between(x_axis, q25, q75, color=c, alpha=0.15)

    plt.xlabel('Iteration $k$', fontsize='10')
    plt.ylabel(r'Objective function value $f(x_k)$', fontsize='10') 
    plt.yscale('log')
    plt.xlim(1, max_iter)
    # 범례 위치 조정
    plt.legend(bbox_to_anchor=(1.02, 0.5), loc='center left', fontsize='10', frameon=True)
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.tight_layout()

    if save_name:
        current_dir = os.getcwd() # 현재 작업 경로 사용
        figures_dir = os.path.join(current_dir, 'figures')

        if not os.path.exists(figures_dir):
            os.makedirs(figures_dir)

        fig_name = f"figure_{save_name}.pdf"
        fig_path = os.path.join(figures_dir, fig_name)

        plt.savefig(fig_path, dpi=1000, bbox_inches='tight') 
        print(f"[Saved] Plot saved to: {fig_path}")

    # plt.show()

    # ==== Table ====
    table_data = []
    for name, runs in histories.items():
        # 각 실행(run)에서 가장 작게 떨어진 에러값(Best Error)들의 분포를 봄
        best_errors = [max(np.min(run_history), machine_epsilon) for run_history in runs]
        table_data.append({
            "Method": name,
            "Mean": np.mean(best_errors),
            "Median": np.median(best_errors),
            "Standard deviation": np.std(best_errors)
        })

    df_summary = pd.DataFrame(table_data)
    df_display = df_summary.copy()
    cols_to_format = ["Mean", "Median", "Standard deviation"]
    
    for col in cols_to_format:
        if col in df_display.columns:
            df_display[col] = df_display[col].apply(format_sci_latex)

    print(f"\n=== Performance Summary ({num_runs} runs) ===")
    display(df_display)

    if latex_code:
        latex_str = df_display.to_latex(index=False, escape=False)

        if save_name:
            table_dir = os.path.join(current_dir, 'tables') # type: ignore

            if not os.path.exists(table_dir):
                os.makedirs(table_dir)

            table_name = f"table_{save_name}.txt"
            table_path = os.path.join(table_dir, table_name)

            with open(table_path, "w", encoding="utf-8") as txt_file:
                txt_file.write(latex_str)
            
            print(f"[Saved] Table saved to: {table_path}")