import json
import io
import sys
import os
import matplotlib
matplotlib.use('Agg')

nb_path = os.path.join('food-demand-analysis', 'notebooks', 'analysis.ipynb')
with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

original_cwd = os.getcwd()
notebooks_dir = os.path.abspath(os.path.join('food-demand-analysis', 'notebooks'))
os.chdir(notebooks_dir)

global_scope = {'__builtins__': __builtins__}

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        print(f"Executing cell {cell['execution_count']}...", flush=True)
        stdout_capture = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = stdout_capture
        try:
            exec(source, global_scope)
        except Exception as e:
            sys.stdout = old_stdout
            os.chdir(original_cwd)
            print(f"Error in cell {cell['execution_count']}: {e}", file=sys.stderr)
            raise e
        sys.stdout = old_stdout
        output_text = stdout_capture.getvalue()
        if output_text:
            cell['outputs'] = [{
                'output_type': 'stream',
                'name': 'stdout',
                'text': output_text.splitlines(keepends=True)
            }]

os.chdir(original_cwd)

with open(nb_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Notebook successfully executed from start to end with zero errors!", flush=True)
