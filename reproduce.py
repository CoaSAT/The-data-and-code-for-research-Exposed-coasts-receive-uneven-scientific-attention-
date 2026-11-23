"""Run the processed-data reproduction of manuscript Figures 1--3."""
from pathlib import Path
import argparse, subprocess, sys, time, json, platform
ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--figure', choices=['all','1','2','3'], default='all')
    args = parser.parse_args()
    logdir = ROOT/'validation'; logdir.mkdir(exist_ok=True)
    results = []
    for figure in ([1,2,3] if args.figure=='all' else [int(args.figure)]):
        folder = ROOT/f'Figure{figure}'
        start = time.time()
        process = subprocess.run([sys.executable, str(folder/'plot.py')], cwd=folder, capture_output=True, text=True, encoding='utf-8', errors='replace')
        (logdir/f'Figure{figure}.log').write_text(process.stdout+'\n'+process.stderr, encoding='utf-8')
        outputs = [str(p.relative_to(ROOT)) for p in (folder/'outputs').glob('*') if p.is_file()]
        result = dict(figure=figure, returncode=process.returncode, seconds=round(time.time()-start,2), outputs=outputs)
        results.append(result)
        print(json.dumps(result), flush=True)
        (logdir/'run_report.json').write_text(json.dumps(dict(python=sys.version,platform=platform.platform(),results=results),indent=2),encoding='utf-8')
        if process.returncode or not outputs:
            raise SystemExit(f'Figure {figure} failed; see validation/Figure{figure}.log')

if __name__ == '__main__':
    main()
