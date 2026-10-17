"""Run once in the environment that can load the original pickle; NO training."""
import argparse,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'service'))
from model_io import export_portable_xgboost
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--force',action='store_true');args=ap.parse_args()
    print(export_portable_xgboost(ROOT/'models/final/xgboost.pkl',force=args.force))
