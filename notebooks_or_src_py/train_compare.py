"""Legacy command forwards to the official, backup-aware training entry point.

Use tune_models.py first; train_compare.py --force is equivalent to train_all.py --force.
Original implementation is preserved in results/baseline_original/source_before_round1.zip.
"""
from train_all import main

if __name__ == '__main__':
    main()
