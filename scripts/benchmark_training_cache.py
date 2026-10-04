"""Bounded full-trainer cache comparison on independent synthetic fixtures."""
import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--frames', type=int, default=128)
    p.add_argument('--cache-mib', type=int, default=512)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = p.parse_args()
    if not (4 <= args.frames <= 768 and 1 <= args.epochs <= 5 and 1 <= args.repeats <= 3 and 1 <= args.cache_mib <= 2048):
        p.error('bounded fixture: frames 4..768, epochs 1..5, repeats 1..3, cache MiB 1..2048')
    if args.output.exists(): p.error('output must be new')
    import torch
    from forza_ai.data.synthetic import generate
    from forza_ai.training.engine import TrainConfig, train
    torch.set_num_threads(1)
    def synchronize():
        if args.device == 'cuda': torch.cuda.synchronize()
    if args.device == 'cuda' and not torch.cuda.is_available(): p.error('CUDA unavailable')
    records = []
    reference = None
    with tempfile.TemporaryDirectory(prefix='forza-full-trainer-cache-') as temporary:
        root = Path(temporary)
        data = generate(root / 'data', sessions=3, frames=args.frames, seed=7)
        # Warm up libraries once; excluded from measurements and not reused.
        with contextlib.redirect_stdout(io.StringIO()):
            train(data, root / 'warmup', epochs=1, config=TrainConfig(cache_mib=0), device=args.device)
        for repeat in range(args.repeats):
            for budget in ([0, args.cache_mib] if repeat % 2 == 0 else [args.cache_mib, 0]):
                synchronize(); start = time.perf_counter()
                diagnostic = io.StringIO()
                with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(diagnostic):
                    saved = train(data, root / f'run-{repeat}-{budget}', epochs=args.epochs,
                                  config=TrainConfig(cache_mib=budget), device=args.device)
                synchronize(); elapsed = time.perf_counter() - start
                if reference is None: reference = saved
                assert saved['history'] == reference['history']
                assert saved['train_sessions'] == reference['train_sessions']
                assert saved['validation_sessions'] == reference['validation_sessions']
                assert all(torch.equal(value, reference['model_state'][key]) for key, value in saved['model_state'].items())
                records.append({'repeat': repeat, 'cache_mib': budget, 'total_s': elapsed,
                                'train_samples_per_epoch': args.frames * 2,
                                'validation_samples_per_epoch': args.frames,
                                'cache_admission': [json.loads(line)['preprocessing_cache'] for line in diagnostic.getvalue().splitlines() if line.startswith('{"preprocessing_cache":')],
                                'exact_history_and_weights_parity': True})
    result = {'synthetic_only': True, 'device': args.device, 'torch': torch.__version__,
              'threads': 1, 'batch_size': 32, 'seed': 7, 'epochs': args.epochs,
              'timing': 'full train call including data validation, cache fill, validation, checkpoint writes; excludes fixture creation and separate warmup',
              'runs': records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle: json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
