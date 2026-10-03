"""Run with forza-train or python -m forza_ai.training.cli."""
import argparse
import json
from pathlib import Path

from forza_ai.data.sessions import Alignment, load_sessions
from forza_ai.data.synthetic import generate


def _alignment(parser):
    parser.add_argument('--label-offset-ms', type=float, default=0)
    parser.add_argument('--max-wheel-gap-ms', type=float, default=50)
    parser.add_argument('--max-telemetry-age-ms', type=float, default=100)


def _get_alignment(args):
    return Alignment(round(args.label_offset_ms * 1e6), round(args.max_wheel_gap_ms * 1e6),
                     round(args.max_telemetry_age_ms * 1e6))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='command', required=True)
    synthetic = subs.add_parser('synthetic', help='create deterministic test sessions')
    synthetic.add_argument('output', type=Path)
    synthetic.add_argument('--sessions', type=int, default=3)
    synthetic.add_argument('--frames', type=int, default=24)
    synthetic.add_argument('--seed', type=int, default=7)
    validate = subs.add_parser('validate', help='validate completed sessions and report excluded frames')
    validate.add_argument('data', type=Path)
    _alignment(validate)
    train = subs.add_parser('train', help='start a new run')
    train.add_argument('data', type=Path)
    train.add_argument('output', type=Path)
    train.add_argument('--epochs', type=int, default=10)
    train.add_argument('--batch-size', type=int, default=32)
    train.add_argument('--learning-rate', type=float, default=1e-3)
    train.add_argument('--validation-fraction', type=float, default=.25)
    train.add_argument('--seed', type=int, default=7)
    train.add_argument('--workers', type=int, default=0)
    train.add_argument('--device', default='auto')
    _alignment(train)
    resume = subs.add_parser('resume', help='continue to a total epoch count using saved configuration')
    resume.add_argument('checkpoint', type=Path)
    resume.add_argument('data', type=Path)
    resume.add_argument('--epochs', type=int, required=True)
    resume.add_argument('--device', default='auto')
    evaluate = subs.add_parser('evaluate', help='report held-out model, zero and training-mean errors')
    evaluate.add_argument('checkpoint', type=Path)
    evaluate.add_argument('data', type=Path)
    evaluate.add_argument('--device', default='cpu')
    evaluate.add_argument('--unseen', action='store_true')
    export = subs.add_parser('export', help='export CPU weights and preprocessing metadata')
    export.add_argument('checkpoint', type=Path)
    export.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    try:
        if args.command == 'synthetic':
            generate(args.output, args.sessions, args.frames, args.seed)
            print(json.dumps({'created': str(args.output), 'synthetic_only': True}))
        elif args.command == 'validate':
            sessions = load_sessions(args.data, _get_alignment(args))
            print(json.dumps([session.summary() for session in sessions], indent=2))
            if any(not session.samples for session in sessions):
                raise ValueError('one or more sessions have no accepted samples')
        else:
            from forza_ai.training import engine
            if args.command == 'train':
                config = engine.TrainConfig(args.batch_size, args.learning_rate,
                                            args.validation_fraction, args.seed, args.workers)
                engine.train(args.data, args.output, args.epochs, config, _get_alignment(args), device=args.device)
            elif args.command == 'resume':
                engine.train(args.data, args.checkpoint.parent, args.epochs,
                             device=args.device, resume=args.checkpoint)
            elif args.command == 'evaluate':
                print(json.dumps(engine.evaluate(args.checkpoint, args.data, args.device, args.unseen), indent=2))
            elif args.command == 'export':
                print(json.dumps(engine.export(args.checkpoint, args.output), indent=2))
    except (ValueError, OSError, KeyError, TypeError, OverflowError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
