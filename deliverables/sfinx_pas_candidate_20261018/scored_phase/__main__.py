import argparse


def main():
    parser=argparse.ArgumentParser(description='PAS candidate preparation. Does not start a trading daemon.')
    parser.add_argument('command',choices=['preflight','collect','backtest'])
    parser.add_argument('--output',help='New directory for a repeated historical backtest')
    args=parser.parse_args()
    if args.command=='preflight':
        from .preflight import main as run
    elif args.command=='collect':
        from .collect import main as run
    else:
        from .backtest import main as run
    if args.command=='backtest':run(args.output)
    else:
        if args.output:parser.error('--output is only valid for backtest')
        run()


if __name__=='__main__':main()
