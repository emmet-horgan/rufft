import wavegen, fft, os
from common import argparse_setup
from pathlib import Path

if __name__ == '__main__':
    args = argparse_setup()
    
    datasets = Path(__file__).parent.parent / 'datasets'
    os.mkdir(datasets)
    os.mkdir(datasets / 'fft')
    os.mkdir(datasets / 'wavegen')

    wavegen.main(args.plot)
    fft.main(args.plot)