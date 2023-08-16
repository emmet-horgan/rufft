from scipy.fft import fft, fftfreq
import numpy as np 
import matplotlib.pyplot as plt
from common import PathManage, Description, gen_sine_data, write_as_json

PATH = "datasets/fft"

def gen_fft_sine_data():
    
    # Arbitrary values
    fsine = 2.0
    fsample = 100.0
    duration = 8.0

    input_data = gen_sine_data(fsine, fsample, duration)
    desc = Description()
    desc.path  = PATH
    desc.input_data = input_data
    desc.ienum = "Array"
    desc.output_data = fft(input_data)

    mag = np.abs(desc.output_data)
    phase = np.angle(desc.output_data)

    desc.output_data = {
        "mag": mag.tolist(),
        "phase": phase.tolist()
    }
    desc.oenum = "ComplexVals"

    desc.func = "fft"

    plt.plot(np.flip(fftfreq(int(np.ceil(duration * fsample)), 1/fsample)), mag)
    plt.show()

    write_as_json(desc)

def gen_fftfreq_data():

    func = "fftfreq"

    n = 1000
    fs = 100
    d = 1 / fs
    input_data = {
        "n": n,
        "d": d
    }
    

    output_data = fftfreq(n, d)
    desc = Description(input_data=input_data, output_data=output_data, func=func, path=PATH, ienum="FftFreqVals", oenum="Array")
    write_as_json(desc)

     
def gen_zero_pad_data():
    pass

if __name__ == "__main__":
    gen_fft_sine_data()
    gen_fftfreq_data()
    gen_zero_pad_data()