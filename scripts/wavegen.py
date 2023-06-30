import numpy as np
import scipy.signal.windows as windows 
from common import Description, gen_sine_data, PathManage, write_as_json, titlespec, axesspec
import matplotlib.pyplot as plt

PATH = "datasets/wavegen"

def gen_sine_freq_data():
    func = "sine_freq"
    # Arbitrary values
    fsine = 2.0
    fsample = 100.0
    duration = 8.0

    desc = Description()
    desc.input_data = {
        "fsine": fsine,
        "fsample": fsample,
        "duration": duration 
    }
    desc.path = PATH
    desc.func = func
    desc.output_data = gen_sine_data(fsine, fsample, duration)

    plt.plot(np.linspace(0, duration, num=int(np.ceil(duration * fsample))), desc.output_data)
    plt.grid(True)
    plt.title(func, **titlespec)
    plt.ylabel("Amplitude", **axesspec)
    plt.xlabel("Time", **axesspec)
    plt.show()

    write_as_json(desc)

def gen_raw_sine_data():
    func = "sine"
    # Arbitrary values
    fsine = 2.0
    fsample = 100.0
    duration = 8.0

    num_points = int(np.ceil(fsample * duration))
    samples = np.linspace(0, duration, num=num_points)
    input_data = 2 * np.pi * fsine * samples
    output_data = np.sin(input_data) 

    desc = Description()
    desc.input_data = input_data
    desc.output_data = output_data
    desc.func = func
    desc.path = PATH

    plt.plot(input_data, output_data)
    plt.grid(True)
    plt.title(func, **titlespec)
    plt.ylabel("Amplitude", **axesspec)
    plt.xlabel("Phase [radians]", **axesspec)
    plt.show()

    write_as_json(desc)

def gen_sinc_data():
    func = "sinc"

    start = -20
    stop = 20
    fs = 100
    num_points = (stop - start) * fs
    input_data = np.linspace(start, stop, num=num_points)

    output_data = np.sinc(input_data)

    plt.plot(input_data, output_data)
    plt.grid(True)
    plt.title(func, **titlespec)
    plt.ylabel("Amplitude", **axesspec)
    plt.xlabel("Phase [np.pi * x]", **axesspec)
    plt.show()

    desc = Description()
    desc.input_data = input_data
    desc.output_data = output_data
    desc.func = func
    desc.path = PATH

    write_as_json(desc)

def gen_square_wave_data():
    pass

def gen_pulse_wave_data():
    pass

def gen_window_data():
    pass

if __name__ == "__main__":
    gen_raw_sine_data()
    gen_sine_freq_data()
    gen_sinc_data()