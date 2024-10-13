import os
from pathlib import Path 
import numpy as np
import json 

titlespec = {
    "fontname": "DejaVu Sans",
    "fontsize": 14,
    }
axesspec = {
    "fontname": "DejaVu Sans",
    "fontstyle": "oblique",
    "fontsize": 12,
    }

class PathManage:

    __root = ""

    def __init__(self):
        self.__root = self.findroot()

    def path(self, path, file=None, dir=None):
        pathlst = list(Path(path).parts)
        for i, val in enumerate(pathlst):
            if val == "." or val == ".." or val == "":
                pathlst.pop(i)
        length = len(pathlst)
        for root, dirs, files in os.walk(self.__root):
            
            for name in dirs:
                cur = os.path.join(root, name)
                cur_parts = list(Path(cur).parts)
                semipath = cur_parts[len(cur_parts) - length:]
                if semipath == pathlst:
                    return cur
        
            for name in files:
                cur = os.path.join(root, name)
                cur_parts = list(Path(cur).parts)
                semipath = cur_parts[len(cur_parts) - length:]
                if semipath == pathlst:
                    return cur

    @staticmethod
    def findroot(root="RusticFourier"):
        path = list(Path(os.getcwd()).parts)
        if root not in path:
            print(path)
            raise FileNotFoundError
        dir = "."
        path.reverse()
        for name in path:
            if name != root:
                dir = os.path.join(dir, "..")
            else:
                return dir  

class Description:

    def __init__(self, input_data=None, output_data=None, func=None, path=None, ienum=None, oenum=None):
        self.__input_data = input_data
        self.__output_data = output_data
        self.__func = func
        self.__path = path
        self.__ienum = ienum
        self.__oenum = oenum
    
    @property
    def input_data(self):
        return self.__input_data
    
    @input_data.setter
    def input_data(self, x):
        self.__input_data = x
    
    @property
    def output_data(self):
        return self.__output_data
    
    @output_data.setter
    def output_data(self, x):
        self.__output_data = x
    
    @property
    def func(self):
        return self.__func
    
    @func.setter
    def func(self, x):
        self.__func = x

    @property
    def path(self):
        return self.__path

    @path.setter
    def path(self, x):
        self.__path = x
    
    @property
    def ienum(self):
        return self.__ienum
    
    @property
    def oenum(self):
        return self.__oenum
    
    @ienum.setter
    def ienum(self, x):
        self.__ienum = x
    
    @oenum.setter
    def oenum(self, x):
        self.__oenum = x

def write_as_json(desc: Description):
    paths = PathManage()
    jsondata = {
        "function": desc.func,
        "path": desc.path,
        "input_data": {f"{desc.ienum}": desc.input_data.tolist() if type(desc.input_data) is np.ndarray else desc.input_data},
        "output_data": {f"{desc.oenum}": desc.output_data.tolist() if type(desc.output_data) is np.ndarray else desc.output_data}
    }
    path = paths.path(desc.path)
    path = os.path.join(path, desc.func)
    if not os.path.isdir(path):
        os.mkdir(path)
    file = os.path.join(path, f"{desc.func}.json")
    with open(file, "w") as f:
        json.dump(jsondata, f)

def gen_sine_data(f: float, fs: float, duration: float, phase_offset: float = 0.0):

    num_points = int(np.ceil(fs * duration))
    samples = np.linspace(0, duration, num=num_points)
    phase = 2 * np.pi * f * samples + phase_offset

    return np.sin(phase)  # Sine wave is most likely an input e.g. fft

def gen_complex_exp_data(f: float, fs: float, duration: float, phase_offset: float = 0.0):

    num_points = int(np.ceil(fs * duration))
    samples = np.linspace(0, duration, num=num_points)
    phase = 1j *2 * np.pi * f * samples + phase_offset

    return np.exp(phase)  # Sine wave is most likely an input e.g. fft