import os  
import sys  
from dotenv import load_dotenv  # Load environment variables from .env file  
import shutil  # High-level file operations  
import multiprocessing  # For parallel processing  
import numpy as np  
import torch  # Main library for pytorch operations  
import torch.nn.functional as F  # Contains standard activation functions and more  
import torchaudio.transforms as tat  # Audio processing functions  
import librosa  # A toolkit for music and audio analysis  
import sounddevice as sd  # For playing and recording sound  
import FreeSimpleGUI as sg  # For GUI elements  

# Load environment variables  
load_dotenv()  

os.environ["OMP_NUM_THREADS"] = "4"  # Set the number of threads for OpenMP  
if sys.platform == "darwin":  
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Enable MPS fallback on macOS  

current_dir = os.getcwd()  # Get current working directory  
sys.path.append(current_dir)  # Add current directory to the system path  

flag_vc = False  # Global flag for voice conversion  

# Custom print function  
def printt(strr, *args):  
    if len(args) == 0:  
        print(strr)  
    else:  
        print(strr % args)  

# Function to perform phase vocoding, used to modify the timing of audio signals without affecting the pitch  
def phase_vocoder(a, b, fade_out, fade_in):  
    window = torch.sqrt(fade_out * fade_in)  # Window function for smooth transition  
    fa = torch.fft.rfft(a * window)  # Real FFT on input 'a'  
    fb = torch.fft.rfft(b * window)  # Real FFT on input 'b'  
    absab = torch.abs(fa) + torch.abs(fb)  # Summation of the amplitude spectrums  
    n = a.shape[0]  

    if n % 2 == 0:  # If even number of samples  
        absab[1:-1] *= 2  
    else:  
        absab[1:] *= 2  

    phia = torch.angle(fa)  # Phase of FFT of 'a'  
    phib = torch.angle(fb)  # Phase of FFT of 'b'  
    deltaphase = phib - phia  # Phase difference  
    deltaphase = deltaphase - 2 * np.pi * torch.floor(deltaphase / 2 / np.pi + 0.5)  # Wrap angle difference  

    # Frequency bins  
    w = 2 * np.pi * torch.arange(n // 2 + 1).to(a) + deltaphase  
    t = torch.arange(n).unsqueeze(-1).to(a) / n  
    result = (  
        a * (fade_out**2)  
        + b * (fade_in**2)  
        + torch.sum(absab * torch.cos(w * t + phia), -1) * window / n  
    )  
    return result  

# Harvest class for multiprocessing  
class Harvest(multiprocessing.Process):  
    def __init__(self, inp_q, opt_q):  
        super().__init__()  
        self.inp_q = inp_q  # Input queue  
        self.opt_q = opt_q  # Output queue  

    def run(self):  
        import pyworld  # Import within the process, needed for harvest function  

        # Infinite loop to get data from input queue, process it, and put result to output queue  
        while True:  
            idx, x, res_f0, n_cpu, ts = self.inp_q.get()  
            f0, t = pyworld.harvest(  
                x.astype(np.double),  
                fs=16000,  
                f0_ceil=1100,  
                f0_floor=50,  
                frame_period=10,  
            )  
            res_f0[idx] = f0  
            if len(res_f0.keys()) >= n_cpu:  
                self.opt_q.put(ts)  


if __name__ == "__main__":  
    import json  
    import re  
    import threading  
    import time  
    from multiprocessing import Queue, cpu_count  
    from queue import Empty  

    # Other custom imports  
    from torchgate import TorchGate  # Custom torch operations and neural networks  
    from infer.lib import rtvc as rtvc_for_realtime  # Import voice conversion for real-time processes  
    from configs.config import Config  # Configuration settings  

    # Queue initialization for multiprocessing  
    inp_q = Queue()  
    opt_q = Queue()  
    n_cpu = min(cpu_count(), 8)  # Cap the number of processes to 8 to avoid overloading  

    for _ in range(n_cpu):  
        p = Harvest(inp_q, opt_q)  
        p.daemon = True  # Allow the process to be stopped with the program  
        p.start()  

    # Configuration class for GUI parameters  
    class GUIConfig:  
        def __init__(self):  
            # Initialization of parameters  
            self.pth_path: str = ""  # Path to model file  
            self.index_path: str = ""  # Path to index file  
            self.pitch: int = 0  # Pitch adjustment  
            self.formant: float = 0.0  # Formant adjustment  
            self.sr_type: str = "sr_model"  # Sample rate type  
            self.block_time: float = 0.25  # Block size in seconds for processing  
            self.threhold: int = -60  # Threshold for response  
            self.crossfade_time: float = 0.05  # Time for crossfading  
            self.extra_time: float = 2.5  # Additional inference time  
            self.I_noise_reduce: bool = False  # Input noise reduction flag  
            self.O_noise_reduce: bool = False  # Output noise reduction flag  
            self.use_pv: bool = False  # Use phase vocoder  
            self.rms_mix_rate: float = 0.0  # Ratio for mixing RMS envelope  
            self.index_rate: float = 0.0  # Index rate for processing  
            self.n_cpu: int = min(n_cpu, 4)  # Number of CPU processes allocated  
            self.f0method: str = "fcpe"  # Fundamental frequency extraction method  
            self.sg_hostapi: str = ""  # Host API for sounddevice  
            self.wasapi_exclusive: bool = False  # WASAPI exclusive mode flag  
            self.sg_input_device: str = ""  # Input device selected in GUI  
            self.sg_output_device: str = ""  # Output device selected in GUI  

    # GUI class for handling UI interactions and parameter collection  
    class GUI:  
        def __init__(self):  
            self.gui_config = GUIConfig()  
            self.config = Config()  
            self.function = "vc"  
            self.delay_time = 0  # Additional delay time  
            self.hostapis = None  # Host APIs list  
            self.input_devices = None  # List of input devices  
            self.output_devices = None  # List of output devices  
            self.input_devices_indices = None  # Indices of input devices  
            self.output_devices_indices = None  # Indices of output devices  
            self.stream = None  # Audio stream  
            self.update_devices()  # Update device lists  
            self.launcher()  # Launch the GUI  

        def load(self):  
            try:  
                # Load configuration if exists or create a new configuration file  
                if not os.path.exists("configs/inuse/config.json"):  
                    shutil.copy("configs/config.json", "configs/inuse/config.json")  
                with open("configs/inuse/config.json", "r") as j:  
                    data = json.load(j)  
                    data["sr_model"] = data["sr_type"] == "sr_model"  # Boolean for model sample rate  
                    data["sr_device"] = data["sr_type"] == "sr_device"  # Boolean for device sample rate  
                    data["pm"] = data["f0method"] == "pm"  # Pitch method  
                    data["harvest"] = data["f0method"] == "harvest"  
                    data["crepe"] = data["f0method"] == "crepe"  
                    data["rmvpe"] = data["f0method"] == "rmvpe"  
                    data["fcpe"] = data["f0method"] == "fcpe"  
                    if data["sg_hostapi"] in self.hostapis:  
                        self.update_devices(hostapi_name=data["sg_hostapi"])  
                        if (  
                            data["sg_input_device"] not in self.input_devices  
                            or data["sg_output_device"] not in self.output_devices  
                        ):  
                            self.update_devices()  
                            data["sg_hostapi"] = self.hostapis[0]  
                            data["sg_input_device"] = self.input_devices[  
                                self.input_devices_indices.index(sd.default.device[0])  
                            ]  
                            data["sg_output_device"] = self.output_devices[  
                                self.output_devices_indices.index(sd.default.device[1])  
                            ]  
                    else:  
                        data["sg_hostapi"] = self.hostapis[0]  
                        data["sg_input_device"] = self.input_devices[  
                            self.input_devices_indices.index(sd.default.device[0])  
                        ]  
                        data["sg_output_device"] = self.output_devices[  
                            self.output_devices_indices.index(sd.default.device[1])  
                        ]  
            except:  
                with open("configs/inuse/config.json", "w") as j:  
                    data = {  
                        "pth_path": "",  
                        "index_path": "",  
                        "sg_hostapi": self.hostapis[0],  
                        "sg_wasapi_exclusive": False,  
                        "sg_input_device": self.input_devices[  
                            self.input_devices_indices.index(sd.default.device[0])  
                        ],  
                        "sg_output_device": self.output_devices[  
                            self.output_devices_indices.index(sd.default.device[1])  
                        ],  
                        "sr_type": "sr_model",  
                        "threhold": -60,  
                        "pitch": 0,  
                        "formant": 0.0,  
                        "index_rate": 0,  
                        "rms_mix_rate": 0,  
                        "block_time": 0.25,  
                        "crossfade_length": 0.05,  
                        "extra_time": 2.5,  
                        "n_cpu": 4,  
                        "f0method": "rmvpe",  
                        "use_jit": False,  
                        "use_pv": False,  
                    }  
                    data["sr_model"] = data["sr_type"] == "sr_model"  
                    data["sr_device"] = data["sr_type"] == "sr_device"  
                    data["pm"] = data["f0method"] == "pm"  
                    data["harvest"] = data["f0method"] == "harvest"  
                    data["crepe"] = data["f0method"] == "crepe"  
                    data["rmvpe"] = data["f0method"] == "rmvpe"  
                    data["fcpe"] = data["f0method"] == "fcpe"  
            return data  

        def launcher(self):  
            # UIBuild  
            data = self.load()  
            self.config.use_jit = False  
            sg.theme("DarkGrey5")  
            layout = [  
                [  
                    sg.Frame(  
                        title=("Load Model"),  
                        layout=[  
                            [  
                                sg.Input(  
                                    default_text=data.get("pth_path", ""),  
                                    key="pth_path",  
                                ),  
                                sg.FileBrowse(  
                                    ("Select .pth file"),  
                                    initial_folder=os.path.join(  
                                        os.getcwd(), "assets/weights"  
                                    ),  
                                    file_types=((". pth"),),  
                                ),  
                            ],  
                            [  
                                sg.Input(  
                                    default_text=data.get("index_path", ""),  
                                    key="index_path",  
                                ),  
                                sg.FileBrowse(  
                                    ("Select .index file"),  
                                    initial_folder=os.path.join(os.getcwd(), "logs"),  
                                    file_types=((". index"),),  
                                ),  
                            ],  
                        ],  
                    )  
                ],  
                [  
                    sg.Frame(  
                        layout=[  
                            [  
                                sg.Text(("Device Type"), visible=False),  
                                sg.Combo(  
                                    self.hostapis,  
                                    key="sg_hostapi",  
                                    default_value=data.get("sg_hostapi", ""),  
                                    enable_events=True,  
                                    size=(20, 1),  
                                    visible=False,  
                                ),  
                                sg.Checkbox(  
                                    ("Exclusive WASAPI device"),  
                                    key="sg_wasapi_exclusive",  
                                    default=data.get("sg_wasapi_exclusive", False),  
                                    enable_events=True,  
                                    visible=False,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Input Device")),  
                                sg.Combo(  
                                    self.input_devices,  
                                    key="sg_input_device",  
                                    default_value=data.get("sg_input_device", ""),  
                                    enable_events=True,  
                                    size=(45, 1),  
                                ),  
                            ],  
                            [  
                                sg.Text(("Output Device")),  
                                sg.Combo(  
                                    self.output_devices,  
                                    key="sg_output_device",  
                                    default_value=data.get("sg_output_device", ""),  
                                    enable_events=True,  
                                    size=(45, 1),  
                                ),  
                            ],  
                            [  
                                sg.Button(("Reload Device List"), key="reload_devices", visible=False),  
                                sg.Radio(  
                                    ("Use model sampling rate"),  
                                    "sr_type",  
                                    key="sr_model",  
                                    default=data.get("sr_model", True),  
                                    enable_events=True,  
                                    visible=False,  
                                ),  
                                sg.Radio(  
                                    ("Use device sampling rate"),  
                                    "sr_type",  
                                    key="sr_device",  
                                    default=data.get("sr_device", False),  
                                    enable_events=True,  
                                    visible=False,  
                                ),  
                                sg.Text(("Sampling Rate:"), visible=False),  
                                sg.Text("", key="sr_stream", visible=False),  
                            ],  
                        ],  
                        title=("Audio Device"),  
                    )  
                ],  
                [  
                    sg.Frame(  
                        layout=[  
                            [  
                                sg.Text(("Response Threshold")),  
                                sg.Slider(  
                                    range=(-60, 0),  
                                    key="threhold",  
                                    resolution=1,  
                                    orientation="h",  
                                    default_value=data.get("threhold", -60),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Pitch Settings")),  
                                sg.Slider(  
                                    range=(-16, 16),  
                                    key="pitch",  
                                    resolution=1,  
                                    orientation="h",  
                                    default_value=data.get("pitch", 0),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Gender factor/voice thickness")),  
                                sg.Slider(  
                                    range=(-2, 2),  
                                    key="formant",  
                                    resolution=0.05,  
                                    orientation="h",  
                                    default_value=data.get("formant", 0.0),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Index Rate")),  
                                sg.Slider(  
                                    range=(0.0, 1.0),  
                                    key="index_rate",  
                                    resolution=0.01,  
                                    orientation="h",  
                                    default_value=data.get("index_rate", 0),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Loudness factor")),  
                                sg.Slider(  
                                    range=(0.0, 1.0),  
                                    key="rms_mix_rate",  
                                    resolution=0.01,  
                                    orientation="h",  
                                    default_value=data.get("rms_mix_rate", 0),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Pitch detection algorithm")),  
                                sg.Radio(  
                                    "pm",  
                                    "f0method",  
                                    key="pm",  
                                    default=data.get("pm", False),  
                                    enable_events=True,  
                                ),  
                                sg.Radio(  
                                    "harvest",  
                                    "f0method",  
                                    key="harvest",  
                                    default=data.get("harvest", False),  
                                    enable_events=True,  
                                ),  
                                sg.Radio(  
                                    "crepe",  
                                    "f0method",  
                                    key="crepe",  
                                    default=data.get("crepe", False),  
                                    enable_events=True,  
                                ),  
                                sg.Radio(  
                                    "rmvpe",  
                                    "f0method",  
                                    key="rmvpe",  
                                    default=data.get("rmvpe", False),  
                                    enable_events=True,  
                                ),  
                                sg.Radio(  
                                    "fcpe",  
                                    "f0method",  
                                    key="fcpe",  
                                    default=data.get("fcpe", True),  
                                    enable_events=True,  
                                ),  
                            ],  
                        ],  
                        title=("General Settings"),  
                        visible=False,  
                    ),  
                    sg.Frame(  
                        layout=[  
                            [  
                                sg.Text(("Sample Length")),  
                                sg.Slider(  
                                    range=(0.02, 1.5),  
                                    key="block_time",  
                                    resolution=0.01,  
                                    orientation="h",  
                                    default_value=data.get("block_time", 0.25),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Number of CPU processes used for harvest pitch algorithm")),  
                                sg.Slider(  
                                    range=(1, n_cpu),  
                                    key="n_cpu",  
                                    resolution=1,  
                                    orientation="h",  
                                    default_value=data.get(  
                                        "n_cpu", min(self.gui_config.n_cpu, n_cpu)  
                                    ),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Fade Length")),  
                                sg.Slider(  
                                    range=(0.01, 0.15),  
                                    key="crossfade_length",  
                                    resolution=0.01,  
                                    orientation="h",  
                                    default_value=data.get("crossfade_length", 0.05),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Text(("Extra inference time")),  
                                sg.Slider(  
                                    range=(0.05, 5.00),  
                                    key="extra_time",  
                                    resolution=0.01,  
                                    orientation="h",  
                                    default_value=data.get("extra_time", 2.5),  
                                    enable_events=True,  
                                ),  
                            ],  
                            [  
                                sg.Checkbox(  
                                    ("Input noise reduction"),  
                                    key="I_noise_reduce",  
                                    enable_events=True,  
                                ),  
                                sg.Checkbox(  
                                    ("Output noise reduction"),  
                                    key="O_noise_reduce",  
                                    enable_events=True,  
                                ),  
                                sg.Checkbox(  
                                    ("Enable Phase Vocoder"),  
                                    key="use_pv",  
                                    default=data.get("use_pv", False),  
                                    enable_events=True,  
                                ),  
                            ],  
                        ],  
                        title=("Performance Settings"),  
                        visible=False,  
                    ),  
                ],  
                [  
                    sg.Button(("Start audio conversion"), key="start_vc"),  
                    sg.Button(("Stop audio conversion"), key="stop_vc"),  
                    sg.Radio(  
                        ("Input voice monitor"),  
                        "function",  
                        key="im",  
                        default=False,  
                        enable_events=True,  
                        visible=False,  
                    ),  
                    sg.Radio(  
                        ("Output converted voice"),  
                        "function",  
                        key="vc",  
                        default=True,  
                        enable_events=True,  
                        visible=False,  
                    ),  
                    sg.Text(("Algorithmic delays(ms):"), visible=False),  
                    sg.Text("0", key="delay_time", visible=False),  
                    sg.Text(("Inference time (ms):"), visible=False),  
                    sg.Text("0", key="infer_time", visible=False),  
                ],  
            ]  
            self.window = sg.Window("VoiceAgent - Derek", layout=layout, finalize=True)  
            self.event_handler()  

        def event_handler(self):  
            global flag_vc  
            while True:  
                event, values = self.window.read()  
                if event == sg.WINDOW_CLOSED:  
                    self.stop_stream()  
                    exit()  # Stop the program on window close event  
                if event == "reload_devices" or event == "sg_hostapi":  
                    # Update devices when devices are reloaded or changed  
                    self.gui_config.sg_hostapi = values["sg_hostapi"]  
                    self.update_devices(hostapi_name=values["sg_hostapi"])  
                    if self.gui_config.sg_hostapi not in self.hostapis:  
                        self.gui_config.sg_hostapi = self.hostapis[0]  
                    self.window["sg_hostapi"].Update(values=self.hostapis)  
                    self.window["sg_hostapi"].Update(value=self.gui_config.sg_hostapi)  
                    if (  
                        self.gui_config.sg_input_device not in self.input_devices  
                        and len(self.input_devices) > 0  
                    ):  
                        self.gui_config.sg_input_device = self.input_devices[0]  
                    self.window["sg_input_device"].Update(values=self.input_devices)  
                    self.window["sg_input_device"].Update(  
                        value=self.gui_config.sg_input_device  
                    )  
                    if self.gui_config.sg_output_device not in self.output_devices:  
                        self.gui_config.sg_output_device = self.output_devices[0]  
                    self.window["sg_output_device"].Update(values=self.output_devices)  
                    self.window["sg_output_device"].Update(  
                        value=self.gui_config.sg_output_device  
                    )  
                if event == "start_vc" and not flag_vc:  
                    # Start voice conversion if it's not already running  
                    if self.set_values(values):  
                        printt("cuda_is_available: %s", torch.cuda.is_available())  
                        self.start_vc()  
                        settings = {  
                            "pth_path": values["pth_path"],  
                            "index_path": values["index_path"],  
                            "sg_hostapi": values["sg_hostapi"],  
                            "sg_wasapi_exclusive": values["sg_wasapi_exclusive"],  
                            "sg_input_device": values["sg_input_device"],  
                            "sg_output_device": values["sg_output_device"],  
                            "sr_type": ["sr_model", "sr_device"][  
                                [  
                                    values["sr_model"],  
                                    values["sr_device"],  
                                ].index(True)  
                            ],  
                            "threhold": values["threhold"],  
                            "pitch": values["pitch"],  
                            "rms_mix_rate": values["rms_mix_rate"],  
                            "index_rate": values["index_rate"],  
                            "block_time": values["block_time"],  
                            "crossfade_length": values["crossfade_length"],  
                            "extra_time": values["extra_time"],  
                            "n_cpu": values["n_cpu"],  
                            "use_jit": False,  
                            "use_pv": values["use_pv"],  
                            "f0method": ["pm", "harvest", "crepe", "rmvpe", "fcpe"][  
                                [  
                                    values["pm"],  
                                    values["harvest"],  
                                    values["crepe"],  
                                    values["rmvpe"],  
                                    values["fcpe"],  
                                ].index(True)  
                            ],  
                        }  
                        with open("configs/inuse/config.json", "w") as j:  
                            json.dump(settings, j)  
                        if self.stream is not None:  
                            self.delay_time = (  
                                self.stream.latency[-1]  
                                + values["block_time"]  
                                + values["crossfade_length"]  
                                + 0.01  
                            )  
                        if values["I_noise_reduce"]:  
                            self.delay_time += min(values["crossfade_length"], 0.04)  
                        self.window["sr_stream"].update(self.gui_config.samplerate)  
                        self.window["delay_time"].update(  
                            int(np.round(self.delay_time * 1000))  
                        )  

                # Parameter hot update  
                if event == "threhold":  
                    self.gui_config.threhold = values["threhold"]  
                elif event == "pitch":  
                    self.gui_config.pitch = values["pitch"]  
                    if hasattr(self, "rvc"):  
                        self.rvc.change_key(values["pitch"])  
                elif event == "formant":  
                    self.gui_config.formant = values["formant"]  
                    if hasattr(self, "rvc"):  
                        self.rvc.change_formant(values["formant"])  
                elif event == "index_rate":  
                    self.gui_config.index_rate = values["index_rate"]  
                    if hasattr(self, "rvc"):  
                        self.rvc.change_index_rate(values["index_rate"])  
                elif event == "rms_mix_rate":  
                    self.gui_config.rms_mix_rate = values["rms_mix_rate"]  
                elif event in ["pm", "harvest", "crepe", "rmvpe", "fcpe"]:  
                    self.gui_config.f0method = event  
                elif event == "I_noise_reduce":  
                    self.gui_config.I_noise_reduce = values["I_noise_reduce"]  
                    if self.stream is not None:  
                        self.delay_time += (  
                            1 if values["I_noise_reduce"] else -1  
                        ) * min(values["crossfade_length"], 0.04)  
                        self.window["delay_time"].update(  
                            int(np.round(self.delay_time * 1000))  
                        )  
                elif event == "O_noise_reduce":  
                    self.gui_config.O_noise_reduce = values["O_noise_reduce"]  
                elif event == "use_pv":  
                    self.gui_config.use_pv = values["use_pv"]  
                elif event in ["vc", "im"]:  
                    self.function = event  
                elif event == "stop_vc" or event != "start_vc":  
                    # Stop stream if voice conversion is stopped or any other event occurs  
                    self.stop_stream()  

        def set_values(self, values):  
            # Validate file paths  
            if len(values["pth_path"].strip()) == 0:  
                sg.popup(("Please choose the .pth file"))  
                return False  
            if len(values["index_path"].strip()) == 0:  
                sg.popup(("Please choose the .index file"))  
                return False  
            pattern = re.compile("[^\x00-\x7F]+")  # Regex to find non-ASCII characters  
            if pattern.findall(values["pth_path"]):  
                sg.popup(("pth file path cannot contain Chinese characters"))  
                return False  
            if pattern.findall(values["index_path"]):  
                sg.popup(("index file path cannot contain Chinese characters"))  
                return False  
            self.set_devices(values["sg_input_device"], values["sg_output_device"])  
            self.config.use_jit = False  # values["use_jit"]  
            self.gui_config.sg_hostapi = values["sg_hostapi"]  
            self.gui_config.sg_wasapi_exclusive = values["sg_wasapi_exclusive"]  
            self.gui_config.sg_input_device = values["sg_input_device"]  
            self.gui_config.sg_output_device = values["sg_output_device"]  
            self.gui_config.pth_path = values["pth_path"]  
            self.gui_config.index_path = values["index_path"]  
            self.gui_config.sr_type = ["sr_model", "sr_device"][  
                [  
                    values["sr_model"],  
                    values["sr_device"],  
                ].index(True)  
            ]  
            self.gui_config.threhold = values["threhold"]  
            self.gui_config.pitch = values["pitch"]  
            self.gui_config.formant = values["formant"]  
            self.gui_config.block_time = values["block_time"]  
            self.gui_config.crossfade_time = values["crossfade_length"]  
            self.gui_config.extra_time = values["extra_time"]  
            self.gui_config.I_noise_reduce = values["I_noise_reduce"]  
            self.gui_config.O_noise_reduce = values["O_noise_reduce"]  
            self.gui_config.use_pv = values["use_pv"]  
            self.gui_config.rms_mix_rate = values["rms_mix_rate"]  
            self.gui_config.index_rate = values["index_rate"]  
            self.gui_config.n_cpu = values["n_cpu"]  
            self.gui_config.f0method = ["pm", "harvest", "crepe", "rmvpe", "fcpe"][  
                [  
                    values["pm"],  
                    values["harvest"],  
                    values["crepe"],  
                    values["rmvpe"],  
                    values["fcpe"],  
                ].index(True)  
            ]  
            return True  

        def start_vc(self):  
            torch.cuda.empty_cache()  # Clear the GPU memory cache  
            self.rvc = rvc_for_realtime.RVC(  
                self.gui_config.pitch,  
                self.gui_config.formant,  
                self.gui_config.pth_path,  
                self.gui_config.index_path,  
                self.gui_config.index_rate,  
                self.gui_config.n_cpu,  
                inp_q,  
                opt_q,  
                self.config,  
                self.rvc if hasattr(self, "rvc") else None,  
            )  
            self.gui_config.samplerate = (  
                self.rvc.tgt_sr  
                if self.gui_config.sr_type == "sr_model"  
                else self.get_device_samplerate()  
            )  
            self.gui_config.channels = self.get_device_channels()  
            self.zc = self.gui_config.samplerate // 100  
            self.block_frame = (  
                int(  
                    np.round(  
                        self.gui_config.block_time  
                        * self.gui_config.samplerate  
                        / self.zc  
                    )  
                )  
                * self.zc  
            )  
            self.block_frame_16k = 160 * self.block_frame // self.zc  
            self.crossfade_frame = (  
                int(  
                    np.round(  
                        self.gui_config.crossfade_time  
                        * self.gui_config.samplerate  
                        / self.zc  
                    )  
                )  
                * self.zc  
            )  
            # Initialize buffer for SOLA algorithm  
            self.sola_buffer_frame = min(self.crossfade_frame, 4 * self.zc)  
            self.sola_search_frame = self.zc  
            self.extra_frame = (  
                int(  
                    np.round(  
                        self.gui_config.extra_time  
                        * self.gui_config.samplerate  
                        / self.zc  
                    )  
                )  
                * self.zc  
            )  
            # Initialize buffers for input, output and RMS  
            self.input_wav: torch.Tensor = torch.zeros(  
                self.extra_frame  
                + self.crossfade_frame  
                + self.sola_search_frame  
                + self.block_frame,  
                device=self.config.device,  
                dtype=torch.float32,  
            )  
            self.input_wav_denoise: torch.Tensor = self.input_wav.clone()  
            self.input_wav_res: torch.Tensor = torch.zeros(  
                160 * self.input_wav.shape[0] // self.zc,  
                device=self.config.device,  
                dtype=torch.float32,  
            )  
            self.rms_buffer: np.ndarray = np.zeros(4 * self.zc, dtype="float32")  
            self.sola_buffer: torch.Tensor = torch.zeros(  
                self.sola_buffer_frame, device=self.config.device, dtype=torch.float32  
            )  
            self.nr_buffer: torch.Tensor = self.sola_buffer.clone()  
            self.output_buffer: torch.Tensor = self.input_wav.clone()  
            self.skip_head = self.extra_frame // self.zc  
            self.return_length = (  
                self.block_frame + self.sola_buffer_frame + self.sola_search_frame  
            ) // self.zc  
            # Initialize fade windows for SOLA  
            self.fade_in_window: torch.Tensor = (  
                torch.sin(  
                    0.5  
                    * np.pi  
                    * torch.linspace(  
                        0.0,  
                        1.0,  
                        steps=self.sola_buffer_frame,  
                        device=self.config.device,  
                        dtype=torch.float32,  
                    )  
                )  
                ** 2  
            )  
            self.fade_out_window: torch.Tensor = 1 - self.fade_in_window  
            # Resampler for converting sample rate  
            self.resampler = tat.Resample(  
                orig_freq=self.gui_config.samplerate,  
                new_freq=16000,  
                dtype=torch.float32,  
            ).to(self.config.device)  
            # If target sample rate is different than input sample rate  
            if self.rvc.tgt_sr != self.gui_config.samplerate:  
                self.resampler2 = tat.Resample(  
                    orig_freq=self.rvc.tgt_sr,  
                    new_freq=self.gui_config.samplerate,  
                    dtype=torch.float32,  
                ).to(self.config.device)  
            else:  
                self.resampler2 = None  
            # Gate for noise reduction  
            self.tg = TorchGate(  
                sr=self.gui_config.samplerate, n_fft=4 * self.zc, prop_decrease=0.9  
            ).to(self.config.device)  
            # Start audio stream  
            self.start_stream()  

        def start_stream(self):  
            global flag_vc  
            if not flag_vc:  
                flag_vc = True  
                if (  
                    "WASAPI" in self.gui_config.sg_hostapi  
                    and self.gui_config.sg_wasapi_exclusive  
                ):  
                    extra_settings = sd.WasapiSettings(exclusive=True)  
                else:  
                    extra_settings = None  
                self.stream = sd.Stream(  
                    callback=self.audio_callback,  
                    blocksize=self.block_frame,  
                    samplerate=self.gui_config.samplerate,  
                    channels=self.gui_config.channels,  
                    dtype="float32",  
                    extra_settings=extra_settings,  
                )  
                self.stream.start()  

        def stop_stream(self):  
            global flag_vc  
            if flag_vc:  
                flag_vc = False  
                if self.stream is not None:  
                    self.stream.abort()  
                    self.stream.close()  
                    self.stream = None  

        def audio_callback(  
            self, indata: np.ndarray, outdata: np.ndarray, frames, times, status  
        ):  
            """  
            Audio processing and conversion callback  
            """  
            global flag_vc  
            start_time = time.perf_counter()  
            indata = librosa.to_mono(indata.T)  
            if self.gui_config.threhold > -60:  
                indata = np.append(self.rms_buffer, indata)  
                rms = librosa.feature.rms(  
                    y=indata, frame_length=4 * self.zc, hop_length=self.zc  
                )[:, 2:]  
                self.rms_buffer[:] = indata[-4 * self.zc :]  
                indata = indata[2 * self.zc - self.zc // 2 :]  
                db_threhold = (  
                    librosa.amplitude_to_db(rms, ref=1.0)[0] < self.gui_config.threhold  
                )  
                for i in range(db_threhold.shape[0]):  
                    if db_threhold[i]:  
                        indata[i * self.zc : (i + 1) * self.zc] = 0  
                indata = indata[self.zc // 2 :]  

            # Update buffers for incoming data  
            self.input_wav[: -self.block_frame] = self.input_wav[  
                self.block_frame :  
            ].clone()  
            self.input_wav[-indata.shape[0] :] = torch.from_numpy(indata).to(  
                self.config.device  
            )  
            self.input_wav_res[: -self.block_frame_16k] = self.input_wav_res[  
                self.block_frame_16k :  
            ].clone()  

            # Input noise reduction and resampling  
            if self.gui_config.I_noise_reduce:  
                self.input_wav_denoise[: -self.block_frame] = self.input_wav_denoise[  
                    self.block_frame :  
                ].clone()  
                input_wav = self.input_wav[-self.sola_buffer_frame - self.block_frame :]  
                # TODO: This section could be optimized by removing redundant operations  
                input_wav = self.tg(  
                    input_wav.unsqueeze(0), self.input_wav.unsqueeze(0)  
                ).squeeze(0)  
                input_wav[: self.sola_buffer_frame] *= self.fade_in_window  
                input_wav[: self.sola_buffer_frame] += (  
                    self.nr_buffer * self.fade_out_window  
                )  
                self.input_wav_denoise[-self.block_frame :] = input_wav[  
                    : self.block_frame  
                ]  
                self.nr_buffer[:] = input_wav[self.block_frame :]  
                self.input_wav_res[-self.block_frame_16k - 160 :] = self.resampler(  
                    self.input_wav_denoise[-self.block_frame - 2 * self.zc :]  
                )[160:]  
            else:  
                self.input_wav_res[-160 * (indata.shape[0] // self.zc + 1) :] = (  
                    self.resampler(self.input_wav[-indata.shape[0] - 2 * self.zc :])[  
                        160:  
                    ]  
                )  

            # Inference logic for conversion  
            if self.function == "vc":  
                infer_wav = self.rvc.infer(  
                    self.input_wav_res,  
                    self.block_frame_16k,  
                    self.skip_head,  
                    self.return_length,  
                    self.gui_config.f0method,  
                )  
                if self.resampler2 is not None:  
                    infer_wav = self.resampler2(infer_wav)  
            elif self.gui_config.I_noise_reduce:  
                infer_wav = self.input_wav_denoise[self.extra_frame :].clone()  
            else:  
                infer_wav = self.input_wav[self.extra_frame :].clone()  

            # Output noise reduction  
            if self.gui_config.O_noise_reduce and self.function == "vc":  
                self.output_buffer[: -self.block_frame] = self.output_buffer[  
                    self.block_frame :  
                ].clone()  
                self.output_buffer[-self.block_frame :] = infer_wav[-self.block_frame :]  
                infer_wav = self.tg(  
                    infer_wav.unsqueeze(0), self.output_buffer.unsqueeze(0)  
                ).squeeze(0)  

            # Volume envelope mixing  
            if self.gui_config.rms_mix_rate < 1 and self.function == "vc":  
                if self.gui_config.I_noise_reduce:  
                    input_wav = self.input_wav_denoise[self.extra_frame :]  
                else:  
                    input_wav = self.input_wav[self.extra_frame :]  
                rms1 = librosa.feature.rms(  
                    y=input_wav[: infer_wav.shape[0]].cpu().numpy(),  
                    frame_length=4 * self.zc,  
                    hop_length=self.zc,  
                )  
                rms1 = torch.from_numpy(rms1).to(self.config.device)  
                rms1 = F.interpolate(  
                    rms1.unsqueeze(0),  
                    size=infer_wav.shape[0] + 1,  
                    mode="linear",  
                    align_corners=True,  
                )[0, 0, :-1]  
                rms2 = librosa.feature.rms(  
                    y=infer_wav[:].cpu().numpy(),  
                    frame_length=4 * self.zc,  
                    hop_length=self.zc,  
                )  
                rms2 = torch.from_numpy(rms2).to(self.config.device)  
                rms2 = F.interpolate(  
                    rms2.unsqueeze(0),  
                    size=infer_wav.shape[0] + 1,  
                    mode="linear",  
                    align_corners=True,  
                )[0, 0, :-1]  
                rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-3)  
                infer_wav *= torch.pow(  
                    rms1 / rms2, torch.tensor(1 - self.gui_config.rms_mix_rate)  
                )  
            # SOLA algorithm modification (time domain synchronization)  
            conv_input = infer_wav[  
                None, None, : self.sola_buffer_frame + self.sola_search_frame  
            ]  
            cor_nom = F.conv1d(conv_input, self.sola_buffer[None, None, :])  
            cor_den = torch.sqrt(  
                F.conv1d(  
                    conv_input**2,  
                    torch.ones(1, 1, self.sola_buffer_frame, device=self.config.device),  
                )  
                + 1e-8  
            )  
            if sys.platform == "darwin":  
                _, sola_offset = torch.max(cor_nom[0, 0] / cor_den[0, 0])  
                sola_offset = sola_offset.item()  
            else:  
                sola_offset = torch.argmax(cor_nom[0, 0] / cor_den[0, 0])  
            printt("sola_offset = %d", int(sola_offset))  
            infer_wav = infer_wav[sola_offset:]  
            if "privateuseone" in str(self.config.device) or not self.gui_config.use_pv:  
                infer_wav[: self.sola_buffer_frame] *= self.fade_in_window  
                infer_wav[: self.sola_buffer_frame] += (  
                    self.sola_buffer * self.fade_out_window  
                )  
            else:  
                infer_wav[: self.sola_buffer_frame] = phase_vocoder(  
                    self.sola_buffer,  
                    infer_wav[: self.sola_buffer_frame],  
                    self.fade_out_window,  
                    self.fade_in_window,  
                )  
            self.sola_buffer[:] = infer_wav[  
                self.block_frame : self.block_frame + self.sola_buffer_frame  
            ]  
            outdata[:] = (  
                infer_wav[: self.block_frame]  
                .repeat(self.gui_config.channels, 1)  
                .t()  
                .cpu()  
                .numpy()  
            )  
            total_time = time.perf_counter() - start_time  
            if flag_vc:  
                self.window["infer_time"].update(int(total_time * 1000))  
            printt("Infer time: %.2f", total_time)  

        def update_devices(self, hostapi_name=None):  
            """Update and retrieve list of input/output devices for GUI"""  
            global flag_vc  
            flag_vc = False  
            sd._terminate()  
            sd._initialize()  
            devices = sd.query_devices()  # Query available audio devices  
            hostapis = sd.query_hostapis()  # Query available host APIs  
            for hostapi in hostapis:  
                for device_idx in hostapi["devices"]:  
                    devices[device_idx]["hostapi_name"] = hostapi["name"]  
            self.hostapis = [hostapi["name"] for hostapi in hostapis]  
            if hostapi_name not in self.hostapis:  
                hostapi_name = self.hostapis[0]  
            self.input_devices = [  
                d["name"]  
                for d in devices  
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name  
            ]  
            self.output_devices = [  
                d["name"]  
                for d in devices  
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name  
            ]  
            self.input_devices_indices = [  
                d["index"] if "index" in d else d["name"]  
                for d in devices  
                if d["max_input_channels"] > 0 and d["hostapi_name"] == hostapi_name  
            ]  
            self.output_devices_indices = [  
                d["index"] if "index" in d else d["name"]  
                for d in devices  
                if d["max_output_channels"] > 0 and d["hostapi_name"] == hostapi_name  
            ]  

        def set_devices(self, input_device, output_device):  
            """Set default audio input/output devices based on user selection"""  
            sd.default.device[0] = self.input_devices_indices[  
                self.input_devices.index(input_device)  
            ]  
            sd.default.device[1] = self.output_devices_indices[  
                self.output_devices.index(output_device)  
            ]  
            printt("Input device: %s:%s", str(sd.default.device[0]), input_device)  
            printt("Output device: %s:%s", str(sd.default.device[1]), output_device)  

        def get_device_samplerate(self):  
            """Get default sample rate of the input device"""  
            return int(  
                sd.query_devices(device=sd.default.device[0])["default_samplerate"]  
            )  

        def get_device_channels(self):  
            """Get the number of channels for input and output devices"""  
            max_input_channels = sd.query_devices(device=sd.default.device[0])[  
                "max_input_channels"  
            ]  
            max_output_channels = sd.query_devices(device=sd.default.device[1])[  
                "max_output_channels"  
            ]  
            return min(max_input_channels, max_output_channels, 2)  

    # Instantiate and run the GUI  
    gui = GUI()