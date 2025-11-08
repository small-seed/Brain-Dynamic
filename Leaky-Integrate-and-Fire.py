import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as ipw
from IPython.display import YouTubeVideo, IFrame, display
import logging

logging.getLogger('matplotlib.font_manager').disabled = True

%config InlineBackend.figure_format = 'retina'

plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

layout = ipw.Layout()

def visualize_membrane_potential(params, membrane_voltage, spike_times):
    threshold = params['V_th']
    time_step, time_array = params['dt'], params['range_t']
    
    if len(spike_times) > 0:
        spike_indices = (spike_times / time_step).astype(int) - 1
        membrane_voltage[spike_indices] += 20  # Enhance spike visibility
    
    plt.plot(time_array, membrane_voltage, color='blue')
    plt.axhline(y=threshold, xmin=0, xmax=1, color='black', linestyle='--')
    plt.xlabel('Time (ms)')
    plt.ylabel('Membrane Potential (mV)')
    plt.legend(['Membrane\nPotential', r'Threshold $V_{th}$'], loc=[1.05, 0.75])
    plt.ylim([-80, -40])
    plt.show()

def plot_noise_current(params, noise_current, membrane_voltage=None, spike_times=None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(params['range_t'][::3], noise_current[::3], color='blue')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(r'$I_{GWN}$ (pA)')
    
    if membrane_voltage is not None and spike_times is not None:
        visualize_membrane_potential(params, membrane_voltage, spike_times)
    else:
        axes[1].remove() 
    
    plt.tight_layout()
    plt.show()

def plot_isi_histograms(intervals1, intervals2, cv1, cv2, noise_std1, noise_std2):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    bin_edges = np.linspace(10, 30, 20)
    
    axes[0].hist(intervals1, bins=bin_edges, color='blue', alpha=0.5)
    axes[0].set_xlabel('ISI (ms)')
    axes[0].set_ylabel('Count')
    axes[0].set_title(r'$\sigma_{GWN}=%.1f$, CV$_{\mathrm{isi}}=%.3f$' % (noise_std1, cv1))
    
    axes[1].hist(intervals2, bins=bin_edges, color='blue', alpha=0.5)
    axes[1].set_xlabel('ISI (ms)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(r'$\sigma_{GWN}=%.1f$, CV$_{\mathrm{isi}}=%.3f$' % (noise_std2, cv2))
    
    plt.tight_layout()
    plt.show()

class VideoPlayer(IFrame):
    def __init__(self, video_id, platform, page=1, width=400, height=300, **kwargs):
        self.video_id = video_id
        if platform == 'Bilibili':
            src_url = f'https://player.bilibili.com/player.html?bvid={video_id}&page={page}'
        elif platform == 'Osf':
            src_url = f'https://mfr.ca-1.osf.io/render?url=https://osf.io/download/{video_id}/?direct%26mode=render'
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        super().__init__(src_url, width=width, height=height, **kwargs)

def show_videos(video_list, width=400, height=300, fullscreen=False):
    tab_widgets = []
    for idx, (platform, vid_id) in enumerate(video_list):
        output_widget = ipw.Output()
        with output_widget:
            if platform == 'Youtube':
                player = YouTubeVideo(id=vid_id, width=width, height=height, fs=fullscreen, rel=0)
                print(f'Video available at https://youtube.com/watch?v={player.id}')
            else:
                player = VideoPlayer(id=vid_id, platform=platform, width=width, height=height, fs=fullscreen, autoplay=False)
                if platform == 'Bilibili':
                    print(f'Video available at https://www.bilibili.com/video/{player.video_id}')
                elif platform == 'Osf':
                    print(f'Video available at https://osf.io/{player.video_id}')
            display(player)
        tab_widgets.append(output_widget)
    return tab_widgets

def get_default_parameters(**overrides):
    params = {
        'V_th': -55.,      # Spike threshold [mV]
        'V_reset': -75.,   # Reset potential [mV]
        'tau_m': 10.,      # Membrane time constant [ms]
        'g_L': 10.,        # Leak conductance [nS]
        'V_init': -75.,    # Initial potential [mV]
        'E_L': -75.,       # Leak reversal potential [mV]
        'tref': 2.,        # Refractory period [ms]
        'T': 400.,         # Simulation duration [ms]
        'dt': 0.1          # Time step [ms]
    }
    
    for key, value in overrides.items():
        params[key] = value
    
    params['range_t'] = np.arange(0, params['T'], params['dt'])
    return params

def simulate_lif_neuron(params, input_current, halt_simulation=False):
    threshold, reset = params['V_th'], params['V_reset']
    time_const, leak_cond = params['tau_m'], params['g_L']
    init_volt, leak_rev = params['V_init'], params['E_L']
    time_step, time_vec = params['dt'], params['range_t']
    total_steps = len(time_vec)
    ref_period = params['tref']
    
    voltage = np.zeros(total_steps)
    voltage[0] = init_volt
    
    if np.isscalar(input_current):
        input_current = np.full(total_steps, input_current)
    
    if halt_simulation:
        half_point = total_steps // 2
        input_current[:half_point - 1000] = 0
        input_current[half_point + 1000:] = 0
    
    spike_list = []
    ref_counter = 0.0 
    
    for step in range(total_steps - 1):
        if ref_counter > 0:  
            voltage[step] = reset
            ref_counter -= 1
        elif voltage[step] >= threshold: 
            spike_list.append(step)
            voltage[step] = reset
            ref_counter = ref_period / time_step
        else:
            voltage_change = (-(voltage[step] - leak_rev) + input_current[step] / leak_cond) * (time_step / time_const)
            voltage[step + 1] = voltage[step] + voltage_change
    
    spike_times = np.array(spike_list) * time_step
    return voltage, spike_times

def generate_gaussian_noise(params, mean_current, std_dev, fixed_seed=None):
    time_step, time_vec = params['dt'], params['range_t']
    num_steps = len(time_vec)
    
    if fixed_seed is not None:
        np.random.seed(fixed_seed)
    else:
        np.random.seed()
    
    noise = mean_current + std_dev * np.random.randn(num_steps) / np.sqrt(time_step / 1000.0)
    return noise

# Example: Neuron response to constant DC input 
"""
def simulate_dc_response(dc_current=200., membrane_tau=10.):
    params = get_default_parameters(T=100.)
    params['tau_m'] = membrane_tau
    voltage, spikes = simulate_lif_neuron(params, input_current=dc_current)
    visualize_membrane_potential(params, voltage, spikes)
    plt.show()
"""

# Example: Neuron response to Gaussian white noise 
"""
def simulate_gwn_response(mean_gwn, std_gwn):
    params = get_default_parameters(T=100.)
    gwn_current = generate_gaussian_noise(params, mean=mean_gwn, std_dev=std_gwn)
    voltage, spikes = simulate_lif_neuron(params, input_current=gwn_current)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(params['range_t'][::3], gwn_current[::3], color='blue')
    axes[0].set_xlabel('Time (ms)')
    axes[0].set_ylabel(r'$I_{GWN}$ (pA)')
    visualize_membrane_potential(params, voltage, spikes)
    plt.tight_layout()
    plt.show()
"""
