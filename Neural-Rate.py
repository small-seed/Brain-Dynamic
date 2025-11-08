import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
import logging
import ipywidgets as ipw
from IPython.display import YouTubeVideo, IFrame, display

# Suppress matplotlib font logging
logging.getLogger('matplotlib.font_manager').disabled = True

# Enable high-resolution inline plots
%config InlineBackend.figure_format = 'retina'

# Apply custom plotting style
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

def visualize_transfer_function(input_vals, output_vals):
    plt.figure(figsize=(6, 4))
    plt.plot(input_vals, output_vals, color='black')
    plt.xlabel('Input (a.u.)', fontsize=14)
    plt.ylabel('F(Input)', fontsize=14)
    plt.show()

def plot_rate_dynamics(rate_vals, rate_deriv, fixed_points=None):
    plt.figure()
    plt.plot(rate_vals, rate_deriv, color='black')
    plt.plot(rate_vals, np.zeros_like(rate_vals), color='black', linestyle='--')
    if fixed_points is not None:
        plt.plot(fixed_points, np.zeros_like(fixed_points), marker='o', color='black', markersize=12)
    plt.xlabel(r'Rate $r$')
    plt.ylabel(r'$\frac{dr}{dt}$', fontsize=20)
    plt.ylim(-0.1, 0.1)
    plt.show()

def plot_transfer_derivative(input_vals, deriv_vals):
    plt.figure()
    plt.plot(input_vals, deriv_vals, color='red')
    plt.xlabel('Input (a.u.)', fontsize=14)
    plt.ylabel('dF/dx', fontsize=14)
    plt.show()

class VideoEmbed(IFrame):
    def __init__(self, video_id, platform, page=1, width=400, height=300, **kwargs):
        self.video_id = video_id
        if platform == 'Bilibili':
            embed_url = f'https://player.bilibili.com/player.html?bvid={video_id}&page={page}'
        elif platform == 'Osf':
            embed_url = f'https://mfr.ca-1.osf.io/render?url=https://osf.io/download/{video_id}/?direct%26mode=render'
        else:
            raise ValueError(f"Unsupported platform: {platform}")
        super().__init__(embed_url, width=width, height=height, **kwargs)

def embed_videos(video_list, width=400, height=300, fullscreen=1):
    tab_outputs = []
    for idx, (platform, vid_id) in enumerate(video_list):
        output_area = ipw.Output()
        with output_area:
            if platform == 'Youtube':
                player = YouTubeVideo(id=vid_id, width=width, height=height, fs=fullscreen, rel=0)
                print(f'Video available at https://youtube.com/watch?v={player.id}')
            else:
                player = VideoEmbed(id=vid_id, platform=platform, width=width, height=height, fs=fullscreen, autoplay=False)
                if platform == 'Bilibili':
                    print(f'Video available at https://www.bilibili.com/video/{player.video_id}')
                elif platform == 'Osf':
                    print(f'Video available at https://osf.io/{player.video_id}')
            display(player)
        tab_outputs.append(output_area)
    return tab_outputs

def get_default_single_population_params(**overrides):
    params = {
        'tau': 1.0,       # Time constant [ms]
        'a': 1.2,         # Gain parameter
        'theta': 2.8,     # Threshold parameter
        'w': 0.0,         # Self-connection weight
        'I_ext': 0.0,     # External input
        'T': 20.0,        # Simulation duration [ms]
        'dt': 0.1,        # Time step [ms]
        'r_init': 0.2     # Initial rate
    } 
    params.update(overrides)
    params['time_vec'] = np.arange(0, params['T'], params['dt'])
    return params

def transfer_function(input_val, gain, thresh):
    sigmoid_shifted = 1 / (1 + np.exp(-gain * (input_val - thresh))) - 1 / (1 + np.exp(gain * thresh))
    return sigmoid_shifted

def interactive_transfer_plot(gain_val, thresh_val):
    input_range = np.arange(0, 10, 0.1)
    plt.figure()
    plt.plot(input_range, transfer_function(input_range, gain_val, thresh_val), color='black')
    plt.xlabel('Input (a.u.)', fontsize=14)
    plt.ylabel('F(Input)', fontsize=14)
    plt.show()

def run_single_population_simulation(params):
    time_const, gain, thresh = params['tau'], params['a'], params['theta']
    self_weight = params['w']
    ext_input = params['I_ext']
    init_rate = params['r_init']
    time_step, time_array = params['dt'], params['time_vec']
    num_steps = len(time_array)

    rates = np.zeros(num_steps)
    rates[0] = init_rate

    ext_input = np.full(num_steps, ext_input)

    for step in range(num_steps - 1):
        rate_change = (time_step / time_const) * (-rates[step] + transfer_function(self_weight * rates[step] + ext_input[step], gain, thresh))
        rates[step + 1] = rates[step] + rate_change
    
    return rates

def plot_excitatory_dynamics_diff_input_timeconst(ext_input_val, time_const_val, params):
    params['I_ext'] = ext_input_val
    params['tau'] = time_const_val

    sim_rates = run_single_population_simulation(params)

    steady_state = transfer_function(ext_input_val, params['a'], params['theta'])
    anal_rates = (params['r_init'] + (steady_state - params['r_init']) * (1 - np.exp(-params['time_vec'] / time_const_val)))

    plt.figure()
    plt.plot(params['time_vec'], sim_rates, color='blue', label=r'$r_{\mathrm{sim}}(t)$', alpha=0.5, zorder=1)
    plt.plot(params['time_vec'], anal_rates, color='blue', linestyle='--', linewidth=5, dashes=(2, 2),
             label=r'$r_{\mathrm{ana}}(t)$', zorder=2)
    plt.plot(params['time_vec'], steady_state * np.ones_like(params['time_vec']), color='black', linestyle='--',
             label=r'$F(I_{\mathrm{ext}})$')
    plt.xlabel('Time (ms)', fontsize=16)
    plt.ylabel('Rate r(t)', fontsize=16)
    plt.legend(loc='best', fontsize=14)
    plt.show()

def calculate_rate_derivative(rate, ext_input, self_weight, gain, thresh, time_const, **unused):
    deriv = (-rate + transfer_function(self_weight * rate + ext_input, gain, thresh)) / time_const
    return deriv

def find_fixed_point_single(rate_guess, gain, thresh, self_weight, ext_input, **unused):
    def residual_eq(rate_val):
        rate = rate_val
        drdt = (-rate + transfer_function(self_weight * rate + ext_input, gain, thresh))
        return np.array([drdt])
    
    initial_guess = np.array([rate_guess])
    fixed_pt = opt.root(residual_eq, initial_guess).x.item()
    return fixed_pt

def validate_fixed_point(fixed_pt_val, gain, thresh, self_weight, ext_input, tolerance=1e-4, **unused):
    residual = fixed_pt_val - transfer_function(self_weight * fixed_pt_val + ext_input, gain, thresh)
    return np.abs(residual) < tolerance

def discover_fixed_points(params, guess_rates, tolerance=1e-4):
    fixed_points = []
    for guess in guess_rates:
        fp_candidate = find_fixed_point_single(guess, **params)
        if validate_fixed_point(fp_candidate, **params, mytol=tolerance):
            fixed_points.append(fp_candidate)
    return fixed_points
