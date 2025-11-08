import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import logging
import ipywidgets as ipw
from IPython.display import YouTubeVideo, IFrame, display

# Suppress font manager warnings
logging.getLogger('matplotlib.font_manager').disabled = True

# Configure high-DPI inline plots
%config InlineBackend.figure_format = 'retina'

# Apply custom matplotlib style
plt.style.use("https://raw.githubusercontent.com/NeuromatchAcademy/course-content/main/nma.mplstyle")

# Plotting Functions
def visualize_inverse_transfer(input_range, gain, threshold):
    fig, ax = plt.subplots()
    ax.plot(input_range, inverse_transfer(input_range, gain=gain, threshold=threshold))
    ax.set(xlabel="$x$", ylabel="$F^{-1}(x)$")

def visualize_ei_curves(input_range, exc_curve, inh_curve):
    plt.figure()
    plt.plot(input_range, exc_curve, color='blue', label='Excitatory Population')
    plt.plot(input_range, inh_curve, color='red', label='Inhibitory Population')
    plt.legend(loc='lower right')
    plt.xlabel('Input (a.u.)')
    plt.ylabel('F(Input)')
    plt.show()

def compare_populations(time_vec, exc1, inh1, exc2, inh2):
    fig, axes = plt.subplots(2, 1, figsize=(8, 6))
    axes[0].plot(time_vec, exc1, color='blue', label='Excitatory')
    axes[0].plot(time_vec, inh1, color='red', label='Inhibitory')
    axes[0].set_ylabel('Activity')
    axes[0].legend(loc='best')
    
    axes[1].plot(time_vec, exc2, color='blue', label='Excitatory')
    axes[1].plot(time_vec, inh2, color='red', label='Inhibitory')
    axes[1].set_xlabel('Time (ms)')
    axes[1].set_ylabel('Activity')
    axes[1].legend(loc='best')
    plt.tight_layout()
    plt.show()

def visualize_nullclines(exc_null_exc, exc_null_inh, inh_null_exc, inh_null_inh):
    plt.figure()
    plt.plot(exc_null_exc, exc_null_inh, color='blue', label='E Nullcline')
    plt.plot(inh_null_exc, inh_null_inh, color='red', label='I Nullcline')
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')
    plt.legend(loc='best')
    plt.show()

def generate_nullcline_plot(params):
    exc_range = np.linspace(-0.01, 0.96, 100)
    inh_range = np.linspace(-0.01, 0.8, 100)
    
    exc_null_inh = compute_e_nullcline(exc_range, **params)
    inh_null_exc = compute_i_nullcline(inh_range, **params)
    
    plt.figure()
    plt.plot(exc_range, exc_null_inh, color='blue', label='E Nullcline')
    plt.plot(inh_null_exc, inh_range, color='red', label='I Nullcline')
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')
    plt.legend(loc='best')
    plt.show()

def add_vector_field(params, skip_factor=2, vector_scale=5):
    grid_vals = np.linspace(0., 1., 20)
    exc_grid, inh_grid = np.meshgrid(grid_vals, grid_vals)
    
    d_exc_dt, d_inh_dt = compute_ei_derivatives(exc_grid, inh_grid, **params)
    
    plt.quiver(exc_grid[::skip_factor, ::skip_factor], inh_grid[::skip_factor, ::skip_factor],
               d_exc_dt[::skip_factor, ::skip_factor], d_inh_dt[::skip_factor, ::skip_factor],
               angles='xy', scale_units='xy', scale=vector_scale, facecolor='cyan')
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')

def plot_single_trajectory(params, trajectory_color, initial_state, trajectory_label):
    temp_params = params.copy()
    temp_params['rE_init'] = initial_state[0]
    temp_params['rI_init'] = initial_state[1]
    
    exc_traj, inh_traj = simulate_ei_network(**temp_params)
    
    plt.plot(exc_traj, inh_traj, color=trajectory_color, label=trajectory_label)
    plt.plot(initial_state[0], initial_state[1], marker='o', color=trajectory_color, markersize=8)
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')

def plot_multiple_trajectories(params, step_size, num_steps, label_text):
    temp_params = params.copy()
    for i in range(num_steps):
        for j in range(num_steps):
            temp_params['rE_init'] = step_size * i
            temp_params['rI_init'] = step_size * j
            exc_traj, inh_traj = simulate_ei_network(**temp_params)
            if i == num_steps - 1 and j == num_steps - 1:
                plt.plot(exc_traj, inh_traj, color='gray', alpha=0.8, label=label_text)
            else:
                plt.plot(exc_traj, inh_traj, color='gray', alpha=0.8)
    plt.xlabel(r'$r_E$')
    plt.ylabel(r'$r_I$')

def full_phase_portrait(params):
    plt.figure(figsize=(7.7, 6))
    plot_multiple_trajectories(params, 0.2, 6, 'Sample Trajectories\nfor Different Initial Conditions')
    plot_single_trajectory(params, 'orange', [0.6, 0.8], 'Sample Trajectory for\nLow Activity')
    plot_single_trajectory(params, 'magenta', [0.6, 0.6], 'Sample Trajectory for\nHigh Activity')
    generate_nullcline_plot(params)
    add_vector_field(params)
    plt.legend(loc=[1.02, 0.57], handlelength=1)

def mark_fixed_point(fixed_point, offset=(0.02, 0.1), angle=0):
    plt.plot(fixed_point[0], fixed_point[1], marker='o', color='black', markersize=8)
    plt.text(fixed_point[0] + offset[0], fixed_point[1] + offset[1],
             f'Fixed Point = \n({fixed_point[0]:.3f}, {fixed_point[1]:.3f})',
             horizontalalignment='center', verticalalignment='bottom', rotation=angle)

def get_standard_parameters(**overrides):
    params = {
        'tau_E': 1.0,         # E time constant [ms]
        'a_E': 1.2,           # E gain
        'theta_E': 2.8,       # E threshold
        'tau_I': 2.0,         # I time constant [ms]
        'a_I': 1.0,           # I gain
        'theta_I': 4.0,       # I threshold
        'wEE': 9.0,           # E to E weight
        'wEI': 4.0,           # I to E weight
        'wIE': 13.0,          # E to I weight
        'wII': 11.0,          # I to I weight
        'I_ext_E': 0.0,       # External input to E
        'I_ext_I': 0.0,       # External input to I
        'T': 50.0,            # Simulation duration [ms]
        'dt': 0.1,            # Time step [ms]
        'rE_init': 0.2,       # Initial E rate
        'rI_init': 0.2        # Initial I rate
    }
    
    for key, val in overrides.items():
        params[key] = val
    
    params['time_vec'] = np.arange(0, params['T'], params['dt'])
    return params

def sigmoid_activation(input_val, gain, threshold):
    return 1 / (1 + np.exp(-gain * (input_val - threshold))) - 1 / (1 + np.exp(gain * threshold))

def sigmoid_derivative(input_val, gain, threshold):
    exp_term = np.exp(-gain * (input_val - threshold))
    return gain * exp_term / (1 + exp_term)**2

class VideoPlayer(IFrame):
    def __init__(self, video_id, platform, page=1, width=400, height=300, **kwargs):
        self.video_id = video_id
        if platform == 'Bilibili':
            embed_src = f'https://player.bilibili.com/player.html?bvid={video_id}&page={page}'
        elif platform == 'Osf':
            embed_src = f'https://mfr.ca-1.osf.io/render?url=https://osf.io/download/{video_id}/?direct%26mode=render'
        super().__init__(embed_src, width=width, height=height, **kwargs)

def embed_video_collection(video_tuples, width=400, height=300, autoplay_flag=1):
    tab_outputs = []
    for idx, (platform, vid_id) in enumerate(video_tuples):
        output_widget = ipw.Output()
        with output_widget:
            if platform == 'Youtube':
                player = YouTubeVideo(id=vid_id, width=width, height=height, fs=autoplay_flag, rel=0)
                print(f'Video at https://youtube.com/watch?v={player.id}')
            else:
                player = VideoPlayer(id=vid_id, platform=platform, width=width, height=height, fs=autoplay_flag, autoplay=False)
                if platform == 'Bilibili':
                    print(f'Video at https://www.bilibili.com/video/{player.video_id}')
                elif platform == 'Osf':
                    print(f'Video at https://osf.io/{player.video_id}')
            display(player)
        tab_outputs.append(output_widget)
    return tab_outputs

def simulate_ei_network(tau_e, a_e, theta_e, tau_i, a_i, theta_i, wee, wei, wie, wii, i_ext_e, i_ext_i, r_e_start, r_i_start, dt, time_array, **extra_params):
    num_timesteps = len(time_array)
    exc_rates = np.append(r_e_start, np.zeros(num_timesteps - 1))
    inh_rates = np.append(r_i_start, np.zeros(num_timesteps - 1))
    ext_e_input = np.full(num_timesteps, i_ext_e)
    ext_i_input = np.full(num_timesteps, i_ext_i)
    
    for timestep in range(num_timesteps - 1):
        exc_update = (dt / tau_e) * (-exc_rates[timestep] + sigmoid_activation(wee * exc_rates[timestep] - wei * inh_rates[timestep] + ext_e_input[timestep], a_e, theta_e))
        inh_update = (dt / tau_i) * (-inh_rates[timestep] + sigmoid_activation(wie * exc_rates[timestep] - wii * inh_rates[timestep] + ext_i_input[timestep], a_i, theta_i))
        exc_rates[timestep + 1] = exc_rates[timestep] + exc_update
        inh_rates[timestep + 1] = inh_rates[timestep] + inh_update
    
    return exc_rates, inh_rates

def visualize_ei_different_starts(exc_start=0.0):
    params = get_standard_parameters(rE_init=exc_start, rI_init=0.15)
    exc_activity, inh_activity = simulate_ei_network(**params)
    
    plt.figure()
    plt.plot(params['time_vec'], exc_activity, color='blue', label='Excitatory Population')
    plt.plot(params['time_vec'], inh_activity, color='red', label='Inhibitory Population')
    plt.xlabel('Time (ms)')
    plt.ylabel('Activity')
    plt.legend(loc='best')
    plt.show()

def phase_activity_snapshot(time_idx):
    plt.figure(figsize=(8, 5.5))
    
    plt.subplot(211)
    plt.plot(params['time_vec'], exc_activity, color='blue', label=r'$r_E$')
    plt.plot(params['time_vec'], inh_activity, color='red', label=r'$r_I$')
    plt.plot(params['time_vec'][time_idx], exc_activity[time_idx], marker='o', color='blue')
    plt.plot(params['time_vec'][time_idx], inh_activity[time_idx], marker='o', color='red')
    plt.axvline(params['time_vec'][time_idx], 0, 1, color='black', linestyle='--')
    plt.xlabel('Time (ms)', fontsize=14)
    plt.ylabel('Activity', fontsize=14)
    plt.legend(loc='best', fontsize=14)
    
    plt.subplot(212)
    plt.plot(exc_activity, inh_activity, color='black')
    plt.plot(exc_activity[time_idx], inh_activity[time_idx], marker='o', color='black')
    plt.xlabel(r'$r_E$', fontsize=18, color='blue')
    plt.ylabel(r'$r_I$', fontsize=18, color='red')
    
    plt.tight_layout()
    plt.show()

def generate_fi_curves():
    params = get_standard_parameters()
    input_vals = np.arange(0, 10, 0.1)
    
    print(params['a_E'], params['theta_E'])
    print(params['a_I'], params['theta_I'])
    
    exc_fi = sigmoid_activation(input_vals, params['a_E'], params['theta_E'])
    inh_fi = sigmoid_activation(input_vals, params['a_I'], params['theta_I'])
    
    with plt.xkcd():
        visualize_ei_curves(input_vals, exc_fi, inh_fi)

def compute_ei_derivatives(exc_rates, inh_rates, tau_e, a_e, theta_e, wee, wei, i_ext_e, tau_i, a_i, theta_i, wie, wii, i_ext_i, **ignored):
    d_exc_dt = (-exc_rates + sigmoid_activation(wee * exc_rates - wei * inh_rates + i_ext_e, a_e, theta_e)) / tau_e
    d_inh_dt = (-inh_rates + sigmoid_activation(wie * exc_rates - wii * inh_rates + i_ext_i, a_i, theta_i)) / tau_i
    return d_exc_dt, d_inh_dt

def compute_e_nullcline(exc_vals, tau_e, a_e, theta_e, wee, wei, i_ext_e, tau_i, a_i, theta_i, wie, wii, i_ext_i, **ignored):
    input_to_e = wee * exc_vals + i_ext_e
    f_inv_e = inverse_transfer(exc_vals * tau_e, a_e, theta_e)  # Placeholder; adjust if needed
    return (input_to_e - f_inv_e) / wei

def compute_i_nullcline(inh_vals, tau_e, a_e, theta_e, wee, wei, i_ext_e, tau_i, a_i, theta_i, wie, wii, i_ext_i, **ignored):
    input_to_i = -wii * inh_vals + i_ext_i
    f_inv_i = inverse_transfer(inh_vals * tau_i, a_i, theta_i)
    return (f_inv_i - input_to_i) / wie  

def inverse_transfer(output_val, gain, threshold):
    return (1/gain) * np.log( (1 / (output_val + 1 / (1 + np.exp(gain * threshold)))) - 1 ) + threshold
