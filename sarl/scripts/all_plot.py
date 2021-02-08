from util.io import parse_command_line_options
from util.io import extract_plot_data, plot_error_bar, save_plot

from matplotlib import pyplot as plt

flags = parse_command_line_options()
itno = flags['itno']
folder = flags['folder']
sample_plot = flags['not_baseline']
start_itno = flags['baseline_num']
if sample_plot:
    hiro_x = -2
    x_index = 0
    fname = 'sample'
else:
    hiro_x = 0
    x_index = 1
    fname = 'time'

x_max = 1e9
x_min = 0

dist_folder = folder + '/dist'
_, _, _, x = extract_plot_data(dist_folder, x_index, start_itno, itno)
y, y_low, y_high, y_max = extract_plot_data(dist_folder, -1, start_itno, itno)
Y = (y/100, y_low/100, y_high/100, y_max/100)
plot_error_bar(x, Y, 'blue', 'A-AVI (Ours)')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

no_dist_folder = folder + '/no_dist'
_, _, _, x = extract_plot_data(no_dist_folder, x_index, start_itno, itno)
y, y_low, y_high, y_max = extract_plot_data(
    no_dist_folder, -1, start_itno, itno)
Y = (y/100, y_low/100, y_high/100, y_max/100)
plot_error_bar(x, Y, 'mediumturquoise', 'Ablation')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

hiro_folder = folder + '/hiro'
_, _, _, x = extract_plot_data(
    hiro_folder, hiro_x, start_itno, itno, csv=True)
x = x[2:]
if hiro_x == 0:
    x = (x - x[0])/60
y, y_low, y_high, y_max = extract_plot_data(
    hiro_folder, -1, start_itno, itno, csv=True)
Y = (y[2:], y_low[2:], y_high[2:], y_max[2:])
plot_error_bar(x, Y, 'tomato', 'HIRO')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

spectrl_folder = folder + '/spectrl'
_, _, _, x = extract_plot_data(spectrl_folder, x_index, start_itno, itno)
y, y_low, y_high, y_max = extract_plot_data(
    spectrl_folder, -1, start_itno, itno)
Y = (y[::150]/100, y_low[::150]/100, y_high[::150]/100, y_max[::150]/100)
plot_error_bar(x[::150], Y, 'seagreen', 'SPECTRL')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

plt.xlim(right=x_max, left=-0.01)
save_plot(folder, '{}{}'.format(fname, flags['env_num']))
