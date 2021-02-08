from util.io import parse_command_line_options
from util.io import extract_plot_data, plot_error_bar, save_plot

from matplotlib import pyplot as plt

flags = parse_command_line_options()
itno = flags['itno']
folder = flags['folder']
if flags['not_baseline']:
    y_index = -1
    fname = 'prob'
else:
    y_index = -2
    fname = 'reward'

x_max = 1e9
x_min = 0

dist_folder = folder + '/dist'
_, _, _, x = extract_plot_data(dist_folder, 0, 0, itno)
y, y_low, y_high, y_max = extract_plot_data(dist_folder, y_index, 0, itno)
Y = (y/100, y_low/100, y_high/100, y_max/100)
plot_error_bar(x, Y, 'blue', 'Doorways')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

dist1_folder = folder + '/dist1'
_, _, _, x = extract_plot_data(dist1_folder, 0, 0, itno)
y, y_low, y_high, y_max = extract_plot_data(dist1_folder, y_index, 0, itno)
Y = (y/100, y_low/100, y_high/100, y_max/100)
plot_error_bar(x, Y, 'violet', 'Full Rooms')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

dist2_folder = folder + '/dist2'
_, _, _, x = extract_plot_data(dist2_folder, 0, 0, itno)
y, y_low, y_high, y_max = extract_plot_data(dist2_folder, y_index, 0, itno)
Y = (y/100, y_low/100, y_high/100, y_max/100)
plot_error_bar(x, Y, 'deepskyblue', 'Room Centers')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

plt.xlim(right=x_max)
save_plot(folder, 'abstract_states_{}{}'.format(fname, flags['env_num']))
