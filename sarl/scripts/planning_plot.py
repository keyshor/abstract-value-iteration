from util.io import parse_command_line_options
from util.io import extract_plot_data, plot_error_bar, save_plot

from matplotlib import pyplot as plt
import numpy as np

flags = parse_command_line_options()
itno = flags['itno']
folder = flags['folder']

x_max = 1e9
x_min = 0
y_index = -1
fname = 'planning'

dist_folder = folder + '/dist'
_, _, _, x = extract_plot_data(dist_folder, 0, 0, itno)
x = x[:3]
y, y_low, y_high, y_max = extract_plot_data(dist_folder, y_index, 0, itno)
Y = (y[:3]/100, y_low[:3]/100, y_high[:3]/100, y_max[:3]/100)
plot_error_bar(x, Y, 'blue', 'A-AVI')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

dist_folder = folder + '/planning/env{}'.format(flags['env_num'])
_, _, _, x0 = extract_plot_data(dist_folder, 0, 0, itno)
y, y_low, y_high, y_max = extract_plot_data(dist_folder, y_index, 0, itno)
Y = [y/100, y_low/100, y_high/100, y_max/100]
x = np.concatenate([x0, x[1:]])
for i in range(4):
    Y[i] = np.append([0], np.repeat(Y[i][1], len(x)-1))
plot_error_bar(x, Y, 'red', 'Transfer')
x_max = min(x_max, x[-1])
x_min = max(x_min, x[0])

plt.xlim(right=x_max)
save_plot(folder, '{}{}'.format(fname, flags['env_num']))
