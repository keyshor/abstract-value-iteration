from util.io import parse_command_line_options
from util.io import extract_plot_data, plot_error_bar, save_plot

flags = parse_command_line_options()
itno = flags['itno']
folder = flags['folder']
alg = flags['alg']
start_itno = flags['env_num']
if flags['not_baseline']:
    y_index = -1
else:
    y_index = -2
x_index = 0

_, _, _, x = extract_plot_data(folder, x_index, start_itno, itno)
y, y_low, y_high, y_max = extract_plot_data(folder, y_index, start_itno, itno)
Y = (y/100, y_low/100, y_high/100, y_max/100)
plot_error_bar(x, Y, 'blue', alg)

save_plot(folder, 'learning_curve_{}'.format(alg))
