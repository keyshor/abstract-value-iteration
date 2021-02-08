import os
import cv2
import sys
import getopt
import pickle
import numpy as np

from matplotlib import pyplot as plt
from PIL import Image
from numpy import genfromtxt


def parse_command_line_options():
    optval = getopt.getopt(sys.argv[1:], 'n:d:e:a:r:p:l:qgbo', [])
    itno = -1
    folder = ''
    env_num = 0
    eval_mode = False
    rl_algo = 'ars'
    parallel = 0
    gpu_flag = False
    not_baseline = True
    load_folder = ''
    baseline_num = 0
    obstacle = False
    for option in optval[0]:
        if option[0] == '-n':
            itno = int(option[1])
        if option[0] == '-d':
            folder = option[1]
        if option[0] == '-e':
            env_num = int(option[1])
        if option[0] == '-q':
            eval_mode = True
        if option[0] == '-a':
            rl_algo = option[1]
        if option[0] == '-g':
            gpu_flag = True
        if option[0] == '-b':
            not_baseline = False
        if option[0] == '-r':
            baseline_num = int(option[1])
        if option[0] == '-p':
            parallel = int(option[1])
        if option[0] == '-l':
            load_folder = option[1]
        if option[0] == '-o':
            obstacle = True
    print('**** Command Line Options ****')
    print('Run number: {}'.format(itno))
    print('Folder to save data: {}'.format(folder))
    flags = {'itno': itno,
             'folder': folder,
             'env_num': env_num,
             'eval_mode': eval_mode,
             'alg': rl_algo,
             'parallel': parallel,
             'gpu_flag': gpu_flag,
             'not_baseline': not_baseline,
             'baseline_num': baseline_num,
             'load_folder': load_folder,
             'obstacle': obstacle}
    return flags


#  open a log file to periodically flush data
def open_log_file(itno, folder):
    fname = _get_prefix(folder) + 'log' + _get_suffix(itno) + '.txt'
    open(fname, 'w').close()
    file = open(fname, 'a')
    return file


# save object
# name: str
# object: any object
def save_object(name, object, itno, folder):
    file = open(_get_prefix(folder) + name + _get_suffix(itno) + '.pkl', 'wb')
    pickle.dump(object, file)
    file.close()


# load object
# name: str
def load_object(name, itno, folder):
    file = open(_get_prefix(folder) + name + _get_suffix(itno) + '.pkl', 'rb')
    object = pickle.load(file)
    file.close()
    return object


# save log_info
def save_log_info(log_info, itno, folder):
    np.save(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.npy', log_info)


# load log_info
def load_log_info(itno, folder, csv=False):
    if csv:
        return genfromtxt(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.csv', delimiter=',')
    else:
        return np.load(_get_prefix(folder) + 'log' + _get_suffix(itno) + '.npy')


# dir to store video images
def get_image_dir(itno, folder):
    image_dir = '{}img{}'.format(_get_prefix(folder), _get_suffix(itno))
    if os.path.exists(image_dir) is False:
        os.mkdir(image_dir)
    return image_dir


# generate video
def generate_video(env, policy, itno, folder, max_step=10000):
    image_dir = get_image_dir(itno, folder)

    done = False
    state = env.reset()
    step = 0
    while not done:
        img_arr = env.render(mode='rgb_array')
        img = Image.fromarray(img_arr)
        img.save(image_dir + '/' + str(step) + '.png')
        action = policy.get_action(state)
        state, _, done, _ = env.step(action)
        step += 1
        if step > max_step:
            done = True

    video_name = image_dir + '/' + 'video.avi'
    images_temp = [img for img in os.listdir(image_dir)]
    images = []
    for i in range(len(images_temp)):
        for j in images_temp:
            directory = str(i) + '.png'
            if directory == j:
                images.append(j)

    frame = cv2.imread(os.path.join(image_dir, images_temp[0]))
    height, width, _ = frame.shape
    video = cv2.VideoWriter(
        video_name, cv2.VideoWriter_fourcc(*'XVID'), 20, (width, height))

    for image in images:
        video.write(cv2.imread(os.path.join(image_dir, image)))

    cv2.destroyAllWindows()
    video.release()

# plot the error bar from the data
#
# samples_per_iter: int (number of sample rollouts per iteration of the algorithm)
# data: (3+)-tuple of np.array (curve, lower error bar, upper error bar, ...)
# color: color of the plot
# label: string


def plot_error_bar(x, data, color, label):
    plt.subplots_adjust(bottom=0.126)
    plt.rcParams.update({'font.size': 18})
    plt.plot(x, data[0], color=color, label=label)
    plt.fill_between(x, data[1], data[2], color=color, alpha=0.15)


# load and parse log_info to generate error bars
#
# folder: string (name of folder)
# column_num: int (column number in log.npy to use)
# l: int (lower limit on run number)
# u: int (upper limit on run number)
# returns 4-tuple of np.array (curve, lower error bar, upper error bar, max_over_runs)
def extract_plot_data(folder, column_num, low, up, csv=False):
    log_infos = []
    min_length = 1000000
    for itno in range(low, up):
        log_info = np.transpose(load_log_info(
            itno, folder, csv=csv))[column_num]
        log_info = np.append([0], log_info)
        min_length = min(min_length, len(log_info))
        log_infos.append(log_info)
    log_infos = [log_info[:min_length] for log_info in log_infos]
    data = np.array(log_infos)
    curve = np.mean(data, axis=0)
    std = 0.5 * np.std(data, axis=0)
    max_curve = np.amax(data, axis=0)
    return curve, (curve - std), (curve + std), max_curve


# save and render current plot
def save_plot(folder, name, show=True):
    plt.rcParams.update({'font.size': 14})
    plt.legend()
    ax = plt.gca()
    ax.xaxis.major.formatter._useMathText = True
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0, 0))
    plt.savefig(_get_prefix(folder) + name + '.pdf', format='pdf')
    if show:
        plt.show()


# get prefix for file name
def _get_prefix(folder):
    if folder == '':
        return ''
    else:
        return folder + '/'


# get suffix from itno
def _get_suffix(itno):
    if itno < 0:
        return ''
    else:
        return str(itno)
