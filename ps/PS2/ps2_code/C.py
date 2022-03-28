import numpy as np
from matplotlib import pyplot as plt


ekf_path = 'ekf_out/'
pf_path = 'pf_out/'
D1_sensor_path = 'D1/D1_sensor_out/'
D1_motion_path = 'D1/D1_motion_out/'

input_f = 'input_data.npy'
output_f = 'output_data.npy'

def error_plots(path, input_f, output_f):

    with np.load(path + input_f) as data:
        actual_pose = data['real_robot_path']

    with np.load(path + output_f) as data:
        filter_estimated_pose = data['mean_trajectory']
        covariance = data['covariance_trajectory']
    
    pose_error = filter_estimated_pose - actual_pose

    t = np.arange(0, 20, 0.1)
    print(len(t))
    print(len(pose_error[:, 0]))

    fig, ax = plt.subplots()
    ax.plot(t, pose_error[:, 0], '.', color='b', label='x pose error')
    ax.plot(t, 3 * np.sqrt(covariance[0, 0, :]), 'r', label='+-3sigma')
    ax.plot(t, -3 * np.sqrt(covariance[0, 0, :]), 'r')
    ax.set_title('x pose error')
    ax.set_xlabel('t')
    ax.legend()
    ax.grid()
    plt.savefig(path+'x pose error.png')

    fig1, ax1 = plt.subplots()
    ax1.plot(t, pose_error[:, 1], '.', color='b', label='y pose error')
    ax1.plot(t, 3 * np.sqrt(covariance[1, 1, :]), 'r', label='+-3sigma')
    ax1.plot(t, -3 * np.sqrt(covariance[1, 1, :]), 'r')
    ax1.set_title('y pose error')
    ax1.set_xlabel('t')
    ax1.legend()
    ax1.grid()
    plt.savefig(path+'y pose error.png')

    fig2, ax2 = plt.subplots()
    ax2.plot(t, pose_error[:, 2], '.', color='b', label='theta pose error')
    ax2.plot(t, 3 * np.sqrt(covariance[2, 2, :]), 'r', label='+-3sigma')
    ax2.plot(t, -3 * np.sqrt(covariance[2, 2, :]), 'r')
    ax2.set_title('theta pose error')
    ax2.set_xlabel('t')
    ax2.legend()
    ax2.grid()
    plt.savefig(path+'theta pose error.png')

    plt.show()


if __name__ == '__main__':

   
    # error_plots(ekf_path, input_f, output_f)
    # error_plots(pf_path, input_f, output_f)

    # error_plots(D1_sensor_path, input_f, output_f)
    # error_plots(D1_motion_path, input_f, output_f)
    print('Done!')