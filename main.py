"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import subprocess
import argparse
from pythonosc import dispatcher
from pythonosc import osc_server
import random
import numpy as np
import h5py
from enum import Enum
import brain_model
hf = h5py.File('data.h5', 'w')

wave_counters = [0, 0, 0, 0, 0]
color = 0
dataset_counter = 0
p = subprocess.Popen(['feh', '/home/mgraves/Pictures/start.jpeg'])
a = np.zeros([100, 5, 3])


class Color(Enum):
    white = 0
    red = 1
    blue = 2


def guess_color():
    model = brain_model.load_existing_model('model.h5')
    data = a[:, :, 1:]
    return brain_model.make_prediction(model, data, max(wave_counters))


def change_color():
    global color
    global p
    if p.poll() is None:
        p.kill()
    color = random.choice([0, 1, 2])
    print('Color Set To: ' + Color(color).name)
    p = subprocess.Popen(["feh", "/home/mgraves/Pictures/"+Color(color).name+".jpeg"])


def append_data_to_file(datasetarray):
    data = datasetarray
    data = np.delete(data, -1, 0)
    print("Full dataset: ")
    print(data)
    global dataset_counter
    dataset_name = "dataset_" + str(dataset_counter)
    print("Dataset Saved As: " + dataset_name)
    hf.create_dataset(dataset_name, data=datasetarray)
    dataset_counter += 1


def print_decibels(unused_addr, args, bels):
    global color
    global a
    global wave_counters
    if wave_counters < [10, 10, 10, 10, 10]:
        try:
            a[wave_counters[args[0]], args[0]] = [color,  args[0], (bels*100)]
            wave_counters[args[0]] += 1
            if np.unique(wave_counters).size == 1 and wave_counters != [0, 0, 0, 0, 0]:
                color_guess = guess_color()
                print(Color(color_guess).name)
        except ValueError: pass
    else:
        wave_counters = [0, 0, 0, 0, 0]
        append_data_to_file(a)
        change_color()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default="0.0.0.0", help="The ip to listen on")
    parser.add_argument("--port", type=int, default=5000, help="The port to listen on")
    args = parser.parse_args()

    dispatcher = dispatcher.Dispatcher()
    # dispatcher.map("/muse/eeg", print)

    dispatcher.map("/muse/elements/alpha_absolute", print_decibels, 0)
    dispatcher.map("/muse/elements/beta_absolute", print_decibels, 1)
    dispatcher.map("/muse/elements/delta_absolute", print_decibels, 2)
    dispatcher.map("/muse/elements/gamma_absolute", print_decibels, 3)
    dispatcher.map("/muse/elements/theta_absolute", print_decibels, 4)
    change_color()
    server = osc_server.ThreadingOSCUDPServer(
          (args.ip, args.port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()