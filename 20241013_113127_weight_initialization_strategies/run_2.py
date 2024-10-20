import argparse
import json
import math
import os.path as osp
import pathlib
import pickle
import random
import struct

import matplotlib.pyplot as plt
import numpy as np
from simple_dataset import (
    generate_circle_data,
    generate_gaussian_data,
    generate_spiral_data,
    generate_xor_data,
)


class NeuralNetwork:
    def __init__(self, batch_size=1, act_func="sigmoid", learning_rate=0.2, init_strategy="default"):
        self.init_strategy = init_strategy
        self.MAX = 1050
        self.NMAX = 10
        self.batch_size = batch_size
        self.in_nodes = 0
        self.out_nodes = 0
        self.hidden_nodes = 0
        self.hidden_nodes2 = 0
        self.all_nodes = 0
        self.act_func = act_func
        self.pos_weights = 0
        self.neg_weights = 0
        self.pos_norms = 0
        self.neg_norms = 0
        self.dead_neurons = 0
        self.ot_in = [
            [0.0 for _ in range(self.MAX + 1)] for _ in range(self.NMAX + 1)
        ]
        self.ot_ot = [
            [0.0 for _ in range(self.MAX + 1)] for _ in range(self.NMAX + 1)
        ]
        self.del_ot = [
            [[0.0, 0.0] for _ in range(self.MAX + 1)]
            for _ in range(self.NMAX + 1)
        ]
        self.w_ot_ot = [
            [[0.0 for _ in range(self.MAX + 1)] for _ in range(self.MAX + 1)]
            for _ in range(self.NMAX + 1)
        ]
        self.w_running_avg = [
            [[0.0 for _ in range(self.MAX + 1)] for _ in range(self.MAX + 1)]
            for _ in range(self.NMAX + 1)
        ]
        self.ow = [0.0 for _ in range(self.MAX + 1)]

        # self.alpha = 0.8
        self.alpha = learning_rate
        self.beta = 0.8
        # self.u0 = 0.4
        self.u0 = 0.4
        self.u1 = 1.0
        self.inival1 = 1.0
        self.inival2 = 1.0
        self.t_loop = 2
        self.res = 0.0

        self.f = [0 for _ in range(15)]
        self.err = 0.0
        self.count = 0

    def sigmf(self, u):
        return 1 / (1 + math.exp(-2 * u / self.u0))

    def apply_derivative_sigmf(self, x):
        return x * (1 - x)

    def step_function(self, u):
        return 1 if u >= 0 else 0

    def silu(self, u):
        return u / (1 + math.exp(-u))

    def hard_sigmf(self, u):
        return max(0, min(1, 1 * u + 0.5))

    def apply_derivative_hard_sigmf(self, x):
        if x > -0.25 and x < 0.25:
            return 1
        else:
            return 0

    def shifted_relu(self, u):
        return max(0, u + 0.5)

    def apply_derivative_shifted_relu(self, x):
        return 1 if x > -0.5 else 0

    def relu(self, u):
        return max(0, u)

    def apply_derivative_relu(self, x):
        return 1 if x > 0 else 0

    def identity(self, u):
        return u

    def tanh(self, u):
        return math.tanh(u)

    def tanh_zero_one(self, u):
        return (math.tanh(u) + 1) / 2

    def apply_derivative_tanh(self, x):
        return 0.5 * (1 - math.tanh(x) ** 2)

    def leaky_relu(self, u):
        return max(0, u) + min(0, 0.01 * u)

    def apply_derivative_leaky_relu(self, x):
        return 1 if x > 0 else 0.01

    def relu_softplus(
        self, u
    ):  # awkward function, not sure why it doesn't converge.
        return max(0, u + math.log(2)) + min(
            math.log(2), math.log(1 + math.exp(u))
        )

    def softplus(self, u):
        return math.log(1 + math.exp(u))

    def apply_derivative_softplus(self, x):
        return 1 / (1 + math.exp(-x))

    def init_network(self, in_nodes, out_nodes, hidden_nodes, hidden_nodes2):
        self.in_nodes = in_nodes * 2
        self.out_nodes = out_nodes
        self.hidden_nodes = hidden_nodes + hidden_nodes2
        self.hidden_nodes2 = hidden_nodes2
        self.all_nodes = self.in_nodes + 1 + self.hidden_nodes

        if self.init_strategy == "xavier":
            limit = math.sqrt(6 / (self.in_nodes + self.hidden_nodes))
        elif self.init_strategy == "he":
            limit = math.sqrt(2 / self.in_nodes)
        elif self.init_strategy == "uniform":
            limit = 1.0
        elif self.init_strategy == "normal":
            limit = 0.1

        for n in range(self.out_nodes):
            if self.init_strategy == "xavier":
                self.ow = [random.uniform(-limit, limit) for _ in range(self.MAX + 1)]
            elif self.init_strategy == "he":
                self.ow = [random.gauss(0, limit) for _ in range(self.MAX + 1)]
            elif self.init_strategy == "uniform":
                self.ow = [random.uniform(-limit, limit) for _ in range(self.MAX + 1)]
            elif self.init_strategy == "normal":
                self.ow = [random.gauss(0, limit) for _ in range(self.MAX + 1)]
            else:
                self.ow = [0.0 for _ in range(self.MAX + 1)]
            for k in range(self.all_nodes + 2):
                self.ow[k] = 2 * ((k + 1) % 2) - 1
            self.ow[self.in_nodes + 2] = 1

            for k in range(self.in_nodes + 2, self.all_nodes + 2):
                for l in range(self.all_nodes + 2):
                    if l < 2:
                        self.w_ot_ot[n][k][l] = self.inival2 * random.random()
                    if l > 1:
                        self.w_ot_ot[n][k][l] = self.inival1 * random.random()
                    if (
                        k > self.all_nodes + 1 - self.hidden_nodes2
                        and 2 <= l < self.in_nodes + 2
                    ):
                        self.w_ot_ot[n][k][l] = 0
                    if (
                        self.f[6] == 1
                        and k != l
                        and k > self.in_nodes + 2
                        and l > self.in_nodes + 1
                    ):
                        self.w_ot_ot[n][k][l] = 0
                    if (
                        self.f[6] == 1
                        and k > self.in_nodes + 1
                        and self.in_nodes + 1 < l < self.in_nodes + 3
                    ):
                        self.w_ot_ot[n][k][l] = 0
                    if (
                        self.f[7] == 1
                        and 2 <= l < self.in_nodes + 2
                        and self.in_nodes + 2 <= k < self.in_nodes + 3
                    ):
                        self.w_ot_ot[n][k][l] = 0
                    if (
                        k > self.all_nodes + 1 - self.hidden_nodes2
                        and l >= self.in_nodes + 3
                    ):
                        self.w_ot_ot[n][k][l] = self.inival1 * random.random()
                    if k == l:
                        if self.f[3] == 1:
                            self.w_ot_ot[n][k][l] = 0
                        else:
                            self.w_ot_ot[n][k][l] = (
                                self.inival1 * random.random()
                            )
                    if (
                        self.f[11] == 0
                        and l < self.in_nodes + 2
                        and (l % 2) == 1
                    ):
                        self.w_ot_ot[n][k][l] = 0
                    self.w_ot_ot[n][k][l] *= self.ow[l] * self.ow[k]

            self.ot_in[n][0] = self.beta
            self.ot_in[n][1] = self.beta

        self.count = 0
        self.err = 0

    def calc(self, indata_input, indata_tch):
        self.output_calc(indata_input)
        self.teach_calc(indata_tch)
        # self.weight_calc()
        self.weight_calc()

    def output_calc(self, indata_input):
        for n in range(self.out_nodes):
            for k in range(2, self.in_nodes + 2):
                self.ot_in[n][k] = indata_input[(k // 2) - 1]
            if self.f[6]:
                for k in range(self.in_nodes + 2, self.all_nodes + 2):
                    self.ot_in[n][k] = 0

            for _ in range(self.t_loop):
                for k in range(self.in_nodes + 2, self.all_nodes + 2):
                    inival = sum(
                        self.w_ot_ot[n][k][m] * self.ot_in[n][m]
                        for m in range(self.all_nodes + 2)
                    )
                    if self.act_func == "sigmoid":
                        self.ot_ot[n][k] = self.sigmf(inival)
                    elif self.act_func == "tanh":
                        self.ot_ot[n][k] = self.tanh(inival)
                    elif self.act_func == "relu":
                        self.ot_ot[n][k] = self.relu(inival)
                    # self.ot_ot[n][k] = self.step_function(
                    #    inival
                    # )  # this didn't work.
                    # self.ot_ot[n][k] = self.silu(
                    #    inival
                    # )  # sometimes works but sometimes not, presumably because of negative values for u < 0
                    # self.ot_ot[n][k] = self.identity(
                    #     inival
                    # )  # blows up because of negative values for u < 0
                    # self.ot_ot[n][k] = self.hard_sigmf(inival)
                    # self.ot_ot[n][k] = self.tanh(inival)
                    # self.ot_ot[n][k] = self.relu(inival)
                    # self.ot_ot[n][k] = self.shifted_relu(inival)
                    # self.ot_ot[n][k] = self.relu_softplus(
                    #    inival
                    # )  # Need to investigate why this doesn't converge. Is it because something is wrong at u=0?
                    # self.ot_ot[n][k] = self.tanh_zero_one(inival) # converged.
                    # self.ot_ot[n][k] = self.leaky_relu(inival)
                    # self.ot_ot[n][k] = self.softplus(inival)
                    # As long as we use the derivative of sigmoid,
                    # and the forward act func is monotonically increasing,
                    # and positive for all x, it converges. -> this is only true for binary xor.
                    # e.g. original tanh and leaky ReLU are not good
                    # because it interferes with the negative values.
                for k in range(self.in_nodes + 2, self.all_nodes + 2):
                    self.ot_in[n][k] = self.ot_ot[n][k]
                # Count the number of dead neurons. Dead neurons are those that
                # never fire.
                if self.ot_ot[n][k] <= 0.1:
                    self.dead_neurons += 1

    def teach_calc(self, indata_tch):
        for l in range(self.out_nodes):
            wkb = indata_tch[l] - self.ot_ot[l][self.in_nodes + 2]
            self.err += abs(wkb)
            if abs(wkb) > 0.5:
                self.count += 1

            if wkb > 0:
                self.del_ot[l][self.in_nodes + 2][0] = wkb
                self.del_ot[l][self.in_nodes + 2][1] = 0
            else:
                self.del_ot[l][self.in_nodes + 2][0] = 0
                self.del_ot[l][self.in_nodes + 2][1] = -wkb

            inival1 = self.del_ot[l][self.in_nodes + 2][0]
            inival2 = self.del_ot[l][self.in_nodes + 2][1]

            for k in range(self.in_nodes + 3, self.all_nodes + 2):
                self.del_ot[l][k][0] = inival1 * self.u1
                self.del_ot[l][k][1] = inival2 * self.u1

    def weight_update(self):
        for n in range(self.out_nodes):
            for k in range(self.in_nodes + 2, self.all_nodes + 2):
                for m in range(self.all_nodes + 2):
                    self.w_ot_ot[n][k][m] += self.w_running_avg[n][k][m]
                    # print(self.w_running_avg[n][k][m])
                    # / self.w_ot_ot[n][k][m]
                    self.w_ot_ot[n][k][m] /= self.batch_size
                    # print(self.w_ot_ot[n][k][m])
                    # self.w_running_avg[n][k][m] = 0
                    # count the number of positive and negative weights
                    if self.w_ot_ot[n][k][m] > 0:
                        self.pos_weights += 1
                        # count the norms
                        self.pos_norms += self.w_ot_ot[n][k][m]
                    else:
                        self.neg_weights += 1
                        self.neg_norms += self.w_ot_ot[n][k][m]
        # print(
        #     f"Pos weights: {self.pos_weights}, Neg weights: {self.neg_weights}"
        # )
        # print(f"Pos norms: {self.pos_norms}, Neg norms: {self.neg_norms}")
        # print(f"Dead neurons: {self.dead_neurons}")
        self.pos_weights = 0
        self.neg_weights = 0
        self.pos_norms = 0
        self.neg_norms = 0
        self.dead_neurons = 0

    def weight_calc(self):
        for n in range(self.out_nodes):
            for k in range(self.in_nodes + 2, self.all_nodes + 2):
                for m in range(self.all_nodes + 2):
                    if self.w_ot_ot[n][k][m] != 0:
                        del_val = self.alpha * self.ot_in[n][m]
                        if self.act_func == "sigmoid":
                            del_val *= self.ot_ot[n][k] * (
                                1 - self.ot_ot[n][k]
                            )
                        elif self.act_func == "tanh":
                            del_val *= self.apply_derivative_tanh(
                                self.ot_ot[n][k]
                            )
                        elif self.act_func == "relu":
                            del_val *= self.apply_derivative_relu(
                                self.ot_ot[n][k]
                            )
                        # del_val *= self.apply_derivative_shifted_relu(
                        #    self.ot_ot[n][k]
                        # )
                        # del_val *= self.ot_ot[n][k] * (1 - self.ot_ot[n][k])
                        # print(self.ot_ot[n][k] * (1 - self.ot_ot[n][k]))
                        # del_val *= self.apply_derivative_hard_sigmf(
                        #    self.ot_ot[n][k]
                        # )
                        # del_val *= 0.1
                        # del_val *= self.apply_derivative_sigmf(
                        # del_val *= self.apply_derivative_relu(self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_tanh(self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_tanh(self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_leaky_relu(
                        #     self.ot_ot[n][k]
                        # )
                        # del_val *= self.apply_derivative_softplus(
                        #     self.ot_ot[n][k]
                        # )
                        if self.f[10] == 1:
                            # self.w_ot_ot[n][k][m] += (
                            #     del_val
                            #     * self.ow[k]
                            #     * (self.del_ot[n][k][0] - self.del_ot[n][k][1])
                            # )
                            self.w_running_avg[n][k][m] += (
                                del_val
                                * self.ow[k]
                                * (self.del_ot[n][k][0] - self.del_ot[n][k][1])
                            )
                        else:
                            if self.ow[m] > 0:
                                # self.w_ot_ot[n][k][m] += (
                                #     del_val
                                #     * self.del_ot[n][k][0]
                                #     * self.ow[m]
                                #     * self.ow[k]
                                # )
                                self.w_running_avg[n][k][m] += (
                                    del_val
                                    * self.del_ot[n][k][0]
                                    * self.ow[m]
                                    * self.ow[k]
                                )
                            else:
                                self.w_running_avg[n][k][m] += (
                                    del_val
                                    * self.del_ot[n][k][1]
                                    * self.ow[m]
                                    * self.ow[k]
                                )

    def weight_calc_original(self):
        for n in range(self.out_nodes):
            for k in range(self.in_nodes + 2, self.all_nodes + 2):
                for m in range(self.all_nodes + 2):
                    if self.w_ot_ot[n][k][m] != 0:
                        del_val = self.alpha * self.ot_in[n][m]
                        del_val *= self.ot_ot[n][k] * (1 - self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_relu(self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_shifted_relu(
                        #    self.ot_ot[n][k]
                        # )

                        # del_val *= self.ot_ot[n][k] * (1 - self.ot_ot[n][k])
                        # print(self.ot_ot[n][k] * (1 - self.ot_ot[n][k]))
                        # del_val *= self.apply_derivative_hard_sigmf(
                        #    self.ot_ot[n][k]
                        # )
                        # del_val *= 0.1
                        # del_val *= self.apply_derivative_sigmf(
                        # del_val *= self.apply_derivative_relu(self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_tanh(self.ot_ot[n][k])
                        # del_val *= self.apply_derivative_leaky_relu(
                        #     self.ot_ot[n][k]
                        # )
                        # del_val *= self.apply_derivative_softplus(
                        #     self.ot_ot[n][k]
                        # )
                        if self.f[10] == 1:
                            self.w_ot_ot[n][k][m] += (
                                del_val
                                * self.ow[k]
                                * (self.del_ot[n][k][0] - self.del_ot[n][k][1])
                            )
                        else:
                            if self.ow[m] > 0:
                                self.w_ot_ot[n][k][m] += (
                                    del_val
                                    * self.del_ot[n][k][0]
                                    * self.ow[m]
                                    * self.ow[k]
                                )
                            else:
                                self.w_ot_ot[n][k][m] += (
                                    del_val
                                    * self.del_ot[n][k][1]
                                    * self.ow[m]
                                    * self.ow[k]
                                )

    def eval(self, input_data, target_data):
        num_total = len(input_data)
        self.err = 0
        self.count = 0
        logits = []
        for indata_input, indata_tch in zip(input_data, target_data):
            self.output_calc(indata_input)
            wkb = indata_tch[0] - self.ot_ot[0][self.in_nodes + 2]
            logits.append(self.ot_ot[0][self.in_nodes + 2])
            self.err += abs(wkb)
            if abs(wkb) < 0.5:
                self.count += 1
        return 1.0 * self.err / num_total, 1.0 * self.count / num_total, logits

    def train(self, input_data, target_data, epochs):
        train_acc = []
        num_total = len(input_data)
        for epoch in range(epochs):
            self.err = 0
            self.count = 0
            # create mini batches
            mini_batches = [
                (
                    input_data[i : i + self.batch_size],
                    target_data[i : i + self.batch_size],
                )
                for i in range(0, len(input_data), self.batch_size)
            ]
            for inp_batch, tgt_batch in mini_batches:
                # print(inp_batch)
                # print(tgt_batch)
                for inp, tgt in zip(inp_batch, tgt_batch):
                    self.calc(inp, tgt)
                self.weight_update()

            print(f"Epoch {epoch + 1}, Error: {self.err}, Count: {self.count}")
            train_acc.append(1 - 1.0 * self.count / num_total)
        return train_acc


def float_to_binary(number, precision=16):
    sign = "-" if number < 0 else ""
    number = abs(number)

    int_part = int(number)
    frac_part = number - int_part

    int_bin = np.binary_repr(int_part)
    frac_bin = ""
    for _ in range(precision):
        frac_part *= 2
        bit = int(frac_part)
        frac_bin += str(bit)
        frac_part -= bit

    if frac_bin:
        bin_rep = f"{sign}{int_bin}.{frac_bin}"
    else:
        bin_rep = f"{sign}{int_bin}"

    return bin_rep


def half_to_binstr(half):
    (bits,) = struct.unpack("!H", struct.pack("!e", half))
    return "{:016b}".format(bits)


def convert_to_binary(input_data):
    new_format_input_data = []
    for sample in input_data:
        new_format = [int(c) for c in half_to_binstr(sample[0])]
        new_format.extend([int(c) for c in half_to_binstr(sample[1])])
        new_format_input_data.append(new_format)
    return new_format_input_data


def plot_2d_decision_boundary(
    nn,
    x,
    y,
    show_plot,
    filename,
    test_acc,  # train_acc
):
    # Find the x1 range and x2 range
    # determine the range by calculating the minimum and maximum values of x1 and x2
    x1_range = np.linspace(np.min(x[:, 0]) - 1, np.max(x[:, 0]) + 1, 100)
    x2_range = np.linspace(np.min(x[:, 1]) - 1, np.max(x[:, 1]) + 1, 100)
    x1, x2 = np.meshgrid(x1_range, x2_range)
    x_2d = np.c_[x1.ravel(), x2.ravel()]
    x_2d = x_2d.tolist()
    _, _, pred = nn.eval(x_2d, np.zeros(len(x_2d)).reshape(-1, 1).tolist())
    # pred = pred.squeeze()

    pred = np.array(pred).reshape(x1.shape)

    # Plot the decision boundary
    plt.figure(figsize=(8, 8))
    plt.title(f"Test Acc: {test_acc:.2f}")  # , Train Acc: {train_acc:.2f}")
    plt.contourf(
        x1,
        x2,
        # np.round(pred),
        pred,
        alpha=0.3,
        cmap="bwr",  # levels=levels, extend="both"
    )
    plt.scatter(x[:, 0], x[:, 1], c=y, alpha=0.8, cmap="bwr")
    if show_plot:
        plt.show()
    else:
        plt.savefig(filename)
        plt.close()

    return {
        "pred": pred,
        "x1": x1,
        "x2": x2,
        "x": x,
        "y": y,
        "test_acc": test_acc,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--num_epochs", type=int, default=40)
    parser.add_argument("--hidden_size", type=int, default=8)
    parser.add_argument("--binary_repr", action="store_true", default=False)
    parser.add_argument("--act_func", type=str, default="sigmoid")
    parser.add_argument("--out_dir", type=str, default="run_0")
    parser.add_argument("--init_strategy", type=str, default="he", choices=["default", "xavier", "he", "uniform", "normal"])

    data_gen_func = {
        "gaussian": generate_gaussian_data,
        "xor": generate_xor_data,
        "circle": generate_circle_data,
        "spiral": generate_spiral_data,
    }
    config = parser.parse_args()

    pathlib.Path(config.out_dir).mkdir(parents=True, exist_ok=True)

    final_infos = {}
    all_results = {}
    for dataset_name in ["gaussian", "xor", "circle", "spiral"]:
        for act_func in ["sigmoid"]:  # , "tanh", "relu"]:
            nn = NeuralNetwork(
                batch_size=config.train_batch_size,
                act_func=act_func,
                learning_rate=config.learning_rate,
                init_strategy=config.init_strategy
            )
            nn.f[6] = 1  # Set some flags as in the original C code
            nn.f[7] = 1
            nn.f[3] = 1
            nn.f[11] = 1
            if config.binary_repr:
                in_nodes = 32
            else:
                in_nodes = 2
            nn.init_network(
                in_nodes=in_nodes,
                out_nodes=1,
                hidden_nodes=config.hidden_size,
                hidden_nodes2=0,
            )

            # data = simple_dataset.generate_xor_data(400)  # this works
            n_samples = 400
            data = data_gen_func[dataset_name](n_samples)  # this works
            permute_indices = np.random.permutation(len(data))
            data = data[permute_indices]

            train_data = data[: int(n_samples * 0.5)]
            eval_data = data[int(n_samples * 0.5) :]
            input_data = np.array(train_data[:, :2], dtype=np.float16)
            target_data = train_data[:, 2].reshape(-1, 1)
            # convert to list of lists
            input_data = input_data.tolist()
            if config.binary_repr:
                input_data = convert_to_binary(input_data)
            target_data = target_data.tolist()

            train_accuracies = nn.train(
                input_data, target_data, epochs=config.num_epochs
            )

            eval_input_data = np.array(eval_data[:, :2], dtype=np.float16)
            eval_target_data = eval_data[:, 2].reshape(-1, 1)
            if config.binary_repr:
                eval_input_data = convert_to_binary(eval_input_data)
            err, acc, eval_logits = nn.eval(eval_input_data, eval_target_data)
            print(f"Error: {err}, Accuracy: {acc}")

            # Check how many overlaps there are between train_data and eval_data
            if config.binary_repr:
                overlap = 0
                for i in range(len(eval_input_data)):
                    if eval_input_data[i] in input_data:
                        overlap += 1
                print(f"Overlap: {overlap} out of {len(eval_data)}")

            # get info to plot decision boundaries
            plot_info = plot_2d_decision_boundary(
                nn, eval_input_data, eval_target_data, False, "test.png", acc
            )
            final_infos[dataset_name + "_" + config.act_func] = {
                "means": {
                    "eval_loss": err,
                    "accuracy": acc,
                    "train_accuracies": train_accuracies,
                }
            }

            all_results[dataset_name + "_" + config.act_func] = {
                "plot_info": plot_info,
                "accuracy": acc,
                "train_accuracies": train_accuracies,
            }
    with open(osp.join(config.out_dir, "final_info.json"), "w") as f:
        json.dump(final_infos, f)

    with open(osp.join(config.out_dir, "all_results.pkl"), "wb") as f:
        pickle.dump(all_results, f)
