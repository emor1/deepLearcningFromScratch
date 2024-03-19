"""_summary_
JSONでモデルを出力するテスト


"""

import numpy as np
import json
import os, sys
sys.path.append(os.pardir)
from two_layer_net_backp import TwoLayerNet

if __name__ == "__main__":

    network_dict = {}
    network = TwoLayerNet(input_size=3, hidden_size=2, output_size=4)

    print(network.params)
    for key in ('W1', 'b1', 'W2', 'b2'):
        param = network.params[key]
        param_list = param.tolist()

        network_dict[key] = param_list

    print(network_dict.keys())

    with open('out.json', 'w') as f:
        json.dump(network_dict, f, indent=2)

    with open('out.json') as f:
        load_data = json.load(f)


    load_params = {}
    for key in ('W1', 'b1', 'W2', 'b2'):
        param = load_data[key]
        param_np = np.array(param)
        load_params[key] = param_np
    print(load_params)

    print(load_params['W1'] == network.params['W1'])
    # json_w1 = json.dumps({ i: e for i, e in enumerate(list_w1)})

