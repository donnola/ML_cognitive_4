#include <iostream>
#include <fstream>
#include <string>
#include <jsoncpp/json/json.h>

using namespace std;

int main(int argc, char* argv[]) {
    if (argc < 2) {
        cout << "No arguments" << endl;
        return 0;
    }
    ifstream layers_file(argv[1]);
//    ifstream layers_file("nn_example.json");
    if (!layers_file.is_open()) {
        cout << "No such file" << endl;
        return 0;
    }
    Json::Reader reader;
    Json::Value layers;
    reader.parse(layers_file, layers);

    Json::Value layer0 = layers["layer0"]["data_shape"];
    vector<vector<int>> data_shape;
    vector<int> layer_data_shape(3);
    int sum_num_params = 0;
    int sum_num_add = 0;
    int sum_num_mul = 0;
    for (int i = 0; i < 3; ++i) {
        int shape_i = layer0.get(char(i), -1).asInt();
        layer_data_shape[i] = shape_i;
    }
    data_shape.emplace_back(layer_data_shape);
    cout << "layer0 input: ";
    for (int n : data_shape[0]){
        cout << n << " ";
    }
    cout << "\n\n";

    for (int i = 1; ; ++i) {
        string layer_name = "layer" + to_string(i);
        if (!layers.isMember(layer_name)) {
            break;
        }
        string layer_i = layers[layer_name]["type"].asString();
        if (layer_i == "conv") {
            cout << layer_name << " type: " << layer_i << "\n";
            int filters;
            vector<int> filter_size;
            vector<int> filter_stride;
            vector<int> padding;
            for (int j = 0; j < 2; ++j) {
                filter_size.push_back(layers[layer_name]["filter_size"].get(char(j), -1).asInt());
                filter_stride.push_back(layers[layer_name]["filter_stride"].get(char(j), -1).asInt());
                padding.push_back(layers[layer_name]["padding"].get(char(j), -1).asInt());
            }
            filters = layers[layer_name]["filters"].asInt();
            int x = (data_shape[i-1][0] - filter_size[0] + 2 * padding[0]) / filter_stride[0] + 1;
            int y = (data_shape[i-1][1] - filter_size[1] + 2 * padding[1]) / filter_stride[1] + 1;
            layer_data_shape[0] = x;
            layer_data_shape[1] = y;
            layer_data_shape[2] = filters;

            int params_num = filters * (filter_size[0] * filter_size[1] * data_shape[i-1][2] + 1);
            sum_num_params += params_num;
            cout << "number of parameters: " << params_num << "\n";

            int window_size = filter_size[0] * filter_size[1];
            int window_entry_num = x * y;
            int add_num = (window_size /*- 1 */) * window_entry_num * data_shape[i-1][2] * filters +
                    window_entry_num * (data_shape[i-1][2] /*- 1 */) * filters;
            cout << "number of add operation: " << add_num << "\n";
            int mul_num = window_size * window_entry_num * data_shape[i-1][2] * filters;
            cout << "number of mul operation: " << mul_num << "\n";

            sum_num_add += add_num;
            sum_num_mul += mul_num;

            data_shape.emplace_back(layer_data_shape);
        }
        else if (layer_i == "neuron") {
            int use_num = layer_data_shape[0] * layer_data_shape[1] * layer_data_shape[2];
            cout << layer_name << " type: " << layer_i << "\n";
            cout << layer_i << " function: " << layers[layer_name][layer_i].asString() << "\n";
            cout << "number of uses of the function: " << use_num << "\n";
            data_shape.emplace_back(layer_data_shape);
        }
        else if (layer_i == "fc") {
            cout << layer_name << " type: " << layer_i << "\n";
            int outputs = layers[layer_name]["outputs"].asInt();
            layer_data_shape[0] = 1;
            layer_data_shape[1] = 1;
            layer_data_shape[2] = outputs;

            int params_num = (data_shape[i-1][2] + 1) * outputs;
            sum_num_params += params_num;
            cout << "number of parameters: " << params_num << "\n";

            int add_num = outputs * data_shape[i-1][0] * data_shape[i-1][1] * data_shape[i-1][2];
            cout << "number of add operation: " << add_num << "\n";
            int mul_num = outputs * data_shape[i-1][0] * data_shape[i-1][1] * data_shape[i-1][2];
            cout << "number of mul operation: " << mul_num << "\n";

            sum_num_add += add_num;
            sum_num_mul += mul_num;

            data_shape.emplace_back(layer_data_shape);
        }

        cout << "output: ";
        for (int n : data_shape[i]){
            cout << n << " ";
        }
        cout << "\n\n";

    }
    cout << "total number of trainable parameters: " << sum_num_params << "\n";
    cout << "total number of add operation: " << sum_num_add << "\n";
    cout << "total number of mul operation: " << sum_num_mul << "\n";
    return 0;
}