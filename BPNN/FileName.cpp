#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <opencv.hpp>
using namespace std;
using namespace cv;

#define generation 5//迭代次数
#define Hid 50//隐含层节点数
const double learning_rate = 0.6;//学习率
//学习率是指每次迭代时，参数更新的幅度，如果学习率过大，会导致参数更新过大，无法收敛；如果学习率过小，会导致参数更新过小，收敛速度过慢
int ERRORcount = 0;//误差计数
vector<vector<double>> ImageTest(10000, vector<double>(28 * 28));
vector<vector<int>> LabelTest(10000, vector<int>(10));

vector<vector<double>> ImageTrain(60000, vector<double>(28 * 28));
vector<vector<int>> LabelTrain(60000, vector<int>(10));

string train_image_path = "E:/DatabaseWork/MNIST/train-images.idx3-ubyte";
string train_label_path = "E:/DatabaseWork/MNIST/train-labels.idx1-ubyte";
string test_image_path = "E:/DatabaseWork/MNIST/t10k-images.idx3-ubyte";
string test_label_path = "E:/DatabaseWork/MNIST/t10k-labels.idx1-ubyte";

uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}

void readFile(const string& mnist_img_path, const string& mnist_label_path, vector<vector<double>>& _image, vector<vector<int>>& _label)
{
    // 以二进制格式读取mnist数据库中的图像文件和标签文件
    ifstream mnist_image(mnist_img_path, ios::in | ios::binary);
    ifstream mnist_label(mnist_label_path, ios::in | ios::binary);
    if (mnist_image.is_open() == false)
    {
        cout << "open mnist image file error!" << endl;
        return;
    }
    if (mnist_label.is_open() == false)
    {
        cout << "open mnist label file error!" << endl;
        return;
    }

    uint32_t magic; // 文件中的魔术数(magic number)
    uint32_t num_items;// mnist图像集文件中的图像数目
    uint32_t num_label;// mnist标签集文件中的标签数目
    uint32_t rows;// 图像的行数
    uint32_t cols;// 图像的列数

    mnist_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051)
    {
        cout << "this is not the mnist image file" << endl;
        return;
    }
    mnist_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049)
    {
        cout << "this is not the mnist label file" << endl;
        return;
    }
    // 读图像/标签数
    mnist_image.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);

    mnist_label.read(reinterpret_cast<char*>(&num_label), 4);
    num_label = swap_endian(num_label);
    // 判断两种标签数是否相等
    if (num_items != num_label)
    {
        cout << "the image file and label file are not a pair" << endl;
    }
    // 读图像行数、列数
    mnist_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    mnist_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);

    // 读取图像
    char* pixels = new char[rows * cols];
    Mat image(rows, cols, CV_8UC1, (uchar*)pixels);
    char label;
    char save_pth[256];
    int size = rows * cols;

    for (int i = 0; i != num_items; i++)
    {
        mnist_image.read(pixels, size);
        mnist_label.read(&label, 1);
        //图像读取
        for (int x = 0; x < 28; x++)
            for (int y = 0; y < 28; y++) {
                _image[i][x * 28 + y] = image.at<uchar>(x, y) / 255.0;//归一化
                //if (_image[i][x * 28 + y] != 0)  _image[i][x * 28 + y] = 1;
            }
        //标签读取
        _label[i][(int)label] = 1;
        /*waitKey(0);*/
    }
    delete[] pixels;
}

//定义激活函数为sigmoid函数
double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
}

//定义激活函数的导数
double sigmoid_derivative(double x) {
    //return sigmoid(x) * (1 - sigmoid(x));
    return x * (1 - x);
}

//定义BP神经网络类
class BPNeuralNetwork {
private:
    //定义输入层、隐含层、输出层的节点数
    int input_size;
    int hidden_size;
    int output_size;

    //定义输入层、隐含层、输出层的值向量
    vector<double> input_layer;
    vector<double> hidden_layer;
    vector<double> output_layer;

    //定义输入层到隐含层、隐含层到输出层的权重矩阵
    vector<vector<double>> input_hidden_weights;
    vector<vector<double>> hidden_output_weights;

    //定义隐含层、输出层的偏置向量
    vector<double> hidden_bias;
    vector<double> output_bias;

public:
    //构造函数，初始化网络结构和参数
    BPNeuralNetwork(int input_size, int hidden_size, int output_size) {
        this->input_size = input_size;
        this->hidden_size = hidden_size;
        this->output_size = output_size;

        //初始化输入层、隐含层、输出层的值向量为0
        input_layer.resize(input_size, 0.0);
        hidden_layer.resize(hidden_size, 0.0);
        output_layer.resize(output_size, 0.0);

        //初始化输入层到隐含层、隐含层到输出层的权重矩阵为[-1,1]之间的随机数
        srand(time(NULL));
        input_hidden_weights.resize(input_size, vector<double>(hidden_size));
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                input_hidden_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }
        hidden_output_weights.resize(hidden_size, vector<double>(output_size));
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                hidden_output_weights[i][j] = ((double)rand() / RAND_MAX) * 2 - 1;
            }
        }

        //初始化隐含层、输出层的偏置向量为[-1,1]之间的随机数
        hidden_bias.resize(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            hidden_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
        output_bias.resize(output_size);
        for (int i = 0; i < output_size; i++) {
            output_bias[i] = ((double)rand() / RAND_MAX) * 2 - 1;
        }
    }

    //前向传播函数，根据输入计算输出
    vector<double> feedforward(vector<double> input) {
        //将输入赋值给输入层
        for (int i = 0; i < input_size; i++) {
            input_layer[i] = input[i];
        }

        //计算隐含层的值
        for (int i = 0; i < hidden_size; i++) {
            //计算第i个隐含层节点的输入，等于输入层的值乘以对应的权重，再加上偏置
            double hidden_input = 0.0;
            for (int j = 0; j < input_size; j++) {
                hidden_input += input_layer[j] * input_hidden_weights[j][i];
            }
            hidden_input += hidden_bias[i];

            //计算第i个隐含层节点的输出，等于激活函数作用于输入
            hidden_layer[i] = sigmoid(hidden_input);
        }

        //计算输出层的值
        for (int i = 0; i < output_size; i++) {
            //计算第i个输出层节点的输入，等于隐含层的值乘以对应的权重，再加上偏置
            double output_input = 0.0;
            for (int j = 0; j < hidden_size; j++) {
                output_input += hidden_layer[j] * hidden_output_weights[j][i];
            }
            output_input += output_bias[i];

            //计算第i个输出层节点的输出，等于激活函数作用于输入
            output_layer[i] = sigmoid(output_input);

        }

        //返回输出层的值
        return output_layer;
    }

    //反向传播函数，根据误差调整参数
    void backpropagate(vector<int> target) {
        //计算输出层的误差项，等于期望输出与实际输出的差乘以激活函数的导数
        vector<double> output_error(output_size);
        for (int i = 0; i < output_size; i++) {
            output_error[i] = (target[i] - output_layer[i]) * sigmoid_derivative(output_layer[i]);
        }

        //计算隐含层的误差项，等于输出层误差项乘以对应的权重，再乘以激活函数的导数
        vector<double> hidden_error(hidden_size);
        for (int i = 0; i < hidden_size; i++) {
            double sum = 0.0;
            for (int j = 0; j < output_size; j++) {
                sum += output_error[j] * hidden_output_weights[i][j];
            }
            hidden_error[i] = sum * sigmoid_derivative(hidden_layer[i]);
        }

        //更新隐含层到输出层的权重，等于学习率乘以输出层误差项乘以隐含层的值
        for (int i = 0; i < hidden_size; i++) {
            for (int j = 0; j < output_size; j++) {
                hidden_output_weights[i][j] += learning_rate * output_error[j] * hidden_layer[i];
            }
        }

        //更新输入层到隐含层的权重，等于学习率乘以隐含层误差项乘以输入层的值
        for (int i = 0; i < input_size; i++) {
            for (int j = 0; j < hidden_size; j++) {
                input_hidden_weights[i][j] += learning_rate * hidden_error[j] * input_layer[i];
            }
        }

        //更新输出层的偏置，等于学习率乘以输出层误差项
        for (int i = 0; i < output_size; i++) {
            output_bias[i] += learning_rate * output_error[i];
        }

        //更新隐含层的偏置，等于学习率乘以隐含层误差项
        for (int i = 0; i < hidden_size; i++) {
            hidden_bias[i] += learning_rate * hidden_error[i];
        }
    }

    //训练函数，根据训练数据和迭代次数来调整网络参数
    void train(vector<vector<double>> input_data, vector<vector<int>> output_data, int epoch) {
        //循环迭代指定的次数
        for (int i = 0; i < epoch; i++) {
            //对每一组训练数据进行前向传播和反向传播
            for (int j = 0; j < input_data.size(); j++) {
                feedforward(input_data[j]);
                backpropagate(output_data[j]);
            }
            //输出当前的误差
            cout << "Epoch " << i + 1 << ": " << get_error(input_data, output_data) << endl;
        }
    }

    //测试函数，根据测试数据来计算网络的输出
    void test(vector<vector<double>> input_data, vector<vector<int>> output_data) {
        //对每一组测试数据进行前向传播
        for (int i = 0; i < input_data.size(); i++) {
            vector<double> result = feedforward(input_data[i]);
            //输出输入和输出
            //cout << "Input: ";
            //for (int j = 0; j < input_size; j++) {
            //    cout << input_data[i][j] << " ";
            //}
            //cout << endl;
            int tempa=0, tempb = 0;
            //输出TestLabel集
            for (int p = 0; p < 10; p++) {
                if (output_data[i][p] == 1) {
                    cout << i << " " << "test:" << p << " ";
                    tempa = p;
                    break;
                }
            }
            double temp = 0; int count = 0;
            for (int j = 0; j < output_size; j++) {
                if (result[j] > temp) {
                    temp = result[j];
                    count = j;
                }
            }
            tempb = count;
            if(tempa != tempb) ERRORcount++;
            cout << " " << count << " ";
            cout << endl;
        }
        //错误率
        cout << "ERRORcount:" << ERRORcount << endl;
        //输出当前的误差
        cout << "Epoch " << ": " << get_error(input_data, output_data) << endl;
    }

    //计算误差函数，根据训练数据和期望输出来计算网络的均方误差
    double get_error(vector<vector<double>> input_data, vector<vector<int>> output_data) {
        //初始化误差为0
        double error = 0.0;
        //对每一组训练数据进行前向传播和误差累加
        for (int i = 0; i < input_data.size(); i++) {
            vector<double> result = feedforward(input_data[i]);
            for (int j = 0; j < output_size; j++) {
                error += pow(output_data[i][j] - result[j], 2);
            }
        }
        //返回误差的平均值
        return error / input_data.size();
    }
};

//主函数，创建BP神经网络对象，并用训练数据和测试数据来训练和测试网络
int main() {
    readFile(test_image_path, test_label_path, ImageTest, LabelTest);
    readFile(train_image_path, train_label_path, ImageTrain, LabelTrain);

    //创建BP神经网络对象，设置输入层、隐含层、输出层的节点数
    BPNeuralNetwork bp(28 * 28, Hid, 10);
    //训练网络，设置学习率和迭代次数
    bp.train(ImageTrain, LabelTrain, generation);
    //测试网络，输出预测结果
    bp.test(ImageTest, LabelTest);

    return 0;
}
