#include <iostream>
#include <cassert>
#include "data.h"
#include "random.h"

class RuleNeuron{
public:
    std::vector<double> W;
    std::vector<double> dW;
    std::vector<double> out;
    double strength{};

    explicit RuleNeuron()
        :   W(4),
            dW(4),
            out(2)
    {
        kki::rng::global.populateReal(W, -1., 1.);
    }

    static double alpha(double x, double a, double b){
        double e = std::exp(b * (x - a));
        return 1. / (1 + e);
    }

    void forward(const std::vector<double>& input){
        assert(input.size() == 2);
        for (unsigned i = 0; i < input.size(); ++i){
            out[i] = alpha(input[i], W[i*2], W[i*2+1]);
        }
        strength =
            std::accumulate(
                out.begin(),
                out.end(),
                1.,
                [](double a, double b){return a*b;});
    }

    void zeroGrad(){
        std::fill(dW.begin(), dW.end(), 0.);
    }

    void backward(){
        for(unsigned i = 0; i < W.size(); ++i){
            W[0] += dW[0];
        }
    }
};

class OutputNeuron{
public:
    std::vector<double> W;
    std::vector<double> dW;
    double out{};

    explicit OutputNeuron()
        :   W(3),
            dW(3)
    {
        kki::rng::global.populateReal(W, -1., 1.);
    }

    void forward(const std::vector<double>& input){
        assert(input.size() == 2);
        out = 0;
        for (unsigned i = 0; i < 2; ++i){
            out += W[i] * input[i];
        }
        out += W.back();
    }

    void backward(){
        for(unsigned i = 0; i < W.size(); ++i){
            W[0] += dW[0];
        }
    }

    void zeroGrad(){
        std::fill(dW.begin(), dW.end(), 0.);
    }
};

class FuzzyNeuralNet{
public:
    std::vector<RuleNeuron>     rule_neurons{};
    std::vector<OutputNeuron>   output_neurons{};
    unsigned rules;
    double out{};
    double lr;


    FuzzyNeuralNet(unsigned rules, double lr)
        :   rules(rules),
            lr(lr)
    {
        for (unsigned i = 0; i < rules; ++i){
            rule_neurons.emplace_back();
            output_neurons.emplace_back();
        }
    }

    void forward(const std::vector<double>& input){
        assert(input.size() == 2);
        double alpha = 0;
        double sum = 0;
        for (unsigned i = 0; i < rules; ++i){
            rule_neurons[i].forward(input);
            alpha += rule_neurons[i].strength;
            output_neurons[i].forward(input);
            sum += output_neurons[i].out * rule_neurons[i].strength;
        }
        out = sum/alpha;
    }

    void grad(const std::vector<double>& input, double Y){
        double error = Y - out;
        double pi = std::accumulate(
            rule_neurons.begin(),
            rule_neurons.end(),
            0.,
            [](double acc, const RuleNeuron& r){ return acc + r.strength;});
        double pi2 = pi*pi;

        for (unsigned i = 0; i < rules; ++i){
            RuleNeuron& rule = rule_neurons[i];
            OutputNeuron& output = output_neurons[i];

            double d_rule = 0;
            for (unsigned j = 0; j < rules; ++j){
                d_rule += rule_neurons[j].strength * (output.out - output_neurons[j].out);
            }
            d_rule *= (lr * error) / pi2;

            double dw1 = d_rule * rule.out[1] * rule.out[0] * (1. - rule.out[0]);
            double dw2 = d_rule * rule.out[0] * rule.out[1] * (1. - rule.out[1]);
            rule.dW[0] += dw1 * rule.W[1];
            rule.dW[1] -= dw1 * (input[0] - rule.W[0]);
            rule.dW[2] += dw2 * rule.W[3];
            rule.dW[3] -= dw2 * (input[1] - rule.W[2]);

            double d_out = (lr * error) * rule.strength / pi;
            output.dW[0] += d_out * input[0];
            output.dW[1] += d_out * input[1];
            output.dW[2] += d_out;
        }
    }

    void backward(){
        for (unsigned i = 0; i < rules; ++i){
            rule_neurons[i].backward();
            rule_neurons[i].zeroGrad();
            output_neurons[i].backward();
            output_neurons[i].zeroGrad();
        }
    }

    double eval(const std::vector<std::pair<std::vector<double>, double>>& data){
        double res = 0.;
        for (auto& d : data){
            forward(d.first);
            double diff = d.second - out;
            res += diff * diff/2;
        }
        return res/data.size();
    }
};


void train(FuzzyNeuralNet& fnn, const std::vector<std::pair<std::vector<double>, double>>& data, unsigned epochs){
    for (unsigned epoch = 0; epoch < epochs; ++epoch){
        for (auto& d : data){
            fnn.forward(d.first);
            fnn.grad(d.first, d.second);
        }
        fnn.backward();
        std::cout << "Epoch: " << epoch << " " << fnn.eval(data) << std::endl;
    }
}

void trainStochastic(FuzzyNeuralNet& fnn, const std::vector<std::pair<std::vector<double>, double>>& data, unsigned epochs){
    for (unsigned epoch = 0; epoch < epochs; ++epoch){
        for (auto& d : data){
            fnn.forward(d.first);
            fnn.grad(d.first, d.second);
            fnn.backward();
        }
        std::cout << "Epoch: " << epoch << " " << fnn.eval(data) << std::endl;
    }
}

int main(int argc, const char* argv[])
{
    if (argc < 3){
        throw std::invalid_argument("Invalid number of arugments!");
    }
    if (argv[1] == std::string("train")){
        unsigned rules = 5;
        if (argc == 4){
            rules = std::stoul(argv[3]);
        }
        auto data = loadData(argv[2]);

        FuzzyNeuralNet fnn(rules, 0.0001);
        train(fnn, data, 10000);
        std::cout << "End!" << std::endl;
    }
    else if (argv[1] == std::string("data")) {
        createData(argv[2]);
    }
    else{
        throw std::invalid_argument("First argument not valid (Valid options 'data' for data creation and 'train' for network training)");
    }
    return 0;
}
