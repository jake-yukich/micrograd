/*
 * (TODO: add description here)
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <math.h>

// -----------------------------------------------------------------------------
// MICROGRAD ENGINE. ...

typedef struct Value Value;
typedef void (*BackwardFunction)(Value*);

struct Value {
    double data;
    double grad;
    char* label;
    int n_prev;
    struct Value** _prev;
    BackwardFunction _backward;
    char* _op;
};

Value* create_value(double data,
                    char* label,
                    int n_prev,
                    struct Value** _prev,
                    BackwardFunction _backward,
                    char* _op) {
    // allocate memory for the value
    Value* v = malloc(sizeof(Value));
    if (v == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    v->data = data;
    v->grad = 0.0;
    v->label = label;
    v->n_prev = n_prev;

    // we require memory for _prev to have been allocated by the caller
    if (n_prev > 0 && _prev == NULL) {
        fprintf(stderr, "Error: _prev is NULL but n_prev is %d\n", n_prev);
        free(v);
        exit(1);
    }
    v->_prev = _prev;

    v->_backward = _backward;
    v->_op = _op;
    return v;
}

// Macros for creating values:

#define VALUE(data, label) create_value(data, label, 0, NULL, NULL, NULL)
#define VALUE_WITH_PREV(data, label, n_prev, _prev, _backward, _op) create_value(data, label, n_prev, _prev, _backward, _op)

void free_value(Value* v) {
    free(v);
}

// define backward functions for each operation:

void backward_add(Value* v) {
    v->_prev[0]->grad += v->grad;
    v->_prev[1]->grad += v->grad;
}

void backward_mul(Value* v) {
    v->_prev[0]->grad += v->_prev[1]->data * v->grad;
    v->_prev[1]->grad += v->_prev[0]->data * v->grad;
}

void backward_pow(Value* v) {
    v->_prev[0]->grad += (v->_prev[0]->data * v->data) * v->grad;
}

void backward_relu(Value* v) {
    v->_prev[0]->grad += (v->data > 0) * v->grad;
}

void backward_tanh(Value* v) {
    v->_prev[0]->grad += (1 - v->data * v->data) * v->grad;
}

void backward_exp(Value* v) {
    v->_prev[0]->grad += v->data * v->grad;
}

void backward_log(Value* v) {
    v->_prev[0]->grad += (1 / v->_prev[0]->data) * v->grad;
}

// define operation functions:

Value* value_add(Value* a, Value* b) {
    Value* out = VALUE_WITH_PREV(a->data + b->data, "add", 2, a, b, "add");
    out->_backward = backward_add;
    return out;
}

Value* value_mul(Value* a, Value* b) {
    Value* out = VALUE_WITH_PREV(a->data * b->data, "mul", 2, a, b, "mul");
    out->_backward = backward_mul;
    return out;
}

Value* value_tanh(Value* a) {
    Value* out = VALUE_WITH_PREV(tanh(a->data), "tanh", 1, a, backward_tanh, "tanh");
    return out;
}

// ...

// void backward(Value* v) {
//     // topological order all of the children in the graph
//     Value* topo[1000];
//     int visited[1000] = {0};
//     int index = 0;

//     void build_topo(Value* v) {
//         if (!visited[v->id]) {
//             visited[v->id] = 1;
//             for (int i = 0; i < v->n_prev; i++) {
//                 build_topo(v->_prev[i]);
//             };
//             topo[index++] = v;
//         };
//     };

//     build_topo(v);
// }

// Value* value_add(Value* a, Value* b);
// Value* value_mul(Value* a, Value* b);
// Value* value_tanh(Value* v);
// // ... other operations

int main() {
    Value v = {2.0};

    printf("Value: {data: %f}\n", v.data);
    return 0;
}

// -----------------------------------------------------------------------------
// NEURAL NETWORK.

// typedef struct Neuron {
//     Value* weights;
//     Value bias;
//     int n_inputs;
//     bool is_linear;
// } Neuron;

// void neuron_forward(Neuron* n, Value* inputs[], Value* output);

// Value* create_value(double data);
// void free_value(Value* v);

// Neuron* create_neuron(int num_inputs, bool nonlin);
// void free_neuron(Neuron* n);

// typedef struct Layer {
//     Neuron* neurons;
//     int n_neurons;
// } Layer;

// typedef struct MLP {
//     Layer* layers;
//     int n_layers;
// } MLP;

// -----------------------------------------------------------------------------
// GRAPHING EXAMPLE.

// #define START -5.0
// #define END 5.0
// #define STEP 0.25

// double f(double x) {
//     return 3 * pow(x, 2) - 4 * x + 5;
// }

// int main() {
//     int num_points = (int)((END - START) / STEP) + 1;
//     double *xs = malloc(num_points * sizeof(double));
//     double *ys = malloc(num_points * sizeof(double));

//     if (xs == NULL || ys == NULL) {
//         fprintf(stderr, "Memory allocation failed\n");
//         return 1;
//     }

//     // Generate x values and calculate corresponding y values
//     for (int i = 0; i < num_points; i++) {
//         xs[i] = START + i * STEP;
//         ys[i] = f(xs[i]);
//     }

//     // Output data to a file
//     FILE *fp = fopen("plot_data.txt", "w");
//     if (fp == NULL) {
//         fprintf(stderr, "Failed to open file\n");
//         free(xs);
//         free(ys);
//         return 1;
//     }

//     for (int i = 0; i < num_points; i++) {
//         fprintf(fp, "%f %f\n", xs[i], ys[i]);
//     }

//     fclose(fp);
//     free(xs);
//     free(ys);

//     printf("Data written to plot_data.txt\n");
//     return 0;
// }