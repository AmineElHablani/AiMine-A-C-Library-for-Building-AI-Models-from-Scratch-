#ifndef AIMINE_H
#define AIMINE_H

#include <stdlib.h>  // For malloc and free
#include <math.h>    // For exp, log, fmax, fmin
#include <string.h>  // For strcmp
#include <stdio.h>   // For printf

// Define the struct for a neuron
struct neurone {
    float *weights;
    float bias;
};

// Define the struct for a layer
struct layer {
    int size;
    struct neurone **data;
    float *result_forward;
    char *activation;
};

// Define the struct for delta
struct delta {
    int size;
    float *value;
};

// Function declarations
int find(int size, int tab[size], int element);
void freeLayer(struct layer* myLayer);
void free_table_layer(struct layer* my_layer);
void free_neurone(struct neurone* table);
void free_delta(struct delta** table);
float substract_vector(int vector_size, float vector1[vector_size], float vector2[vector_size], float result[vector_size]);
float prod_vectors(int size_vect, float vect1[size_vect], float vect2[size_vect]);
float Sum(int size, float array[]);
float* zeros(int size);
float init_weights();
void free_dense_layer(struct neurone* table, int size);
void ReLu(int size, float input_table[size]);
float sigmoid(float x);
float sigmoid_derivative(float x);
void softmax(int size, float input_table[size]);
void Droupout(int size, float input_table[size], float percentage);
void free_dropout(int size, int disabled_list[size]);
float sigmoid_backward(float delta, float output);
float relu_derivative(float x);
float relu_backward(float delta, float output);
void activation(struct layer* current_layer);
float cross_entropy_loss(float predicted, float target);
void freeDelta(struct delta* myDelta);
void freeDeltaTable(struct delta* delta_table, int number_layers);
struct delta initializeDelta(int size);
void from_dense_froward(struct layer* current_layer, struct layer prev_layer);
void from_input_froward(struct layer* current_layer, int input_size, float x_i[input_size]);
void Forward(int input_size, int num_layers, float X_i[input_size], struct layer tables[], float *output);
void delta(int number_of_layers, struct layer tables[], struct delta delta_table[], float target);
void weight_update(struct layer tables[], struct delta delta_table[], int num_layers, int input_size, float learning_rate);
struct layer initializeLayer(int size, int prev_layer_size, char* activation);
float accuracy(int size, float y_pred[], float target[]);


float nsquare(float number);
float find_min(int vector_size, float vect[vector_size]);
float find_max(int vector_size, float vect[vector_size]);
float tri(int vector_size, float vect[vector_size]);
float median(int vector_size, float tri_vect[vector_size]);
float iqr(int vector_size, float tri_vect[vector_size]);
float mean(int vector_size, float vect[vector_size]);
float standard_deviation(int vector_size, float vect[vector_size], float x_bar);
void extract_column(int row_size, int columns_size, float tab[row_size][columns_size], float result[row_size], int idx_column);
void extract_row(int row_size, int columns_size, float train_data[row_size][columns_size], float row[columns_size], int idx_row);
void standard_scaler(int row_size, int columns_size, float table[row_size][columns_size], float mean_table[columns_size], float std_table[columns_size]);
void transform_standard_scaler(int row_size, int column_size, float table[row_size][column_size], float mean_table[column_size], float std_table[column_size]);
void min_max_scaling(int row_size, int column_size, float table[row_size][column_size], float min_table[column_size], float max_table[column_size]);
void transform_min_max_scaler(int row_size, int column_size, float table[row_size][column_size], float min_table[column_size], float max_table[column_size]);
void robust_scaler(int row_size, int column_size, float table[row_size][column_size], float median_table[column_size], float iqr_table[column_size]);
void transform_robust_scaler(int row_size, int column_size, float table[row_size][column_size], float median_table[column_size], float iqr_table[column_size]);

void read_csv(const char *filename, int row_size, int column_size, float table[row_size][column_size], char columns[][50]);
void display(int row_size, int column_size, float tab[row_size][column_size], char label[column_size][50]);
void shuffle(int row_size, int column_size, float table[row_size][column_size]);
void get_x_train(int start, int x_size, int column_size, int y_index, float table[][column_size], float x[][column_size-1]);
void get_x_test(int start, int x_size, int column_size, int y_index, float table[][column_size], float x[][column_size-1]);
void split_data(int row_size, int column_size, float table[row_size][column_size], float x_train[][column_size-1], float y_train[], float x_test[][column_size-1], float y_test[], float test_percentage, int y_index);


#endif // AIMINE_H
