#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "AiMine.h"




/*/Define Neurone structure
struct neurone {
    float bias;
    float* weights;
};


//define layer structure to use it in the forward and backwar procedure
struct layer{
    int size;    //size of the layer (number of neurones) => size of data table ==> size of table of neurones
    char* activation;
    struct neurone** data;
    float* result_forward;
};

struct delta{
    int size;
    float* value;
};*/
 

//find element in table  (used to find index)
int find(int size, int tab[size], int element){
    int res = 0;
    int i=0;
    while (i < size)
    {
        if (tab[i]==element)
        {
            res = 1;
            return res;
        }
    }
    return res;
}

// Function to free memory allocated for a layer
void freeLayer(struct layer* myLayer) {
    for (int i = 0; i < myLayer->size; ++i) {
        free(myLayer->data[i]->weights);
    }
    free(myLayer->data);
    free(myLayer->result_forward);
}

void free_table_layer(struct layer* my_layer) {
    // Free memory for the activation string
    free(my_layer->activation);

    // Free memory for the data array (neurons)
    free(my_layer->data);

    // Free memory for the result_forward array
    free(my_layer->result_forward);

    // Finally, free the memory for the layer itself
    free(my_layer);
}


// Function to free memory for layer
void free_neurone(struct neurone* table) {
    if (table != NULL) {
        free(table->weights);
        // Free other members as needed
        free(table);
    }
}

// Function to free memory for delta
void free_delta(struct delta** table) {
    if (table != NULL) {
        free((*table)->value);
        // Free other members as needed
        free(table);
    }
}




// substract a vector from a vector 
float substract_vector(int vector_size, float vector1[vector_size], float vector2[vector_size], float result[vector_size]){
    for(int i=0; i< vector_size;i++){
        result[i]= vector1[i] - vector2[i];
    }
}

//multiplicate vector * vector 
float prod_vectors(int size_vect,float vect1[size_vect],float vect2[size_vect]){
    float s=0;
    for(int i=0; i< size_vect;i++){
        s+= vect1[i]*vect2[i] ; 
    }
    return s;
}

//sum of vectors items 
float Sum(int size, float array[]) {
    float sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += array[i];
    }
    return sum;
}

//fucntion initialize array of zeros

float* zeros(int size){
    float* array=(float*)malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++)
    {
        array[i]=0;
    }
    return array;
}


//initialize random weight : function

float init_weights(){
    return ((float)rand() / (float)RAND_MAX);
}

void free_dense_layer(struct neurone* table, int size) {
    if (table) {
        for (int i = 0; i < size; i++) {
            free(table->weights);
        }
        
        free(table);
    }
}

//Relu : Activation function

void ReLu(int size , float input_table[size]){
    for (size_t i = 0; i < size ; i++)
    {
        if (input_table[i] < 0)
        {
            input_table[i] = 0 ;
        }
    }
}

//sigmoid function

float sigmoid(float x){

    return ( 1 / ( 1 + exp(- x)));
}

//sigmoid derivative
float sigmoid_derivative(float x){
    return sigmoid(x) * (1 - sigmoid(x));
}


//softmax
void softmax(int size, float input_table[size]){
    //caculate sum exp(xi)
    float sum = 0;
    for (size_t i = 0; i < size; i++)
    {
        sum += exp(input_table[i]);
    }

    for (size_t i = 0; i < size; i++)
    {
        input_table[i] = exp(input_table[i]) / sum ; 
    }
}


//Droupout (the dropout function will generate an array of indexes of disabled neurone to use it while training)
void Droupout(int size , float input_table[size] , float percentage){
    
    //initialize a list contain the index of disabled neurones (randomly)
    int size_disabled= round(percentage * size) ;
    int* disabled_list = (int*)malloc( size_disabled * sizeof(int*));

    // Seed the random number generator with the current time
    srand(time(NULL)); 

    //fill the list 
    size_t i = 0 ;
    while(i < size_disabled){
        //generate random integer lower than the layer size
        int random= rand()%size ;

        //verify that we didnt already choose that index (verify that he dosent exist in the list of the index of  the disabled neurones)
        if (find(i, disabled_list, random)==0){
            disabled_list[i]=random ;
            i+=1;
        }
    }
}

void free_dropout(int size, int disabled_list[size]){

    free(disabled_list);
}


//BatchNormalization


//sigmoid_backward : sigmoid derivative * delta

float sigmoid_backward(float delta, float output){

    return delta * sigmoid_derivative(output);
}

//relu derivative 

float relu_derivative(float x){
    if (x >0 )
    {
        return 1;
    }
    else {
        return 0;
    }
}

//relu_backward : relu_derivative * delta
float relu_backward(float delta, float output){
    return delta * relu_derivative(output);
}



//apply activation 
void activation(struct layer* current_layer){
    if (strcmp(current_layer->activation, "ReLu") == 0)
    {
        ReLu(current_layer->size, current_layer->result_forward);
    }
    else if (strcmp(current_layer->activation, "sigmoid") == 0) {
        //apply sigmoid function to all the neurones 
        for (size_t i = 0; i < current_layer->size; i++)
        {
            current_layer->result_forward[i]=sigmoid(current_layer->result_forward[i]);
        }
    }    
}

//Forward
//cross_entropy_loss 

float cross_entropy_loss(float predicted, float target) {
    // Avoid log(0) by adding a small epsilon value
    float epsilon = 1e-15;
    
    // Clip predicted values to avoid log(0) or log(1)
    predicted = fmax(epsilon, fmin(1 - epsilon, predicted));

    // Calculate cross-entropy loss
    float loss = -target * log(predicted) - (1 - target) * log(1 - predicted);
    return loss;
}




// Function to free memory allocated for struct delta
void freeDelta(struct delta* myDelta) {
    free(myDelta->value);
}

// Function to free memory allocated for delta table
void freeDeltaTable(struct delta* delta_table, int number_layers) {
    for (int layer = 0; layer < number_layers; ++layer) {
        freeDelta(&delta_table[layer]);
    }
    free(delta_table);
}

//function train neural networks 

// Function to initialize struct delta with variable-size value array
struct delta initializeDelta(int size) {
    struct delta myDelta;

    // Set the size
    myDelta.size = size;
    printf("size delta = %d\n",size);

    // Allocate memory for the value array
    myDelta.value = (float*)malloc(size * sizeof(float));

    // Check if memory allocation was successful
    if (myDelta.value == NULL) {
        perror("Memory allocation failed1\n");
        exit(0);  // Exit the program with an error code
    }

    // Initialize the value array with the provided values
    for (int i = 0; i < size; ++i) {
        myDelta.value[i] = 0;
    }

    return myDelta;
}
//forward procedure that take dense layer as previous
void from_dense_froward(struct layer* current_layer, struct layer prev_layer){

        // sum = sum prod(X ,weights_j) + bj
        float sum = 0 ;

        for (size_t curr_neurone = 0; curr_neurone < current_layer->size; curr_neurone++)
        {
            //iterate in the previous layer to calculate  y= WX 
            for (size_t idx_weight = 0; idx_weight < prev_layer.size; idx_weight++)
            {
                sum += current_layer->data[curr_neurone]-> weights[idx_weight] * prev_layer.result_forward[idx_weight];
            }
            //result = W*X_i +b_i
            current_layer->result_forward[curr_neurone]= sum + current_layer->data[curr_neurone]->bias ;
            
        }    
}

//forward procedure that take input layer as previous 
void from_input_froward(struct layer* current_layer, int input_size, float x_i[input_size] ){

        // sum = sum prod(X ,weights_j) + bj
        float sum = 0 ;

        for (size_t curr_neurone = 0; curr_neurone < current_layer->size; curr_neurone++)
        {
            //iterate in the previous layer to calculate  y= WX 
            for (size_t idx_weight = 0; idx_weight < input_size; idx_weight++)
            {
                sum += current_layer->data[curr_neurone]-> weights[idx_weight] * x_i[idx_weight];
            }
            //result = W*X_i +b_i
            current_layer->result_forward[curr_neurone]= sum + current_layer->data[curr_neurone]->bias ;
            
        }    
}

void Forward(int input_size, int num_layers, float X_i[input_size], struct layer tables[],float *output){

    //iterate in layers
    size_t i = 0 ;
    while (i < num_layers)
    {
        if (i == 0)
        {
            struct layer current_layer = tables[i];
            //one layer forward
            //printf("before forward pass %d : %f\n",i,current_layer.result_forward[0]);
            from_input_froward(&current_layer, input_size, X_i);
            //printf("after forward pass %d : %f\n",i,current_layer.result_forward[0]);
            //apply activation function
            activation(&current_layer);
            //printf("after activation %d : %f\n",i,current_layer.result_forward[0]);


            i++;
            
        }
        else
        {            
            struct layer current_layer = tables[i];
            struct layer prev_layer = tables[i-1];
            //one layer forward
            //printf("before forward pass %d : %f\n",i,current_layer.result_forward[0]);
            from_dense_froward(&current_layer, prev_layer);
            //printf("after forward pass %d : %f\n",i,current_layer.result_forward[0]);

            //apply acctivation
            activation(&current_layer);
            //printf("after activation %d : %f\n",i,current_layer.result_forward[0]);

            i++;
        }
        
    }
    //output in case of 1 neyrone (code must be optimised in the future) 
    *output = tables[num_layers - 1].result_forward[0]; //ouput is an element in y_pred
    
}

//calculate delta

///////////////////////////::
void delta(int number_of_layers, struct layer tables[], struct delta delta_table[], float target) {
    // Calculate delta for output layer (y_pred - y_test)
    float loss_gradient = tables[number_of_layers - 1].result_forward[0] - target;
    // Delta = loss_gradient * derivative_sigmoid(output)
    delta_table[number_of_layers - 1].value[0] = sigmoid_derivative(tables[number_of_layers - 1].result_forward[0]) * loss_gradient;


    // Calculate delta for the hidden layers with ReLU activation
    //printf("number_of_layers - 2 : %d\n",number_of_layers - 2);
    for (int layer_idx = number_of_layers - 2; layer_idx >-1; layer_idx--) {
        //printf("ok %d\n", layer_idx);
        for (int neurone_idx = 0; neurone_idx < tables[layer_idx].size; neurone_idx++) {
            float prod_weight_delta = 0;

            // Calculate the sum (weight coming from this neuron * delta of the resulting neuron)
            for (size_t i = 0; i < tables[layer_idx + 1].size; i++) {
                prod_weight_delta += tables[layer_idx + 1].data[i]->weights[neurone_idx] * delta_table[layer_idx + 1].value[i];
                //printf("delta_table[layer_idx + 1].value[i] %d : %f\n",neurone_idx,delta_table[layer_idx + 1].value[i]);
            }

            // Delta = derivative_relu * sum(weight * delta)
            
            delta_table[layer_idx].value[neurone_idx] = relu_derivative(tables[layer_idx].result_forward[neurone_idx]) * prod_weight_delta;

        }
    }
}


void weight_update(struct layer tables[], struct delta delta_table[], int num_layers, int input_size, float learning_rate) {
    // Update the first hidden layer
    for (size_t neuron_idx = 0; neuron_idx < tables[0].size; neuron_idx++) {
        for (size_t w = 0; w < input_size; w++) {

            // Update each weight coming to this neuron 
            //printf("\nbefore: Weight[%zu] = %f\n", w, tables[0].data[neuron_idx]->weights[w]);
            
            tables[0].data[neuron_idx]->weights[w] -= learning_rate * tables[0].result_forward[neuron_idx] * delta_table[0].value[neuron_idx];
            //printf("After: Weight[%zu] = %f\n", w, tables[0].data[neuron_idx]->weights[w]);

        }
        // Update bias: bias -= lr * sum of deltas in that layer
        //printf("deleting = %d\n", delta_table[0].size);

        tables[0].data[neuron_idx]->bias -= learning_rate * Sum(delta_table[0].size, delta_table[0].value);
        //printf("Updated Bias = %f\n", tables[0].data[neuron_idx]->bias);
        //printf("########################################");

    }

    // Update all the rest layers (if exists)
    for (size_t layer = 1; layer < num_layers; layer++) {
        for (size_t neuron_idx = 0; neuron_idx < tables[layer].size; neuron_idx++) {
            // Update weights and bias for each neuron
            for (size_t w = 0; w < input_size; w++) {
                // Update each weight coming to this neuron 
                tables[layer].data[neuron_idx]->weights[w] -= learning_rate * tables[layer].result_forward[neuron_idx] * delta_table[layer].value[neuron_idx];
            }
            // Update bias: bias -= lr * sum of deltas in that layer
            tables[layer].data[neuron_idx]->bias -= learning_rate * Sum(delta_table[layer].size, delta_table[layer].value);
        }
    }
}




// Function to initialize a layer
struct layer initializeLayer(int size,int prev_layer_size, char* activation) {
    struct layer myLayer;

    myLayer.size = size;
    myLayer.activation = activation;

    // Allocate memory for the array of neurones
    myLayer.data = (struct neurone**)malloc(size * sizeof(struct neurone));

    // Initialize each neurone in the layer
    for (int i = 0; i < size; ++i) {
        
        myLayer.data[i] = (struct neurone*)malloc(sizeof(struct neurone));

        myLayer.data[i]->bias = init_weights();  // You can set the initial bias to any value

        // Allocate memory for the array of weights for each neurone
        myLayer.data[i]->weights = (float*)malloc(prev_layer_size * sizeof(float));
        // Initialize weights as needed
        for (int j = 0; j < prev_layer_size; ++j) {
            myLayer.data[i]->weights[j] = init_weights();  // You can set the initial weight to any value
        }
    }

    // Allocate memory for the array to store forward results
    myLayer.result_forward = (float*)malloc(size * sizeof(float));
        // Initialize each neurone in the layer
    for (int i = 0; i < size; ++i) {
        myLayer.result_forward[i] = 0.0;  //initialize result
    }
    return myLayer;
}




//accuracy 
float accuracy(int size, float y_pred[], float target[]){
    float s = 0 ;
    for (size_t i = 0; i < size; i++)
    {
        if (y_pred[i] == target[i])
        {
            s+=1;
        } 
    }
    return s/size ;
}

///////////////////////////////////////////////////////////////////scaling////////////////////////////////

//calculate the square 
float nsquare(float number){
    return number * number ;
}

float find_min(int vector_size, float vect[vector_size]){
    int min = vect[0];
    for(int i=1 ; i < vector_size ; i++){
        if ( vect[i] < min)
        {
            min = vect[i];
        }
    }
    return min;
}

float find_max(int vector_size, float vect[vector_size]){
    int max = vect[0];
    for(int i=1; i < vector_size ; i++){
        if ( vect[i] > max )
        {
            max = vect[i];
        }
    }
    return max ;
}

//tri
float tri(int vector_size, float vect[vector_size]){
    int aux;
    for(int i = 1 ; i < vector_size ; i++){
        for(int j = i ; j < vector_size ; j++){
            if (vect[j] < vect[i])
            {
                aux = vect[i];
                vect[i] = vect[j];
                vect[j] = aux ;
            }
        }
    }
}

//median
float median(int vector_size, float tri_vect[vector_size]){
    return tri_vect[vector_size / 2];
}

//iqr : X_75 - X_25
float iqr(int vector_size, float tri_vect[vector_size]){
    // table[vector_size * 0.75] - table[vector_size * 0.25]  == > 75% - 25 %
    return tri_vect[( vector_size*3)/4] - tri_vect[vector_size / 4] ;
}

// calculate the mean of a column
float mean(int vector_size, float vect[vector_size]){
    float s = 0 ;
    for(int i = 0 ; i < vector_size ; i++){
        s+= vect[i];
    }
    return (s / vector_size) ;
}

//calculate the standard deviation
float standard_deviation(int vector_size, float vect[vector_size], float x_bar){
    float s=0;
    // calculate sum (x- x_bar)²
    for(int i=0 ; i < vector_size ; i++){
        s+= nsquare(vect[i] - x_bar);
    }
    // return square( sum( x - x_bar)²/ n)
    return sqrt( s / vector_size) ; 
}

// extract a column y_label 
void extract_column(int row_size, int columns_size, float tab[row_size][columns_size],float result[row_size],int idx_column){
    for(int i=0 ; i <row_size ; i++){
        result[i]= tab[i][idx_column];
    }
}


//extract a row "x_i" from training data 
void extract_row(int row_size, int columns_size,float train_data[row_size][columns_size], float row[columns_size], int idx_row){
    for(int i=0; i<columns_size ;i++){
        row[i]= train_data[idx_row][i];
    }
}

// normalization : standard sclaer 
void standard_scaler(int row_size, int columns_size, float table[row_size][columns_size], float mean_table[columns_size], float std_table[columns_size]){

    //calculate mean of every column (save it in table)
    for( int index_column=0; index_column < columns_size ; index_column++){
        //get the column 
        float y_i[row_size];
        extract_column(row_size, columns_size, table, y_i, index_column);

        //calculate mean of the column
        //int x_bar = mean(row_size,y_i);

        //save the mean
        mean_table[index_column] = mean(row_size,y_i);
    }

    //calculate standard deviation of every column 
    for( int index_column = 0 ; index_column < columns_size ; index_column++){
        //get the column 
        float y_i[row_size];
        extract_column(row_size, columns_size, table, y_i, index_column);

        //calculate std of the column
        //int std = standard_deviation(row_size, y_i, mean_table[index_column]);

        //save the std 
        std_table[index_column] = standard_deviation(row_size, y_i, mean_table[index_column]);
    }

    //standardization
    // fix column  
    for (int col_idx = 0; col_idx < columns_size; col_idx++)
    {
        for(int row_idx = 0 ; row_idx < row_size ; row_idx++){
            table[row_idx][col_idx] = (table[row_idx][col_idx] - mean_table[col_idx]) / std_table[col_idx] ; 
        }
    }        
}

// transform standard scaler 
void transform_standard_scaler(int row_size ,int column_size, float table[row_size][column_size], float mean_table[column_size], float std_table[column_size] ){
    for(int row_idx = 0 ; row_idx < row_size ; row_idx++){
        for(int col_idx = 0 ; col_idx < column_size ; col_idx){
            // x' = ( x - mean ) / std
            table[row_idx][col_idx] = (table[row_idx][col_idx] - mean_table[col_idx]) / std_table[col_idx];
        }
    }
}

// noramlization : min max scaling 
void min_max_scaling(int row_size, int column_size, float table[row_size][column_size], float min_table[column_size], float max_table[column_size]){

        // find the min and max values for each column
        for( int index_column = 0; index_column < column_size ; index_column++){
            //get column
            float y_i[row_size];
            extract_column(row_size, column_size, table, y_i, index_column);

            //get the max value 
            max_table[index_column] = find_max(row_size, y_i);

            //get the min value
            min_table[index_column] = find_min(row_size, y_i);

        }

        //min_max_scaling
        //fix row
        for(int row_idx=0 ; row_idx < row_size ; row_idx++){
            for( int col_idx=0; col_idx < column_size ; col_idx++){
                // (x- min / max - min) 
                table[row_idx][col_idx] = (table[row_idx][col_idx] - min_table[col_idx])/(max_table[col_idx] - min_table[col_idx]);
            }
        }
}

//transform min_max_scaler
void transform_min_max_scaler(int row_size, int column_size, float table[row_size][column_size], float min_table[column_size], float max_table[column_size]){
    for(int row_idx = 0 ; row_idx < row_size ; row_idx++){
        for(int col_idx = 0 ; col_idx < column_size ; col_idx++){

            //x' = ( x - min ) / (max - min)
            table[row_idx][col_idx]= (table[row_idx][col_idx] - min_table[col_idx]) / (max_table[col_idx] - min_table[col_idx]) ;
        }
    }
}

//robust scaler 
void robust_scaler(int row_size, int column_size, float table[row_size][column_size],float median_table[column_size], float iqr_table[column_size]){
   
   // find iqr and median for each column
   for(int column_idx = 0; column_idx < column_size ; column_idx ++){

        //get column 
        float y_i[row_size];
        extract_column(row_size, column_size, table, y_i, column_idx);

        //tri column
        tri(row_size, y_i);

        //calculate median
        median_table[column_idx] = median(row_size, y_i); 

        //calculate iqr
        iqr_table[column_idx] = iqr(row_size, y_i);
   }
   
   //robust scaler
   for(int row_idx=0 ; row_idx < row_size; row_idx++){
        for(int col_idx=0 ; col_idx < column_size ; col_idx++){

            // x' = ( x - median) / iqr
            table[row_idx][col_idx] = (table[row_idx][col_idx] - median_table[col_idx]) / iqr_table[col_idx] ;
        }
   }
}

//transform robust scaler
void transform_robust_scaler(int row_size , int column_size , float table[row_size][column_size], float median_table[column_size], float iqr_table[column_size]){
    for( int row_idx = 0 ; row_idx < row_size ; row_idx++){
        for( int col_idx = 0 ; col_idx < column_size ; col_idx++){
            // x' = (x - median ) / iqr
            table[row_idx][col_idx]= (table[row_idx][col_idx] - median_table[col_idx]) /  iqr_table[col_idx] ;
        }
    }
}



/////////////////////////////////////////////////////////////////////splitdata.c//////////////////////////////////////////:


//read_csv
void read_csv(const char *filename ,int row_size,int column_size,float table[row_size][column_size], char columns[][50] ){

    FILE* file = fopen(filename, "r");
    //FILE *file= fopen("card_transdata2.csv","r");
    printf("\ndone");    
    if (file == NULL){
        perror("Unable to open the file please check its existence");
        exit(1);
    }

    char line[20000];

    int i=0;

    while(fgets(line, sizeof(line), file)){
        //printf("\ndone : %d",i);
        char *token;
        int j=0;   //idx for table
        int k=0;   // idx for columns

        token = strtok(line,",");

        while(token != NULL){
            while(token =="\n" | token == "\t" | token == "\0"){
                //skip the \n 
                token = strtok(NULL,",");

            }
            //do not change columns labels to float; 
            if(i ==0 ){
                strcpy(columns[k++],token);

                token = strtok(NULL,",");
            }
            else{
                float val;
                val = atof(token);
                table[i-1][j] = val ;

                token = strtok(NULL,",");
                j++ ;
                //printf("\t%f",val);


            }

        }
        //printf("\n");
        

        i++;
    }
    //printf("\n");

    fclose(file);
}



//display the data 
void display(int row_size, int column_size, float tab[row_size][column_size], char label[column_size][50]){


    for(int idx=0;idx < column_size;idx++){
        printf("%s\t",label[idx]);
    }
    printf("\n");
    for(int idx1=0 ; idx1 < row_size ; idx1++){
        //printf("row: %d\n",idx1);
        for(int idx2=0 ; idx2 < column_size ; idx2++){
            //printf("ligne :%d , colonne : %d:\t",idx1,idx2);
            printf("%f\t",tab[idx1][idx2]);
        }
        printf("\n");
    }
    printf("sizeof table : %d\n", row_size);
}



void shuffle(int row_size,int column_size, float table[row_size][column_size]){
    srand(time(NULL));
    if (row_size>0)
    {
        for(int i=0; i< row_size; i++){
            //int row2= i + rand()/(RAND_MAX/(row_size -i)) ;
            int row2= rand()%row_size ;
            for (int j=0; j < column_size; j++)
            {
                int aux = table[i][j];
                table[i][j] = table[row2][j];
                table[row2][j] = aux ;
            }
        }
    }
}


//get the X data ( data used to predict )
void get_x_train(int start, int x_size, int column_size, int y_index,float table[][column_size], float x[][column_size-1]){
    for(int i=start; i< x_size ; i++){
        int k=0;
        for (int j=0; j < column_size; j++)
        {

            if ( j != 7)
            {   //printf("%f",table[i][k]);
                //printf("\t");
                x[i][k]=table[i][j];
                //printf("%f",table[i][k]);
                //printf("\t");

                k++;    
            }

        }
        //printf("\n");
    }    
}

//get x_test
void get_x_test(int start, int x_size, int column_size, int y_index,float table[][column_size], float x[][column_size-1]){
    
    int index_table = 0;    //iterate in table 
    for(int i=0; i< x_size ; i++){
        int k=0;
        
        for (int j=0; j < column_size; j++)
        {

            if ( j != 7)
            {   //printf("%f",table[i][k]);
                //printf("\t");
                x[i][k]=table[index_table][j];
                //printf("%f",table[i][k]);
                //printf("\t");

                k++;    
            }

        }
        index_table++;
        //printf("\n");
    }    
}

//train_test_split
void split_data(int row_size, int column_size, float table[row_size][column_size], float x_train[][column_size-1], float y_train[], float x_test[][column_size -1], float y_test[], float test_precentage, int y_index){
                    //shuffle the data
                    shuffle(column_size, row_size, table);

                    // split data to train and test 
                    int test_size= test_precentage * row_size;
                    int train_size = row_size - test_size;
                    

                    //get the X_train
                    get_x_train(0, train_size, column_size, y_index, table, x_train);




                    //get the y_train 
                    extract_column(train_size, column_size, table, y_train, y_index);

                    //get the x_test
                    printf("############################################################################################\n");
                    printf("############################################################################################\n");
                    get_x_test(train_size, row_size, column_size, y_index, table, x_test);
                    //get the y_test
                    extract_column(test_size, column_size, table, y_test, y_index);
                    
                }

