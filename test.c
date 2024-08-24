#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <string.h>
#include "AiMine.h"


#define  MAX_ROWS 10000 
#define  MAX_COLUMNS 8

#define target_index 7



int main() {
    float table[MAX_ROWS][MAX_COLUMNS] ;
    char columns[MAX_COLUMNS][50];

    float train_data[MAX_ROWS][MAX_COLUMNS-1];
    float target[MAX_ROWS];
    printf("ok\n");
    read_csv("card_transdata2.csv",10000,MAX_COLUMNS ,table, columns);

    //printf("before shuffling :\n");
    printf("head(10) :\n");
    display(10, MAX_COLUMNS, table, columns);

    //shuffle(MAX_ROWS, MAX_COLUMNS, table);
    //printf("\n after suffling :\n");
    //display(10, MAX_COLUMNS, table, columns);




    //train_test_split
    float x_train[MAX_ROWS][MAX_COLUMNS-1];
    float x_test[MAX_ROWS][MAX_COLUMNS-1];
    float y_train[MAX_ROWS];
    float y_test[MAX_ROWS];

    split_data(10000, MAX_COLUMNS, table, x_train, y_train, x_test, y_test, 0.3, target_index);
    
    
    printf("after standard_scaler (X_train): \n ");
    int xtrain_rowsize = 1000 * 0.3;
    float mean_table[xtrain_rowsize];
    float std_table[xtrain_rowsize];
    standard_scaler(xtrain_rowsize, MAX_COLUMNS -1, x_train, mean_table, std_table);


    display(10, MAX_COLUMNS -1, x_train, columns);
    printf("head of x_train\n");
    //display(30, MAX_COLUMNS-1 , x_train, columns);

    // Define dense layers
    // Example: Initializing three layers
    struct layer layer1, layer2, layer3;

    /*/ Initialize each layer
    initializeLayer(&layer1, 5, "ReLu");
    initializeLayer(&layer2, 5, "ReLu");
    initializeLayer(&layer3, 3, "sigmoid");
    */
    // Use the layers...
    //input (first line in main)
    int size_table[]={3,5,1};
    //char* activation_table={"ReLu","ReLu","sigmoid"}
    int number_layers = 3;

    // Allocate memory for delta table
    //struct delta* delta_table = (struct delta*)malloc(number_layers * sizeof(struct delta));

    
    
    //define table 
    struct delta delta_table[number_layers];
    struct layer layerTable[number_layers];
    

    // Fill the table with data

    layerTable[0] = initializeLayer( 5,7, "ReLu");
    layerTable[1] = initializeLayer( 3,5, "ReLu");
    layerTable[2] = initializeLayer( 1,3, "sigmoid");

    printf("ok\n");
    printf("tables[0].size = %d \n",layerTable[0].size);
    delta_table[0] = initializeDelta(layerTable[0].size);
    delta_table[1] = initializeDelta(layerTable[1].size);
    delta_table[2] = initializeDelta(layerTable[2].size);
    printf("after initialization : \n");
    for (size_t i = 0; i < 3; i++)
    {
        printf("size of delta_table[%d] = %d\n",i,delta_table[i].size);
    }



    /*/ Initialize delta table for each layer
    for (size_t layer = 0; layer < number_layers; layer++) {
        delta_table[layer] = initializeDelta(layerTable[layer].size);
        printf("size %d : %d\n",layer,layerTable[layer].size);
    }*/

    /*for (size_t i = 0; i < number_layers; i++)
    {
        printf("size %d :%d \t",i,layerTable[i].size);
    }*/
    printf("\n");

    printf("train_manually\n");
    printf("\t");



    int row_size=MAX_ROWS,column_size=MAX_COLUMNS - 1;
    float learning_rate=0.1;
    int epochs= 20;

    // Start training
    for (size_t ep = 0; ep < epochs; ep++) {
        printf("\nepoch %d :  ", ep);
        float error = 0;
        float y_pred[MAX_ROWS];
        float y_true[MAX_ROWS];
        int idx=0 ;
        float y_pred_i;

        for (size_t iter = 0; iter < MAX_ROWS; iter++) {
            //printf("\n%d", iter);
            // Extract row
            float x_i[column_size];
            extract_row(row_size, column_size, x_train, x_i, iter);

            // Forward pass
            Forward(column_size, number_layers, x_i, layerTable, &y_pred_i);
            // Calculate loss
            error += cross_entropy_loss(y_pred_i, y_train[iter]);
            
            // Calculate delta values
            delta(number_layers, layerTable, delta_table, y_train[iter]);

            //Weight update 
            weight_update(layerTable,delta_table, number_layers, column_size, learning_rate);

            printf("pred : %f   ,  true = %f\n",layerTable[2].result_forward[0],y_train[idx]);
            y_true[idx] = y_train[idx];
            y_pred[idx] = round(layerTable[2].result_forward[0]);
            //iterator
            idx++;
        }
        printf("accuracy = %f  ,  loss = %f \n",accuracy(idx, y_pred, y_true),error);
        //printf("pred : %f   ,  true = %f\n",layerTable[2].result_forward[0],y_train[idx -1]);

        // Update weights
        for (size_t i = 0; i < 5; i++)
        {
            printf("weight[%d] : %f \t",i,layerTable[0].data[0]->weights[i]);
        }
    }
    // Free memory for delta
    for (size_t layer = 0; layer < number_layers; layer++) {
        freeDelta(&delta_table[layer]);
    }


    // Free memory for delta table
    freeDeltaTable(delta_table, number_layers);


    // Free memory for layers
    freeLayer(&layer1);
    freeLayer(&layer2);
    freeLayer(&layer3);

    // Free memory for layers table
    free_table_layer(layerTable);

    
    // Free the memory for each layer
    for (int i = 0; i < number_layers; ++i) {
        freeLayer(&layerTable[i]);
    }

    // Free the memory for the array of pointers
    free(layerTable);

    return 0;
}

