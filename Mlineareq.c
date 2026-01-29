#include<stdio.h>
#include<stdlib.h>
#include<time.h>
#include<math.h>
int train[][3] = {
    {0,0,0},
    {1,0,1},
    {0,1,1},
    {1,1,1}
};
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float (){
    return (float)rand() / (float)RAND_MAX;
}

float sigmoid (float x){
    return 1.f/(1.f+exp(-x));
}

float cost_func (float w1 ,float w2,float b){
    float result = 0.0;
    for (size_t i = 0; i < train_count; i++)
    {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float z = w1*x1 + w2*x2 + b;
        float y_pred = sigmoid(z);
        float error = y_pred - train[i][2];
        float error_cubed = error * error;
        result += error_cubed;
    }
    result /= 2*train_count;
    return result;
}
int main(){
    srand(time(0));
    float w1 = rand_float();
    float w2 = rand_float();
    float b = rand_float();
    float eps = 1e-3;
    float lr = 0.1f;
    printf("w1 = %f ,w2 = %f , b = %f \n" , w1 , w2 , b);
    for (size_t i = 0; i < 1000*1000; i++)
    {
        float dw1 = (cost_func(w1+eps ,w2 , b)-cost_func(w1,w2,b))/eps;
        float dw2 = (cost_func(w1,w2+eps , b)-cost_func(w1,w2,b))/eps;
        float db = (cost_func(w1 ,w2 , b+eps)-cost_func(w1,w2,b))/eps;
        w1 -= lr * dw1;
        w2 -= lr * dw2;
        b -= lr * db;
        //printf("%f \n" , cost_func(w1,w2,b));
    }
    // printf("---------------- \n");
     printf("w1 = %f , w2 = %f , b = %f \n" , w1 , w2 ,b);
     for (size_t i = 0; i < 4; i++)
     {
        float x1 = train[i][0];
        float x2 = train[i][1];
        float z = w1*x1+w2*x2+b;
        printf("%f , %f , %f \n" ,x1,x2,sigmoid(z));
     }
     
}
