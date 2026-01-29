#include<stdio.h>
#include<stdlib.h>
#include<time.h>

int train[][2] = {
    {0,0},
    {1,1},
    {2,4},
    {3,9},
    {4,16}
};
#define train_count (sizeof(train)/sizeof(train[0]))

float rand_float (){
    return (float)rand() / (float)RAND_MAX;
}

float cost_func (float w ,float b){
    float result = 0.0;
    for (size_t i = 0; i < train_count; i++)
    {
        float x = train[i][0];
        float y_pred = w*x*x + b;
        float error = y_pred - train[i][1];
        float error_cubed = error * error;
        result += error_cubed;
    }
    result /= 2*train_count;
    return result;
}
int main(){
    srand(time(0));
    float w = rand_float()*100.0f;
    float b = rand_float()*100.0f;
    float eps = 1e-3;
    float lr = 2e-2;
    printf("w = %f ,b = %f \n" , w , b);
    for (size_t i = 0; i < 3000; i++)
    {
        float dw = (cost_func(w+eps , b)-cost_func(w,b))/eps;
        float db = (cost_func(w , b+eps)-cost_func(w,b))/eps;
        w -= lr * dw;
        b -= lr * db;
        //printf("%f \n" , cost_func(w,b));
    }
    // printf("---------------- \n");
     printf("w = %f , b = %f" , w ,b);
}