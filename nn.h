#ifndef NN_H_
#define NN_H_
#include<stddef.h>
#include<stdio.h>
#include<assert.h>
#include<malloc.h>
#include<stdlib.h>
#include<math.h>
#include<time.h>
#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif
#ifndef NN_ASSERT
#define NN_ASSERT assert
#endif
    #define ARRAY_LEN(xs) sizeof((xs))/sizeof((xs)[0])
    float float_rand();
    float sigmoidf(float x);
    typedef struct
    {
        size_t rows;
        size_t cols;
        size_t stride;
        float * es;
    }Mat;
    #define Mat_AT(m,i,j) (m).es[(i)*(m).stride + (j)]
    Mat mat_alloc(size_t rows , size_t cols);
    void sig(Mat m);
    Mat mat_row(Mat m , size_t rows);
    void mat_copy(Mat dst , Mat src);
    void mat_fill (Mat m , float x);
    void mat_rand(Mat m , float high , float low);
    void mat_dot(Mat dst , Mat a , Mat b);
    void mat_sum(Mat dst , Mat a);
    #define MAT_PRINT(m) mat_print(m , #m , 0)
    void mat_print(Mat m , char * name , size_t padding);
    typedef struct {
    size_t count;
    Mat *ws;
    Mat *bs;
    Mat *as;
    } NN;
    #define NN_INPUT(nn) (nn).as[0]
    #define NN_OUTPUT(nn) (nn).as[(nn).count]
    NN nn_alloc(size_t * arch , size_t arch_count);
    void nn_print(NN nn , const char * name);
    #define NN_PRINT(nn) nn_print(nn , #nn)
    void nn_rand(NN nn , float high , float low);
    void nn_forward(NN nn);
    float nn_cost(NN nn , Mat ti , Mat to);
    void nn_finite_diff(NN m , NN g , float eps , Mat ti , Mat to);
    void nn_learn(NN nn , NN g , float lr);
#endif 
#ifdef NN_IMPLEMENTATION
    float sigmoidf(float x){
        return 1/(1+exp(-x));
    }
    void sig(Mat m){
        for (size_t i = 0; i < m.rows; i++)
        {
            for (size_t j = 0; j < m.cols; j++)
            {
              Mat_AT(m,i,j) = sigmoidf(Mat_AT(m,i,j));
            }  
        }
    }
    float float_rand(){  
        return (float)rand() / (float)RAND_MAX;
    }
    
    void mat_copy(Mat dst , Mat src){
        NN_ASSERT(dst.cols == src.cols);
        NN_ASSERT(dst.rows == src.rows);
        for (size_t i = 0; i < dst.rows; i++)
        {
            for (size_t j = 0; j < dst.cols; j++)
            {
                Mat_AT(dst ,i,j) = Mat_AT(src,i,j);
            }
            
        }
        
    }
    Mat mat_alloc(size_t rows , size_t cols){
        Mat m;
        m.rows = rows;
        m.cols = cols;
        m.stride = cols;
        m.es = NN_MALLOC(sizeof(*m.es)*rows*cols);
        NN_ASSERT(m.es != NULL);
        return m;
    }
    Mat mat_row(Mat m , size_t rows){
        return (Mat){
            .rows =1,
            .cols = m.cols,
            .stride = m.stride,
            .es = &Mat_AT(m,rows,0)
        };
    }
    void mat_dot(Mat dst , Mat a , Mat b){
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(dst.cols == b.cols);
    NN_ASSERT(dst.rows == a.rows);

    mat_fill(dst, 0.0f);   

    for (size_t i = 0; i < dst.rows; i++)
        for (size_t j = 0; j < dst.cols; j++)
            for (size_t k = 0; k < a.cols; k++)
                Mat_AT(dst,i,j) += Mat_AT(a,i,k)*Mat_AT(b,k,j);
}

    void mat_sum(Mat dst , Mat a){
        NN_ASSERT(dst.rows == a.rows);
        NN_ASSERT(dst.cols == a.cols);
        for (size_t i = 0; i < dst.rows; i++)
        {
            for (size_t j = 0; j < dst.cols; j++)
            {
               Mat_AT(dst ,i ,j) += Mat_AT(a,i,j);
            }  
        }
    }
    void mat_print(Mat m , char * name , size_t padding){
        printf("%*s%s = [\n" , (int)padding , "" , name);
        for (size_t i = 0; i < m.rows; i++)
        {
            printf("%*s    " , (int)padding , "" );
            for (size_t j = 0; j < m.cols; j++)
            {
                printf("%f " , Mat_AT(m,i,j));
            }
            printf("\n");
        }
        printf("%*s]\n" , (int)padding , "");
    }
    void mat_fill(Mat m , float x){
        for (size_t i = 0; i < m.rows; i++)
        {
            for (size_t j = 0; j < m.cols; j++)
            {
                Mat_AT(m,i,j) = x;
            }
        }
    }
    void mat_rand(Mat m , float high , float low){
        for (size_t i = 0; i < m.rows; i++)
        {
            for (size_t j = 0; j < m.cols; j++)
            {
                Mat_AT(m,i,j) = float_rand()*(high-low) + low;
            }
        }
    }
    NN nn_alloc(size_t * arch , size_t arch_count){
    NN_ASSERT(arch_count > 0);

    NN nn;
    nn.count = arch_count - 1;

    nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));

    NN_ASSERT(nn.ws && nn.bs && nn.as);

    nn.as[0] = mat_alloc(1, arch[0]);

    for (size_t i = 1; i <= nn.count; i++) {
        nn.ws[i-1] = mat_alloc(nn.as[i-1].cols, arch[i]);
        mat_fill(nn.ws[i-1] , 0);
        nn.bs[i-1] = mat_alloc(1, arch[i]);
        mat_fill(nn.bs[i-1] , 0);
        nn.as[i]   = mat_alloc(1, arch[i]);
        mat_fill(nn.as[i] , 0);
    }

    return nn;
}

    void nn_print(NN nn , const char * name){
        char buf[256];

        printf("%s = [ \n" , name);
        for (size_t i = 0; i < nn.count; i++)
        {
            snprintf(buf , sizeof(buf) , "ws%zu" , i);
            mat_print(nn.ws[i] , buf , 4);
            snprintf(buf , sizeof(buf) , "bs%zu" , i);
            mat_print(nn.bs[i] , buf , 4);
        }
        
        printf("] \n");
    }
    void nn_rand(NN nn , float high , float low){
        for (size_t i = 0; i < nn.count; i++)
        {
            mat_rand(nn.ws[i] , high , low);
            mat_rand(nn.bs[i] , high , low);
        }
        
    }
    void nn_forward(NN nn){
    for (size_t i = 0; i < nn.count; i++) {
        mat_fill(nn.as[i+1], 0.0f);
        mat_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
        mat_sum(nn.as[i+1], nn.bs[i]);
        sig(nn.as[i+1]);
    }
}

    float nn_cost(NN nn , Mat ti , Mat to){
        assert(ti.rows == to.rows);
        assert(to.cols == NN_OUTPUT(nn).cols);
        size_t n = ti.rows;
        float c = 0;
        for (size_t i = 0; i < n; i++)
        {
            Mat x = mat_row(ti,i);
            Mat y = mat_row(to,i);
            mat_copy(NN_INPUT(nn) , x);
            nn_forward(nn);
            size_t q = to.cols;
            for (size_t j = 0; j < q; j++)
            {
                float d = Mat_AT(NN_OUTPUT(nn) , 0 , j) - Mat_AT(y , 0,j);
                c += d*d;
            }
        }
        return c/n;
    }
    void nn_finite_diff(NN nn , NN g , float eps , Mat ti , Mat to){
        float saved;
        float c = nn_cost(nn , ti ,to);
        for (size_t i = 0; i < nn.count; i++)
        {
            for (size_t j = 0; j < nn.ws[i].rows; j++)
            {
                for (size_t k = 0; k < nn.ws[i].cols; k++)
                {
                    saved = Mat_AT(nn.ws[i],j,k); 
                    Mat_AT(nn.ws[i],j,k) += eps;
                    Mat_AT(g.ws[i],j,k) = (nn_cost(nn , ti ,to)-c)/eps;
                    Mat_AT(nn.ws[i],j,k) = saved;
                }   
            }

            for (size_t j = 0; j < nn.bs[i].rows; j++)
            {
                for (size_t k = 0; k < nn.bs[i].cols; k++)
                {
                    saved = Mat_AT(nn.bs[i],j,k); 
                    Mat_AT(nn.bs[i],j,k) += eps;
                    Mat_AT(g.bs[i],j,k) = (nn_cost(nn , ti ,to)-c)/eps;
                    Mat_AT(nn.bs[i],j,k) = saved;
                }
                
            }
        }
        
    }
    void nn_learn(NN nn , NN g , float lr){
        for (size_t i = 0; i < nn.count; i++)
        {
            for (size_t j = 0; j < nn.ws[i].rows; j++)
            {
                for (size_t k = 0; k < nn.ws[i].cols; k++)
                {
                    Mat_AT(nn.ws[i] , j,k) -= lr*Mat_AT(g.ws[i],j,k);
                }   
            }

            for (size_t j = 0; j < nn.bs[i].rows; j++)
            {
                for (size_t k = 0; k < nn.bs[i].cols; k++)
                {
                    Mat_AT(nn.bs[i] , j,k) -= lr*Mat_AT(g.bs[i],j,k);
                }
                
            }
        }
    }

#endif