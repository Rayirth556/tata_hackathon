#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include "model_reg.h"
#include "model_clf.h"

#define BENCHMARK_ITERATIONS 1000000

int main() {
    double input[13];
    
    // Read 13 comma-separated double features from stdin
    char line[1024];
    if (fgets(line, sizeof(line), stdin) != NULL) {
        char *token = strtok(line, ",");
        int idx = 0;
        while (token != NULL && idx < 13) {
            input[idx++] = atof(token);
            token = strtok(NULL, ",");
        }
        if (idx < 13) {
            fprintf(stderr, "Error: Only read %d of 13 features from stdin.\n", idx);
            return 1;
        }
    } else {
        fprintf(stderr, "Error: Empty input\n");
        return 1;
    }
    
    // Warm-up and basic prediction check
    double reg_pred = predict_knee_cycle(input);
    double clf_out[2];
    predict_knee_early(input, clf_out);
    
    // --- Benchmark Regression ---
    struct timespec start_time, end_time;
    double dummy_sum = 0.0;
    
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        // Feed in input and keep compiler from optimizing away the loop
        dummy_sum += predict_knee_cycle(input);
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double reg_total_time_ns = (end_time.tv_sec - start_time.tv_sec) * 1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    double reg_avg_time_us = (reg_total_time_ns / BENCHMARK_ITERATIONS) / 1e3;
    
    // --- Benchmark Classification ---
    double dummy_clf_sum = 0.0;
    clock_gettime(CLOCK_MONOTONIC, &start_time);
    for (int i = 0; i < BENCHMARK_ITERATIONS; ++i) {
        predict_knee_early(input, clf_out);
        dummy_clf_sum += clf_out[1];
    }
    clock_gettime(CLOCK_MONOTONIC, &end_time);
    
    double clf_total_time_ns = (end_time.tv_sec - start_time.tv_sec) * 1e9 + (end_time.tv_nsec - start_time.tv_nsec);
    double clf_avg_time_us = (clf_total_time_ns / BENCHMARK_ITERATIONS) / 1e3;
    
    // Run one final time to make sure outputs are correct
    predict_knee_early(input, clf_out);
    
    // Print outputs in a structured format for parser consumption
    printf("regression_prediction: %.8f\n", reg_pred);
    printf("classification_prediction_prob_0: %.8f\n", clf_out[0]);
    printf("classification_prediction_prob_1: %.8f\n", clf_out[1]);
    printf("reg_latency_us: %.6f\n", reg_avg_time_us);
    printf("clf_latency_us: %.6f\n", clf_avg_time_us);
    
    // Prevent optimization of loops
    if (dummy_sum == 12345.67 && dummy_clf_sum == 76543.21) {
        printf("dummy: %f\n", dummy_sum);
    }
    
    return 0;
}
