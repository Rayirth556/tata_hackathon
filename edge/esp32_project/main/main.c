#include <stdio.h>
#include <string.h>
#include "esp_timer.h"
#include "esp_log.h"
#include "model_reg.h"
#include "model_clf.h"

static const char *TAG = "VitalEdge";

void app_main(void) {
    ESP_LOGI(TAG, "Starting VitalEdge ESP32 Edge Inference Benchmark");
    
    // Sample features for HUST Cell 1-1 (extracted at cycle 100)
    double input[13] = {
        1.158849e+00,  // QD_100
        3.455126e-02,  // IR_cycle2
        3.373422e-02,  // IR_cycle100
        -8.170401e-04, // IR_diff
        1.828277e-01,  // dVdQ_var_10
        2.074819e-01,  // dVdQ_var_100
        2.465424e-02,  // dVdQ_var_diff
        2.043448e+00,  // I_var_10
        2.055635e+00,  // I_var_100
        1.218672e-02,  // I_var_diff
        5.610000e+02,  // chargetime_s_mean_2to6
        -1.280764e-04, // fade_slope
        1.170966e+00   // fade_intercept
    };
    
    // Warm-up run
    double reg_pred = predict_knee_cycle(input);
    double clf_out[2];
    predict_knee_early(input, clf_out);
    
    // --- Benchmark Regression ---
    int64_t start_time = esp_timer_get_time();
    double dummy_sum = 0.0;
    int iterations = 10000;
    
    for (int i = 0; i < iterations; ++i) {
        dummy_sum += predict_knee_cycle(input);
    }
    int64_t end_time = esp_timer_get_time();
    double reg_latency_us = (double)(end_time - start_time) / iterations;
    
    // --- Benchmark Classification ---
    double dummy_clf_sum = 0.0;
    start_time = esp_timer_get_time();
    for (int i = 0; i < iterations; ++i) {
        predict_knee_early(input, clf_out);
        dummy_clf_sum += clf_out[1];
    }
    end_time = esp_timer_get_time();
    double clf_latency_us = (double)(end_time - start_time) / iterations;
    
    // Final check run
    predict_knee_early(input, clf_out);
    
    // Log results
    ESP_LOGI(TAG, "=== Benchmark Results ===");
    ESP_LOGI(TAG, "Regression RUL Knee-Cycle Prediction: %.4f", reg_pred);
    ESP_LOGI(TAG, "Classification Early Degradation Prob: %.4f (Class 0: %.4f)", clf_out[1], clf_out[0]);
    ESP_LOGI(TAG, "Regression Latency: %.4f us per inference", reg_latency_us);
    ESP_LOGI(TAG, "Classification Latency: %.4f us per inference", clf_latency_us);
    ESP_LOGI(TAG, "=========================");
    
    // Prevent compiler loop optimization
    if (dummy_sum == 123456.7 && dummy_clf_sum == 765432.1) {
        printf("dummy: %f %f\n", dummy_sum, dummy_clf_sum);
    }
}
