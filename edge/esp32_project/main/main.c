#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "esp_timer.h"
#include "esp_log.h"
#include "model_reg.h"
#include "model_clf.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define BUF_SIZE 1024

static const char *TAG = "VitalEdge";

void app_main(void) {
    ESP_LOGI(TAG, "VitalEdge Interactive Serial Inference Loop Ready (115200 Baud)");
    
    char rx_buf[BUF_SIZE];
    
    while (1) {
        int len = 0;
        while (len < BUF_SIZE - 1) {
            int c = fgetc(stdin);
            if (c == EOF || c < 0) {
                clearerr(stdin);
                vTaskDelay(pdMS_TO_TICKS(10));
                continue;
            }
            if (c == '\n') {
                break;
            }
            if (c != '\r') {
                rx_buf[len++] = (char)c;
            }
        }
        rx_buf[len] = '\0';
        
        if (len > 0) {
            // Parse 13 comma-separated floats
            double input[13];
            int idx = 0;
            char *token = strtok(rx_buf, ",");
            while (token != NULL && idx < 13) {
                input[idx++] = atof(token);
                token = strtok(NULL, ",");
            }
            
            if (idx == 13) {
                // Run Regression & measure latency
                int64_t start_time = esp_timer_get_time();
                double reg_pred = predict_knee_cycle(input);
                int64_t end_time = esp_timer_get_time();
                double reg_latency_us = (double)(end_time - start_time);
                
                // Run Classification & measure latency
                double clf_out[2];
                start_time = esp_timer_get_time();
                predict_knee_early(input, clf_out);
                end_time = esp_timer_get_time();
                double clf_latency_us = (double)(end_time - start_time);
                
                // Print JSON payload back to the UART
                printf("{\"knee_cycle\": %.4f, \"early_prob\": %.4f, \"reg_latency_us\": %.2f, \"clf_latency_us\": %.2f}\n", 
                       reg_pred, clf_out[1], reg_latency_us, clf_latency_us);
                fflush(stdout);
            } else {
                printf("Error: Expected 13 features but parsed %d\n", idx);
                fflush(stdout);
            }
        }
    }
}
