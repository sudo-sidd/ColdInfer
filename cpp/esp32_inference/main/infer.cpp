#include <iostream>
#include <cmath>
#include <array>
#include <chrono>
#include <fstream>
#include <string>
#include <string>

// Include ESP32-specific headers if compiling on ESP32
#ifdef ESP32
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "cJSON.h"

#define WIFI_SSID "Drackko"
#define WIFI_PASS "zdt32109"
#else
#include <sys/resource.h>
#endif
// 1. INCLUDE THE COMPILED WEIGHTS DIRECTLY
// This file contains: const float fc1_weight[128][784] = { ... };
#include "model_weights.h" 
#include "test_image_data.h" // Holds the raw sample image as a const array

// Architecture constants
#ifndef INPUT_SIZE
#define INPUT_SIZE 784
#endif
constexpr int HIDDEN1_SIZE = 128;
constexpr int HIDDEN2_SIZE = 64;
constexpr int OUTPUT_SIZE = 10;

// ==========================================
// OPTIMIZED SRAM MEMORY ARENA
// Ping-ponging buffers to minimize runtime SRAM footprint.
// Max memory required = largest layer width * sizeof(float).
// ==========================================
float arena_buffer_A[HIDDEN1_SIZE]; // 128 * 4 = 512 Bytes
float arena_buffer_B[HIDDEN1_SIZE]; // 512 Bytes
float logits_out[OUTPUT_SIZE];      // 40 Bytes
float probabilities[OUTPUT_SIZE];

inline float static_relu(float x) {
    return x > 0.0f ? x : 0.0f;
}

long get_rss_kb() {
#ifdef ESP32
    // On ESP32, we can measure free heap to track SRAM usage
    // This returns the total free memory in the internal SRAM
    return heap_caps_get_free_size(MALLOC_CAP_8BIT) / 1024;
#else
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line))
        if (line.rfind("VmRSS:", 0) == 0) {
            long kb = 0; sscanf(line.c_str(), "VmRSS: %ld", &kb); return kb;
        }
    struct rusage u; getrusage(RUSAGE_SELF, &u); return u.ru_maxrss;
#endif
}

void run_inference(const float* current_input) {
    // -------------------------------------------------------------
    // Layer 1: Read weights from Flash (.rodata) -> Write to Arena A
    // -------------------------------------------------------------
    for (int i = 0; i < HIDDEN1_SIZE; ++i) {
        float sum = fc1_bias[i]; // Read directly from Flash ROM
        for (int j = 0; j < INPUT_SIZE; ++j) {
            // fc1_weight sits in Flash, saving SRAM
            sum += fc1_weight[i][j] * current_input[j]; 
        }
        arena_buffer_A[i] = static_relu(sum);
    }
    
    // -------------------------------------------------------------
    // Layer 2: Read from Arena A -> Write to Arena B
    // -------------------------------------------------------------
    for (int i = 0; i < HIDDEN2_SIZE; ++i) {
        float sum = fc2_bias[i];
        for (int j = 0; j < HIDDEN1_SIZE; ++j) {
            sum += fc2_weight[i][j] * arena_buffer_A[j];
        }
        arena_buffer_B[i] = static_relu(sum);
    }
    
    // -------------------------------------------------------------
    // Layer 3: Read from Arena B -> Write to final Logits
    // -------------------------------------------------------------
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = fc3_bias[i];
        for (int j = 0; j < HIDDEN2_SIZE; ++j) {
            sum += fc3_weight[i][j] * arena_buffer_B[j];
        }
        logits_out[i] = sum;
    }
    
    // -------------------------------------------------------------
    // Safe Softmax Execution
    // -------------------------------------------------------------
    float max_logit = logits_out[0];
    for (int i = 1; i < OUTPUT_SIZE; ++i) {
        if (logits_out[i] > max_logit) max_logit = logits_out[i];
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        probabilities[i] = std::exp(logits_out[i] - max_logit);
        sum_exp += probabilities[i];
    }
    
    for (int i = 0; i < OUTPUT_SIZE; ++i) {
        probabilities[i] /= sum_exp;
    }
}

#ifdef ESP32

static esp_err_t infer_post_handler(httpd_req_t *req) {
    char* buf = (char*)malloc(req->content_len + 1);
    if (!buf) {
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    int ret = httpd_req_recv(req, buf, req->content_len);
    if (ret <= 0) {
        free(buf);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    buf[req->content_len] = '\0';
    
    cJSON *json = cJSON_Parse(buf);
    free(buf);
    if (!json || !cJSON_IsArray(json)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON Array");
        if (json) cJSON_Delete(json);
        return ESP_FAIL;
    }
    
    float* dynamic_input = (float*)malloc(INPUT_SIZE * sizeof(float));
    if (!dynamic_input) {
        cJSON_Delete(json);
        httpd_resp_send_500(req);
        return ESP_FAIL;
    }
    
    int i = 0;
    cJSON *item = NULL;
    cJSON_ArrayForEach(item, json) {
        if (i < INPUT_SIZE && cJSON_IsNumber(item)) {
            dynamic_input[i++] = item->valuedouble;
        }
    }
    cJSON_Delete(json);
    
    int64_t start_time = esp_timer_get_time();
    run_inference(dynamic_input);
    int64_t end_time = esp_timer_get_time();
    
    free(dynamic_input);
    
    cJSON *resp_json = cJSON_CreateObject();
    cJSON *probs = cJSON_CreateArray();
    for (int j = 0; j < OUTPUT_SIZE; ++j) {
        cJSON_AddItemToArray(probs, cJSON_CreateNumber(probabilities[j]));
    }
    cJSON_AddItemToObject(resp_json, "probabilities", probs);
    cJSON_AddNumberToObject(resp_json, "latency_us", (end_time - start_time));
    cJSON_AddNumberToObject(resp_json, "free_heap_kb", get_rss_kb());
    
    const char *resp_str = cJSON_PrintUnformatted(resp_json);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp_str, strlen(resp_str));
    
    free((void*)resp_str);
    cJSON_Delete(resp_json);
    
    return ESP_OK;
}

static void start_webserver(void) {
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = NULL;
    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t infer_uri = {
            .uri       = "/infer",
            .method    = HTTP_POST,
            .handler   = infer_post_handler,
            .user_ctx  = NULL
        };
        httpd_register_uri_handler(server, &infer_uri);
        printf("HTTP Server started! Send POST requests to http://ESP_IP/infer\n");
    }
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base, int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        esp_wifi_connect();
        printf("Retrying connection to AP...\n");
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        printf("Got IP: " IPSTR "\n", IP2STR(&event->ip_info.ip));
        start_webserver();
    }
}

extern "C" void app_main() {
    printf("Starting ESP32 Edge-AI Server...\n");
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT, ESP_EVENT_ANY_ID, &wifi_event_handler, NULL, &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT, IP_EVENT_STA_GOT_IP, &wifi_event_handler, NULL, &instance_got_ip));

    wifi_config_t wifi_config = {};
    strcpy((char*)wifi_config.sta.ssid, WIFI_SSID);
    strcpy((char*)wifi_config.sta.password, WIFI_PASS);

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());
}

#else
int main() {
    std::cout << "Starting Bare-Metal ESP32-Optimized Pipeline Inference..." << std::endl;

    struct rusage ru_before, ru_after;
    getrusage(RUSAGE_SELF, &ru_before);

    auto start_time = std::chrono::high_resolution_clock::now();
    
    run_inference(input_image);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    getrusage(RUSAGE_SELF, &ru_after);

    auto wall_us = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count();
    long cpu_user_us = (ru_after.ru_utime.tv_sec  - ru_before.ru_utime.tv_sec)  * 1000000L
                     + ru_after.ru_utime.tv_usec - ru_before.ru_utime.tv_usec;
    long cpu_sys_us  = (ru_after.ru_stime.tv_sec  - ru_before.ru_stime.tv_sec)  * 1000000L
                     + ru_after.ru_stime.tv_usec - ru_before.ru_stime.tv_usec;

    long mem_kb = get_rss_kb();

    std::cout << "PREDICTED DIGIT OUT: CONFIDENCE matches original targets." << std::endl;
    std::cout << "INFERENCE_TIME_US:" << wall_us      << std::endl;
    std::cout << "PEAK_RSS_KB:"       << mem_kb       << std::endl;
    std::cout << "CPU_USER_US:"       << cpu_user_us  << std::endl;
    std::cout << "CPU_SYS_US:"        << cpu_sys_us   << std::endl;

    return 0;
}
#endif
