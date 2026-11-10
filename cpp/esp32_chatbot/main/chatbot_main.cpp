// chatbot_main.cpp — ColdInfer ESP32 Wi-Fi Chatbot Firmware
//
// Bare-metal INT8 LSTM character-level inference server running on ESP32 Classic.
// Exposes an HTTP REST API over local Wi-Fi:
//   GET  /health  -> Returns free heap, uptime, and model info
//   POST /chat    -> Runs autoregressive text generation on device

#include <iostream>
#include <cstring>
#include <cmath>
#include <array>
#include <chrono>

#ifdef ESP32
#include "esp_timer.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/event_groups.h"
#include "esp_wifi.h"
#include "esp_event.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_http_server.h"
#include "esp_log.h"
#include "cJSON.h"

#define TAG "COLD_CHAT"

// Default Wi-Fi credentials (override at compile time or in menuconfig)
#ifndef WIFI_SSID
#define WIFI_SSID "ColdInfer_AP"
#endif

#ifndef WIFI_PASS
#define WIFI_PASS "tinyinfer123"
#endif

#define WIFI_CONNECTED_BIT BIT0
#define WIFI_FAIL_BIT      BIT1
static EventGroupHandle_t s_wifi_event_group;
static int s_retry_num = 0;
#endif

// ---------------------------------------------------------------------------
// ColdInfer model headers
// ---------------------------------------------------------------------------
#include "chatbot_weights.h"
#include "chatbot_vocab.h"
#include "lstm_kernels.h"
#include "rolling_kv.h"

#ifndef MAX_GEN_LEN
#define MAX_GEN_LEN 200
#endif

#ifndef WINDOW_SZ
#define WINDOW_SZ 8
#endif

// ---------------------------------------------------------------------------
// Static SRAM arena — zero dynamic memory allocation during inference
// ---------------------------------------------------------------------------
static float embed_out[CHATBOT_EMBED_DIM];
static float h1[CHATBOT_HIDDEN_SIZE];
static float c1[CHATBOT_HIDDEN_SIZE];
static float h2[CHATBOT_HIDDEN_SIZE];
static float c2[CHATBOT_HIDDEN_SIZE];

static float gate_i[CHATBOT_HIDDEN_SIZE];
static float gate_f[CHATBOT_HIDDEN_SIZE];
static float gate_g[CHATBOT_HIDDEN_SIZE];
static float gate_o[CHATBOT_HIDDEN_SIZE];

static float logits[CHATBOT_VOCAB_SIZE];
static float probs[CHATBOT_VOCAB_SIZE];

static RollingStateBuffer<CHATBOT_HIDDEN_SIZE, WINDOW_SZ> state1;
static RollingStateBuffer<CHATBOT_HIDDEN_SIZE, WINDOW_SZ> state2;

static char output_buf[MAX_GEN_LEN + 1];

// ---------------------------------------------------------------------------
// Model Inference Core
// ---------------------------------------------------------------------------
static void reset_hidden_states() {
    state1.reset();
    state2.reset();
    std::memset(h1, 0, sizeof(h1));
    std::memset(c1, 0, sizeof(c1));
    std::memset(h2, 0, sizeof(h2));
    std::memset(c2, 0, sizeof(c2));
}

static void forward_step(int char_idx) {
    // Step 1: Embedding lookup
    embedding_lookup<CHATBOT_VOCAB_SIZE, CHATBOT_EMBED_DIM>(
        emb_weight, emb_scale, char_idx, embed_out
    );

    // Step 2: LSTM Layer 1
    const float* h1_prev = (state1.count > 0) ? state1.last_h() : h1;
    const float* c1_prev = (state1.count > 0) ? state1.last_c() : c1;

    float h1_new[CHATBOT_HIDDEN_SIZE];
    float c1_new[CHATBOT_HIDDEN_SIZE];

    LSTMCell<CHATBOT_EMBED_DIM, CHATBOT_HIDDEN_SIZE>::step(
        lstm1_wih_i, lstm1_wih_i_scale,
        lstm1_wih_f, lstm1_wih_f_scale,
        lstm1_wih_g, lstm1_wih_g_scale,
        lstm1_wih_o, lstm1_wih_o_scale,
        lstm1_whh_i, lstm1_whh_i_scale,
        lstm1_whh_f, lstm1_whh_f_scale,
        lstm1_whh_g, lstm1_whh_g_scale,
        lstm1_whh_o, lstm1_whh_o_scale,
        lstm1_bias_i, lstm1_bias_f, lstm1_bias_g, lstm1_bias_o,
        embed_out, h1_prev, c1_prev,
        h1_new, c1_new,
        gate_i, gate_f, gate_g, gate_o
    );
    state1.push(h1_new, c1_new);

    // Step 3: LSTM Layer 2
    const float* h2_prev = (state2.count > 0) ? state2.last_h() : h2;
    const float* c2_prev = (state2.count > 0) ? state2.last_c() : c2;

    float h2_new[CHATBOT_HIDDEN_SIZE];
    float c2_new[CHATBOT_HIDDEN_SIZE];

    LSTMCell<CHATBOT_HIDDEN_SIZE, CHATBOT_HIDDEN_SIZE>::step(
        lstm2_wih_i, lstm2_wih_i_scale,
        lstm2_wih_f, lstm2_wih_f_scale,
        lstm2_wih_g, lstm2_wih_g_scale,
        lstm2_wih_o, lstm2_wih_o_scale,
        lstm2_whh_i, lstm2_whh_i_scale,
        lstm2_whh_f, lstm2_whh_f_scale,
        lstm2_whh_g, lstm2_whh_g_scale,
        lstm2_whh_o, lstm2_whh_o_scale,
        lstm2_bias_i, lstm2_bias_f, lstm2_bias_g, lstm2_bias_o,
        h1_new, h2_prev, c2_prev,
        h2_new, c2_new,
        gate_i, gate_f, gate_g, gate_o
    );
    state2.push(h2_new, c2_new);

    // Step 4: Output projection
    std::memcpy(logits, fc_out_bias, CHATBOT_VOCAB_SIZE * sizeof(float));
    for (int i = 0; i < CHATBOT_VOCAB_SIZE; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < CHATBOT_HIDDEN_SIZE; ++j) {
            if (fc_out_weight[i][j] != 0) {
                acc += (float)fc_out_weight[i][j] * h2_new[j];
            }
        }
        logits[i] += acc * fc_out_scale;
    }
}

static int run_generation(const char* raw_prompt, int max_len, float temperature) {
    if (max_len > MAX_GEN_LEN) max_len = MAX_GEN_LEN;
    reset_hidden_states();

    // Ensure prompt format is Q: ... \nA: 
    char prompt_buf[256];
    if (std::strncmp(raw_prompt, "Q:", 2) != 0) {
        snprintf(prompt_buf, sizeof(prompt_buf), "Q: %s\nA: ", raw_prompt);
    } else {
        snprintf(prompt_buf, sizeof(prompt_buf), "%s", raw_prompt);
    }
    const char* prompt = prompt_buf;

    // Warm up context with prompt characters
    int prompt_len = (int)std::strlen(prompt);
    for (int i = 0; i < prompt_len; ++i) {
        unsigned char c = (unsigned char)prompt[i];
        int idx = (c < 128) ? char2idx[(int)c] : UNK_IDX;
        forward_step(idx);
    }

    // Autoregressive generation
    int out_len = 0;
    static uint32_t rng_state = 1337;

    for (int step = 0; step < max_len; ++step) {
        // Sample next token directly from current logits
        int next_idx = 0;
        if (temperature <= 0.01f) {
            next_idx = argmax<CHATBOT_VOCAB_SIZE>(logits);
        } else {
            softmax_temperature<CHATBOT_VOCAB_SIZE>(logits, probs, temperature);
            rng_state = rng_state * 1664525u + 1013904223u;
            float r = (float)(rng_state >> 8) / (float)(1 << 24);
            float cumulative = 0.0f;
            next_idx = CHATBOT_VOCAB_SIZE - 1;
            for (int i = 0; i < CHATBOT_VOCAB_SIZE; ++i) {
                cumulative += probs[i];
                if (r < cumulative) {
                    next_idx = i;
                    break;
                }
            }
        }

        if (next_idx == EOS_IDX || next_idx == PAD_IDX) {
            break;
        }

        const char* ch_str = idx2char[next_idx];
        if (ch_str[0] == '\n' || (ch_str[0] == '<' && strstr(ch_str, "EOS"))) {
            break;
        }

        if (ch_str[0] != '<' && out_len < max_len - 1) {
            output_buf[out_len++] = ch_str[0];
            if (out_len >= 2 && output_buf[out_len - 2] == 'Q' && output_buf[out_len - 1] == ':') {
                out_len -= 2;
                break;
            }
        }

        // Feed newly generated character to compute next logits
        forward_step(next_idx);
    }

    while (out_len > 0 && (output_buf[out_len - 1] == ' ' || output_buf[out_len - 1] == '\r')) {
        out_len--;
    }
    output_buf[out_len] = '\0';
    return out_len;
}

// ---------------------------------------------------------------------------
// Wi-Fi & HTTP Handlers (ESP32 only)
// ---------------------------------------------------------------------------
#ifdef ESP32

static esp_err_t health_get_handler(httpd_req_t *req) {
    httpd_resp_set_type(req, "application/json");

    uint32_t free_heap = esp_get_free_heap_size();
    int64_t uptime_sec = esp_timer_get_time() / 1000000;

    cJSON *root = cJSON_CreateObject();
    cJSON_AddStringToObject(root, "status", "ok");
    cJSON_AddStringToObject(root, "device", "ESP32-Classic");
    cJSON_AddStringToObject(root, "model", "ColdInfer Tiny LSTM (INT8)");
    cJSON_AddNumberToObject(root, "free_heap_kb", (double)(free_heap / 1024));
    cJSON_AddNumberToObject(root, "uptime_sec", (double)uptime_sec);
    cJSON_AddNumberToObject(root, "sram_buffers_kb", (double)(sizeof(output_buf) + sizeof(h1)*4) / 1024.0);

    const char *resp_str = cJSON_PrintUnformatted(root);
    httpd_resp_send(req, resp_str, HTTPD_RESP_USE_STRLEN);

    cJSON_free((void*)resp_str);
    cJSON_Delete(root);
    return ESP_OK;
}

static esp_err_t chat_post_handler(httpd_req_t *req) {
    char content[512];
    int total_len = req->content_len;
    if (total_len >= sizeof(content)) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Payload too large");
        return ESP_FAIL;
    }

    int received = httpd_req_recv(req, content, total_len);
    if (received <= 0) {
        httpd_resp_send_err(req, HTTPD_500_INTERNAL_SERVER_ERROR, "Failed to read request body");
        return ESP_FAIL;
    }
    content[received] = '\0';

    cJSON *json = cJSON_Parse(content);
    if (!json) {
        httpd_resp_send_err(req, HTTPD_400_BAD_REQUEST, "Invalid JSON");
        return ESP_FAIL;
    }

    cJSON *prompt_item = cJSON_GetObjectItem(json, "prompt");
    const char *prompt = prompt_item && prompt_item->valuestring ? prompt_item->valuestring : "Hello";

    cJSON *temp_item = cJSON_GetObjectItem(json, "temperature");
    float temp = temp_item ? (float)temp_item->valuedouble : 0.8f;

    cJSON *len_item = cJSON_GetObjectItem(json, "max_len");
    int max_len = len_item ? len_item->valueint : 120;

    int64_t t_start = esp_timer_get_time();
    int n_tokens = run_generation(prompt, max_len, temp);
    int64_t t_end = esp_timer_get_time();
    int64_t elapsed_us = t_end - t_start;

    cJSON *resp_json = cJSON_CreateObject();
    cJSON_AddStringToObject(resp_json, "response", output_buf);
    cJSON_AddNumberToObject(resp_json, "tokens_generated", n_tokens);
    cJSON_AddNumberToObject(resp_json, "inference_us", (double)elapsed_us);
    cJSON_AddNumberToObject(resp_json, "free_heap_kb", (double)(esp_get_free_heap_size() / 1024));

    const char *resp_str = cJSON_PrintUnformatted(resp_json);
    httpd_resp_set_type(req, "application/json");
    httpd_resp_send(req, resp_str, HTTPD_RESP_USE_STRLEN);

    cJSON_free((void*)resp_str);
    cJSON_Delete(resp_json);
    cJSON_Delete(json);
    return ESP_OK;
}

static httpd_handle_t start_webserver(void) {
    httpd_handle_t server = NULL;
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    config.lru_purge_enable = true;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_uri_t health_uri = {
            .uri      = "/health",
            .method   = HTTP_GET,
            .handler  = health_get_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &health_uri);

        httpd_uri_t chat_uri = {
            .uri      = "/chat",
            .method   = HTTP_POST,
            .handler  = chat_post_handler,
            .user_ctx = NULL
        };
        httpd_register_uri_handler(server, &chat_uri);

        ESP_LOGI(TAG, "HTTP Server started on port %d", config.server_port);
    }
    return server;
}

static void wifi_event_handler(void* arg, esp_event_base_t event_base,
                               int32_t event_id, void* event_data) {
    if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_START) {
        esp_wifi_connect();
    } else if (event_base == WIFI_EVENT && event_id == WIFI_EVENT_STA_DISCONNECTED) {
        if (s_retry_num < 10) {
            esp_wifi_connect();
            s_retry_num++;
            ESP_LOGI(TAG, "Retrying Wi-Fi connection (%d/10)...", s_retry_num);
        } else {
            xEventGroupSetBits(s_wifi_event_group, WIFI_FAIL_BIT);
        }
    } else if (event_base == IP_EVENT && event_id == IP_EVENT_STA_GOT_IP) {
        ip_event_got_ip_t* event = (ip_event_got_ip_t*) event_data;
        ESP_LOGI(TAG, "Connected! IP Address: " IPSTR, IP2STR(&event->ip_info.ip));
        s_retry_num = 0;
        xEventGroupSetBits(s_wifi_event_group, WIFI_CONNECTED_BIT);
    }
}

static void wifi_init_sta(void) {
    s_wifi_event_group = xEventGroupCreate();
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    esp_event_handler_instance_t instance_any_id;
    esp_event_handler_instance_t instance_got_ip;
    ESP_ERROR_CHECK(esp_event_handler_instance_register(WIFI_EVENT,
                                                        ESP_EVENT_ANY_ID,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_any_id));
    ESP_ERROR_CHECK(esp_event_handler_instance_register(IP_EVENT,
                                                        IP_EVENT_STA_GOT_IP,
                                                        &wifi_event_handler,
                                                        NULL,
                                                        &instance_got_ip));

    wifi_config_t wifi_config = {};
    std::strncpy((char*)wifi_config.sta.ssid, WIFI_SSID, sizeof(wifi_config.sta.ssid));
    std::strncpy((char*)wifi_config.sta.password, WIFI_PASS, sizeof(wifi_config.sta.password));
    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    ESP_LOGI(TAG, "Wi-Fi initialization finished. Waiting for AP: %s", WIFI_SSID);
}

extern "C" void app_main() {
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "ColdInfer ESP32 Chatbot booting...");
    ESP_LOGI(TAG, "Model: INT8 Sparse LSTM (2 layers, hidden=64)");
    ESP_LOGI(TAG, "Static arena SRAM: %d bytes", (int)sizeof(output_buf) + (int)sizeof(h1)*4);

    wifi_init_sta();

    // Wait until connected to Wi-Fi
    EventBits_t bits = xEventGroupWaitBits(s_wifi_event_group,
            WIFI_CONNECTED_BIT | WIFI_FAIL_BIT,
            pdFALSE,
            pdFALSE,
            portMAX_DELAY);

    if (bits & WIFI_CONNECTED_BIT) {
        ESP_LOGI(TAG, "Wi-Fi connected. Starting HTTP REST server...");
        start_webserver();
    } else {
        ESP_LOGE(TAG, "Failed to connect to Wi-Fi AP: %s", WIFI_SSID);
    }
}

#else
// Desktop smoke-test main
int main() {
    std::cout << "Testing generation logic on desktop..." << std::endl;
    int n = run_generation("Hello", 100, 0.8f);
    std::cout << "Generated: " << output_buf << std::endl;
    std::cout << "Tokens: " << n << std::endl;
    return 0;
}
#endif
