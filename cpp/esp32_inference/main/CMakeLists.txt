idf_component_register(SRCS "infer.cpp"
                       INCLUDE_DIRS "."
                       REQUIRES esp_wifi esp_netif esp_http_server nvs_flash esp_event esp_timer)

# Pass the ESP32 definition to your compiler
target_compile_definitions(${COMPONENT_LIB} PRIVATE ESP32)
