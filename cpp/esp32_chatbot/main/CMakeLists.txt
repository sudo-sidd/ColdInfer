idf_component_register(
    SRCS "chatbot_main.cpp"
    INCLUDE_DIRS "." ".." "../../" "../../../weights"
    REQUIRES esp_wifi esp_netif esp_http_server nvs_flash esp_event esp_timer
)

target_compile_definitions(${COMPONENT_LIB} PRIVATE ESP32)
