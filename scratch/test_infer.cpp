#include <iostream>
#include <cstring>
#include <cmath>
#include "chatbot_weights.h"
#include "chatbot_vocab.h"
#include "lstm_kernels.h"
#include "rolling_kv.h"

static float embed_out[CHATBOT_EMBED_DIM];
static float h1[CHATBOT_HIDDEN_SIZE];
static float c1[CHATBOT_HIDDEN_SIZE];
static float h2[CHATBOT_HIDDEN_SIZE];
static float c2[CHATBOT_HIDDEN_SIZE];
static float gate_i[CHATBOT_HIDDEN_SIZE];
static float gate_f[CHATBOT_HIDDEN_SIZE];
static float gate_g[CHATBOT_HIDDEN_SIZE];
static float gate_o[CHATBOT_HIDDEN_SIZE];
static float lstm_out[CHATBOT_HIDDEN_SIZE];
static float logits[CHATBOT_VOCAB_SIZE];

void forward_step(int char_idx) {
    embedding_lookup<CHATBOT_VOCAB_SIZE, CHATBOT_EMBED_DIM>(emb_weight, emb_scale, char_idx, embed_out);

    float h1_new[CHATBOT_HIDDEN_SIZE];
    float c1_new[CHATBOT_HIDDEN_SIZE];
    LSTMCell<CHATBOT_EMBED_DIM, CHATBOT_HIDDEN_SIZE>::step(
        lstm1_wih_i, lstm1_wih_i_scale, lstm1_wih_f, lstm1_wih_f_scale,
        lstm1_wih_g, lstm1_wih_g_scale, lstm1_wih_o, lstm1_wih_o_scale,
        lstm1_whh_i, lstm1_whh_i_scale, lstm1_whh_f, lstm1_whh_f_scale,
        lstm1_whh_g, lstm1_whh_g_scale, lstm1_whh_o, lstm1_whh_o_scale,
        lstm1_bias_i, lstm1_bias_f, lstm1_bias_g, lstm1_bias_o,
        embed_out, h1, c1, h1_new, c1_new,
        gate_i, gate_f, gate_g, gate_o
    );
    memcpy(h1, h1_new, sizeof(h1));
    memcpy(c1, c1_new, sizeof(c1));

    float h2_new[CHATBOT_HIDDEN_SIZE];
    float c2_new[CHATBOT_HIDDEN_SIZE];
    LSTMCell<CHATBOT_HIDDEN_SIZE, CHATBOT_HIDDEN_SIZE>::step(
        lstm2_wih_i, lstm2_wih_i_scale, lstm2_wih_f, lstm2_wih_f_scale,
        lstm2_wih_g, lstm2_wih_g_scale, lstm2_wih_o, lstm2_wih_o_scale,
        lstm2_whh_i, lstm2_whh_i_scale, lstm2_whh_f, lstm2_whh_f_scale,
        lstm2_whh_g, lstm2_whh_g_scale, lstm2_whh_o, lstm2_whh_o_scale,
        lstm2_bias_i, lstm2_bias_f, lstm2_bias_g, lstm2_bias_o,
        h1, h2, c2, h2_new, c2_new,
        gate_i, gate_f, gate_g, gate_o
    );
    memcpy(h2, h2_new, sizeof(h2));
    memcpy(c2, c2_new, sizeof(c2));

    memcpy(logits, fc_out_bias, sizeof(logits));
    for (int i = 0; i < CHATBOT_VOCAB_SIZE; ++i) {
        float acc = 0.0f;
        for (int j = 0; j < CHATBOT_HIDDEN_SIZE; ++j) {
            if (fc_out_weight[i][j] != 0) acc += (float)fc_out_weight[i][j] * h2[j];
        }
        logits[i] += acc * fc_out_scale;
    }
}

int main() {
    memset(h1, 0, sizeof(h1));
    memset(c1, 0, sizeof(c1));
    memset(h2, 0, sizeof(h2));
    memset(c2, 0, sizeof(c2));

    const char* prompt = "Q: hlo\nA: ";
    int len = strlen(prompt);
    for (int i = 0; i < len; ++i) {
        unsigned char c = (unsigned char)prompt[i];
        int idx = (c < 128) ? char2idx[(int)c] : UNK_IDX;
        forward_step(idx);
        std::cout << "Fed '" << c << "' (idx " << idx << "), top pred: '" 
                  << idx2char[argmax<CHATBOT_VOCAB_SIZE>(logits)] << "'\n";
    }

    std::cout << "Generation: ";
    for (int step = 0; step < 80; ++step) {
        int next_idx = argmax<CHATBOT_VOCAB_SIZE>(logits);
        const char* ch = idx2char[next_idx];
        if (ch[0] == '\n') break;
        std::cout << ch;
        forward_step(next_idx);
    }
    std::cout << std::endl;
    return 0;
}
