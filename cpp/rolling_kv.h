// rolling_kv.h — Fixed-size circular buffer for LSTM hidden + cell states
// Part of ColdInfer — bare-metal LSTM inference engine
//
// The rolling context window caps SRAM usage at O(N * hidden_size) regardless
// of total sequence length. N is set at compile time so the compiler can
// size the static arrays exactly — zero dynamic allocation.

#pragma once
#include <cstring>

template <int HIDDEN_SIZE, int WINDOW_SIZE = 8>
struct RollingStateBuffer {
    // We store the last WINDOW_SIZE hidden states and cell states
    // for a single LSTM layer. Two layers → two instances of this struct.
    float h[WINDOW_SIZE][HIDDEN_SIZE];   // hidden states
    float c[WINDOW_SIZE][HIDDEN_SIZE];   // cell states
    int   head = 0;                      // index of the most recent state
    int   count = 0;                     // number of valid entries so far

    // Zero-initialise all buffers
    void reset() {
        memset(h, 0, sizeof(h));
        memset(c, 0, sizeof(c));
        head  = 0;
        count = 0;
    }

    // Write the current hidden + cell state into the circular buffer
    void push(const float* new_h, const float* new_c) {
        memcpy(h[head], new_h, HIDDEN_SIZE * sizeof(float));
        memcpy(c[head], new_c, HIDDEN_SIZE * sizeof(float));
        head = (head + 1) % WINDOW_SIZE;
        if (count < WINDOW_SIZE) count++;
    }

    // Read the most recently stored hidden state (h_{t-1})
    const float* last_h() const {
        int idx = (head - 1 + WINDOW_SIZE) % WINDOW_SIZE;
        return h[idx];
    }

    // Read the most recently stored cell state (c_{t-1})
    const float* last_c() const {
        int idx = (head - 1 + WINDOW_SIZE) % WINDOW_SIZE;
        return c[idx];
    }

    // Read the hidden state N steps back (0 = most recent)
    const float* h_at(int steps_back) const {
        int idx = (head - 1 - steps_back + WINDOW_SIZE * 2) % WINDOW_SIZE;
        return h[idx];
    }
};
