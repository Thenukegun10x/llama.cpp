package com.example.llama

import android.content.Context

class InferenceSettings(context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    var contextSize: Int
        get() = preferences.getInt(KEY_CONTEXT_SIZE, DEFAULT_CONTEXT_SIZE)
        set(value) = preferences.edit().putInt(KEY_CONTEXT_SIZE, value).apply()

    var maxTokens: Int
        get() = preferences.getInt(KEY_MAX_TOKENS, DEFAULT_MAX_TOKENS)
        set(value) = preferences.edit().putInt(KEY_MAX_TOKENS, value).apply()

    var temperature: Float
        get() = preferences.getFloat(KEY_TEMPERATURE, DEFAULT_TEMPERATURE)
        set(value) = preferences.edit().putFloat(KEY_TEMPERATURE, value).apply()

    var topK: Int
        get() = preferences.getInt(KEY_TOP_K, DEFAULT_TOP_K)
        set(value) = preferences.edit().putInt(KEY_TOP_K, value).apply()

    var topP: Float
        get() = preferences.getFloat(KEY_TOP_P, DEFAULT_TOP_P)
        set(value) = preferences.edit().putFloat(KEY_TOP_P, value).apply()

    var repeatPenalty: Float
        get() = preferences.getFloat(KEY_REPEAT_PENALTY, DEFAULT_REPEAT_PENALTY)
        set(value) = preferences.edit().putFloat(KEY_REPEAT_PENALTY, value).apply()

    var kvCacheType: Int
        get() = preferences.getInt(KEY_KV_CACHE_TYPE, KV_F16)
        set(value) = preferences.edit().putInt(KEY_KV_CACHE_TYPE, value).apply()

    var threads: Int
        get() = preferences.getInt(KEY_THREADS, DEFAULT_THREADS).coerceAtLeast(1)
        set(value) = preferences.edit().putInt(KEY_THREADS, value).apply()

    companion object {
        private const val PREFERENCES_NAME = "llama_inference_settings"
        private const val KEY_CONTEXT_SIZE = "context_size"
        private const val KEY_MAX_TOKENS = "max_tokens"
        private const val KEY_TEMPERATURE = "temperature"
        private const val KEY_TOP_K = "top_k"
        private const val KEY_TOP_P = "top_p"
        private const val KEY_REPEAT_PENALTY = "repeat_penalty"
        private const val KEY_KV_CACHE_TYPE = "kv_cache_type"
        private const val KEY_THREADS = "threads"

        const val DEFAULT_CONTEXT_SIZE = 4096
        const val DEFAULT_MAX_TOKENS = 1024
        const val DEFAULT_TEMPERATURE = 0.3f
        const val DEFAULT_TOP_K = 40
        const val DEFAULT_TOP_P = 0.95f
        const val DEFAULT_REPEAT_PENALTY = 1.0f
        val MAX_THREADS = Runtime.getRuntime().availableProcessors().coerceAtLeast(2)
        val DEFAULT_THREADS = minOf(4, MAX_THREADS)

        const val KV_F16 = 0
        const val KV_Q8_0 = 1
        const val KV_Q4_0 = 2

        val CONTEXT_OPTIONS = intArrayOf(2048, 4096, 8192, 16384)
        val KV_LABELS = arrayOf("F16 (best quality)", "Q8_0 (balanced)", "Q4_0 (low memory)")
    }
}
