package com.example.llama

import android.content.Context

class ToolSettings(context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    var allToolsEnabled: Boolean
        get() = preferences.getBoolean(KEY_ALL_TOOLS_ENABLED, true)
        set(value) = preferences.edit().putBoolean(KEY_ALL_TOOLS_ENABLED, value).apply()

    var sandboxToolsEnabled: Boolean
        get() = preferences.getBoolean(KEY_SANDBOX_TOOLS_ENABLED, true)
        set(value) = preferences.edit().putBoolean(KEY_SANDBOX_TOOLS_ENABLED, value).apply()

    var webToolsEnabled: Boolean
        get() = preferences.getBoolean(KEY_WEB_TOOLS_ENABLED, true)
        set(value) = preferences.edit().putBoolean(KEY_WEB_TOOLS_ENABLED, value).apply()

    companion object {
        private const val PREFERENCES_NAME = "llama_tool_settings"
        private const val KEY_ALL_TOOLS_ENABLED = "all_tools_enabled"
        private const val KEY_SANDBOX_TOOLS_ENABLED = "sandbox_tools_enabled"
        private const val KEY_WEB_TOOLS_ENABLED = "web_tools_enabled"
    }
}
