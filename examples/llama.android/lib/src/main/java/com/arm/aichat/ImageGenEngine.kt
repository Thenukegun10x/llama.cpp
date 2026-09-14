package com.arm.aichat

import kotlinx.coroutines.flow.StateFlow

interface ImageGenEngine {
    val state: StateFlow<State>

    suspend fun init()
    fun systemInfo(): String
    suspend fun loadModel(pathToModel: String)
    suspend fun generateImage(
        prompt: String,
        negativePrompt: String = "",
        width: Int = DEFAULT_WIDTH,
        height: Int = DEFAULT_HEIGHT,
        steps: Int = DEFAULT_STEPS,
        cfgScale: Float = DEFAULT_CFG_SCALE,
        seed: Long = DEFAULT_SEED
    ): String?

    fun cancelGeneration()
    fun unload()
    fun destroy()

    enum class Mode {
        UNLOADED,
        LOADING,
        READY,
        GENERATING,
        ERROR
    }

    data class State(
        val mode: Mode = Mode.UNLOADED,
        val error: String? = null
    )

    companion object {
        const val DEFAULT_WIDTH = 512
        const val DEFAULT_HEIGHT = 512
        const val DEFAULT_STEPS = 20
        const val DEFAULT_CFG_SCALE = 7.0f
        const val DEFAULT_SEED = -1L
    }
}
