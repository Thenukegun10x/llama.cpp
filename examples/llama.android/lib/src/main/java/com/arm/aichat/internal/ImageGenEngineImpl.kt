package com.arm.aichat.internal

import android.util.Log
import com.arm.aichat.ImageGenEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.withContext
import java.io.File

class ImageGenEngineImpl(private val outputDir: File) : ImageGenEngine {
    private val _state = MutableStateFlow(ImageGenEngine.State())
    override val state: StateFlow<ImageGenEngine.State> = _state

    init {
        try {
            System.loadLibrary("sd-chat")
        } catch (e: UnsatisfiedLinkError) {
            Log.w(TAG, "Image generation library not available: ${e.message}")
            _state.value = ImageGenEngine.State(
                mode = ImageGenEngine.Mode.ERROR,
                error = "libsd-chat.so not found"
            )
        }
    }

    override suspend fun init() {
        withContext(Dispatchers.IO) {
            try {
                _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.LOADING)
                callNativeInit()
                _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.UNLOADED)
            } catch (e: Exception) {
                Log.e(TAG, "Image gen init failed", e)
                _state.value = ImageGenEngine.State(
                    mode = ImageGenEngine.Mode.ERROR,
                    error = e.message
                )
            }
        }
    }

    override fun systemInfo(): String = try {
        callNativeSystemInfo()
    } catch (e: Exception) {
        "unavailable: ${e.message}"
    }

    override suspend fun loadModel(pathToModel: String) {
        withContext(Dispatchers.IO) {
            try {
                _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.LOADING)
                val ok = callNativeLoadModel(pathToModel)
                if (ok) {
                    _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.READY)
                } else {
                    _state.value = ImageGenEngine.State(
                        mode = ImageGenEngine.Mode.ERROR,
                        error = "Failed to load model"
                    )
                }
            } catch (e: Exception) {
                Log.e(TAG, "Model load failed", e)
                _state.value = ImageGenEngine.State(
                    mode = ImageGenEngine.Mode.ERROR,
                    error = e.message
                )
            }
        }
    }

    override suspend fun generateImage(
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfgScale: Float,
        seed: Long
    ): String? = withContext(Dispatchers.IO) {
        try {
            _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.GENERATING)
            outputDir.mkdirs()
            val outputFile = File(outputDir, "gen_${System.currentTimeMillis()}.png")
            val resolvedSeed = if (seed < 0) (Math.random() * Long.MAX_VALUE).toLong() else seed

            val result = callNativeGenerateImage(
                prompt, negativePrompt,
                width, height,
                steps, cfgScale,
                resolvedSeed,
                outputFile.absolutePath
            )

            if (result != null && outputFile.exists() && outputFile.length() > 0) {
                _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.READY)
                outputFile.absolutePath
            } else {
                _state.value = ImageGenEngine.State(
                    mode = ImageGenEngine.Mode.ERROR,
                    error = "Image generation returned empty result"
                )
                null
            }
        } catch (e: Exception) {
            Log.e(TAG, "Image generation failed", e)
            _state.value = ImageGenEngine.State(
                mode = ImageGenEngine.Mode.ERROR,
                error = e.message
            )
            null
        }
    }

    override fun cancelGeneration() {
        try {
            callNativeCancel()
        } catch (e: Exception) {
            Log.w(TAG, "Cancel failed", e)
        }
    }

    override fun unload() {
        try {
            callNativeUnload()
        } catch (e: Exception) {
            Log.w(TAG, "Unload failed", e)
        }
        _state.value = ImageGenEngine.State(mode = ImageGenEngine.Mode.UNLOADED)
    }

    override fun destroy() {
        unload()
    }

    private external fun callNativeInit()
    private external fun callNativeSystemInfo(): String
    private external fun callNativeLoadModel(modelPath: String): Boolean
    private external fun callNativeGenerateImage(
        prompt: String,
        negativePrompt: String,
        width: Int,
        height: Int,
        steps: Int,
        cfgScale: Float,
        seed: Long,
        outputPath: String
    ): String?
    private external fun callNativeCancel()
    private external fun callNativeUnload()

    companion object {
        private val TAG = ImageGenEngineImpl::class.java.simpleName
    }
}
