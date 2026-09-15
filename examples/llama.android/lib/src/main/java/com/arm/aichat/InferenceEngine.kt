package com.arm.aichat

import com.arm.aichat.InferenceEngine.State
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.StateFlow

/**
 * Interface defining the core LLM inference operations.
 */
interface InferenceEngine {
    /**
     * Current state of the inference engine
     */
    val state: StateFlow<State>

    /**
     * Load a model from the given path.
     *
     * @throws UnsupportedArchitectureException if model architecture not supported
     */
    suspend fun loadModel(pathToModel: String)

    /**
     * Set context size, KV cache type and thread count. Applies on the next [loadModel] call.
     *
     * @param nCtx context size in tokens (e.g. 2048, 4096, 8192)
     * @param kvCacheType 0 = F16, 1 = Q8_0, 2 = Q4_0
     * @param threads worker threads, or 0/negative for auto
     */
    suspend fun configure(nCtx: Int, kvCacheType: Int, threads: Int = 0)

    /**
     * Update sampling params. Applies live when the engine is idle.
     */
    suspend fun updateSampling(temp: Float, topK: Int, topP: Float, penaltyRepeat: Float)

    /**
     * Tokens currently held in context and the context size.
     */
    suspend fun contextUsage(): Pair<Int, Int>

    /**
     * Sends a system prompt to the loaded model
     */
    suspend fun setSystemPrompt(systemPrompt: String)

    /**
     * Set tools JSON schema (OpenAI format). Enables native template tool formatting and parsing.
     */
    suspend fun setTools(toolsJson: String)

    /**
     * Returns true if the active model chat template natively supports tool use.
     */
    suspend fun hasNativeToolSupport(): Boolean

    /**
     * Parses the assistant response using the native template PEG parser.
     */
    suspend fun parseResponse(rawText: String, isPartial: Boolean = false): ParsedResponse

    /**
     * Sends a user prompt to the loaded model and returns a Flow of generated tokens.
     */
    fun sendUserPrompt(message: String, predictLength: Int = DEFAULT_PREDICT_LENGTH): Flow<String>

    /**
     * Sends a tool execution result to the loaded model and returns a Flow of generated tokens.
     */
    fun sendToolResponse(
        toolName: String,
        callId: String = "",
        content: String,
        predictLength: Int = DEFAULT_PREDICT_LENGTH
    ): Flow<String>

    /**
     * Requests in-progress generation to stop. Safe to call from any thread.
     */
    fun stopGeneration()

    /**
     * Runs a benchmark with the specified parameters.
     */
    suspend fun bench(pp: Int, tg: Int, pl: Int, nr: Int = 1): String

    /**
     * Unloads the currently loaded model. Suspends instead of blocking
     * the caller, so it is safe to call from the main thread's scope.
     */
    suspend fun cleanUp()

    /**
     * Cleans up resources when the engine is no longer needed.
     */
    fun destroy()

    /**
     * States of the inference engine
     */
    sealed class State {
        object Uninitialized : State()
        object Initializing : State()
        object Initialized : State()

        object LoadingModel : State()
        object UnloadingModel : State()
        object ModelReady : State()

        object Benchmarking : State()
        object ProcessingSystemPrompt : State()
        object ProcessingUserPrompt : State()

        object Generating : State()

        data class Error(val exception: Exception) : State()
    }

    companion object {
        const val DEFAULT_PREDICT_LENGTH = 1024
    }
}

val State.isUninterruptible
    get() = this is State.Initializing ||
        this is State.LoadingModel ||
        this is State.UnloadingModel ||
        this is State.Benchmarking ||
        this is State.ProcessingSystemPrompt ||
        this is State.ProcessingUserPrompt

val State.isModelLoaded: Boolean
    get() = this is State.ModelReady ||
        this is State.Benchmarking ||
        this is State.ProcessingSystemPrompt ||
        this is State.ProcessingUserPrompt ||
        this is State.Generating

class UnsupportedArchitectureException : Exception()

data class ToolCallInfo(
    val name: String,
    val arguments: String,
    val id: String = ""
)

data class ParsedResponse(
    val content: String,
    val thinking: String,
    val toolCalls: List<ToolCallInfo>,
    val hasParser: Boolean
)
