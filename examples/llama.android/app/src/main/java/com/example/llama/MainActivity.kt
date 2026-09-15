package com.example.llama

import android.Manifest
import android.app.ActivityManager
import android.content.ComponentCallbacks2
import android.content.Context
import android.content.pm.PackageManager
import android.content.res.ColorStateList
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.MotionEvent
import android.view.View
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.ProgressBar
import android.widget.Spinner
import android.widget.TextView
import android.widget.Toast
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.lifecycle.lifecycleScope
import androidx.recyclerview.widget.LinearLayoutManager
import androidx.recyclerview.widget.RecyclerView
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.ParsedResponse
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.isModelLoaded
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import android.view.ViewGroup
import org.json.JSONObject

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

import java.io.File
import java.io.FileOutputStream
import java.io.InputStream
import java.util.Locale
import java.util.UUID

private data class ResponseSplit(val thinking: String, val tooling: String, val answer: String)

private sealed class ToolCall {
    data class Sandbox(val call: SandboxToolCall) : ToolCall()
    data class WebSearch(val call: WebSearchCall) : ToolCall()
}

class MainActivity : AppCompatActivity() {
    private lateinit var toolbar: MaterialToolbar
    private lateinit var modelStatusTv: TextView
    private lateinit var messagesRv: RecyclerView
    private lateinit var emptyStateTv: TextView
    private lateinit var userInputEt: TextInputEditText
    private lateinit var userActionBtn: MaterialButton
    private lateinit var micButton: MaterialButton
    private lateinit var changeModelBtn: MaterialButton
    private lateinit var ctxTracker: View
    private lateinit var ctxUsageTv: TextView
    private lateinit var ctxBar: ProgressBar
    private lateinit var ramTracker: View
    private lateinit var ramUsageTv: TextView
    private lateinit var ramBar: ProgressBar
    private lateinit var chatStore: ConversationStore
    private lateinit var sandboxTools: SandboxFileTools
    private lateinit var webSearchTools: WebSearchTool
    private lateinit var modelRepository: HuggingFaceModelRepository
    private lateinit var speechRecognizer: OnDeviceSpeechRecognizer
    private lateinit var toolSettings: ToolSettings
    private lateinit var inferenceSettings: InferenceSettings

    private var engine: InferenceEngine? = null
    private var engineReady = false
    private var isModelReady = false
    private var isGenerating = false
    private var selectedModelName: String? = null
    private var generationJob: Job? = null
    private var modelDownloadJob: Job? = null
    private var repositoryIndexJob: Job? = null
    private var speechJob: Job? = null
    private var nativeToolsSupported = false

    private val requestMicrophonePermission = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted ->
        if (granted) startVoiceInput()
        else showToast(getString(R.string.microphone_permission_required))
    }

    private val conversations = mutableListOf<Conversation>()
    private var activeConversationId: String? = null
    private val messages = mutableListOf<Message>()
    private val messageAdapter = MessageAdapter(messages)

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContentView(R.layout.activity_main)

        chatStore = ConversationStore(applicationContext)
        sandboxTools = SandboxFileTools(applicationContext)
        webSearchTools = WebSearchTool()
        modelRepository = HuggingFaceModelRepository(applicationContext)
        speechRecognizer = OnDeviceSpeechRecognizer(applicationContext)
        toolSettings = ToolSettings(applicationContext)
        inferenceSettings = InferenceSettings(applicationContext)
        toolbar = findViewById(R.id.toolbar)
        modelStatusTv = findViewById(R.id.model_status)
        messagesRv = findViewById(R.id.messages)
        emptyStateTv = findViewById(R.id.empty_state)
        userInputEt = findViewById(R.id.user_input)
        userActionBtn = findViewById(R.id.user_action)
        micButton = findViewById(R.id.mic_button)
        changeModelBtn = findViewById(R.id.change_model)
        ctxTracker = findViewById(R.id.ctx_tracker)
        ctxUsageTv = findViewById(R.id.ctx_usage)
        ctxBar = findViewById(R.id.ctx_bar)
        ramTracker = findViewById(R.id.ram_tracker)
        ramUsageTv = findViewById(R.id.ram_usage)
        ramBar = findViewById(R.id.ram_bar)
        startRamTracker()

        messagesRv.layoutManager = LinearLayoutManager(this).apply { stackFromEnd = true }
        messagesRv.adapter = messageAdapter
        setupKeyboardHandling()

        setupToolbar()
        restoreConversations()
        refreshUi()
        initializeEngine()

        userActionBtn.setOnClickListener {
            if (isGenerating) {
                stopGeneration()
            } else if (isModelReady) {
                handleUserInput()
            } else {
                showModelManager()
            }
        }
        changeModelBtn.setOnClickListener { showModelManager() }
        micButton.setOnClickListener { toggleVoiceInput() }
    }

    private fun setupToolbar() {
        toolbar.title = getString(R.string.app_name)
        toolbar.subtitle = null
        toolbar.setNavigationIcon(R.drawable.outline_history_24)
        toolbar.setNavigationContentDescription(R.string.conversation_history)
        toolbar.inflateMenu(R.menu.menu_chat)
        val toolbarIconColor = getColor(R.color.top_bar_foreground)
        toolbar.navigationIcon?.setTint(toolbarIconColor)
        toolbar.overflowIcon?.setTint(toolbarIconColor)
        for (index in 0 until toolbar.menu.size()) {
            toolbar.menu.getItem(index).icon?.setTint(toolbarIconColor)
        }
        toolbar.setNavigationOnClickListener { showConversationHistory() }
        toolbar.setOnMenuItemClickListener { item ->
            when (item.itemId) {
                R.id.action_new_chat -> {
                    createConversation()
                    true
                }
                R.id.action_history -> {
                    showConversationHistory()
                    true
                }
                R.id.action_select_model -> {
                    showModelManager()
                    true
                }
                R.id.action_sandbox_files -> {
                    showSandboxFiles()
                    true
                }
                R.id.action_tool_settings -> {
                    showToolSettings()
                    true
                }
                R.id.action_inference_settings -> {
                    showInferenceSettings()
                    true
                }
                R.id.action_unload_model -> {
                    unloadAllModels()
                    true
                }
                else -> false
            }
        }
    }

    private fun initializeEngine() {
        lifecycleScope.launch {
            try {
                val initializedEngine = withContext(Dispatchers.Default) {
                    AiChat.getInferenceEngine(applicationContext).also { inferenceEngine ->
                        val state = inferenceEngine.state.filter {
                            it is InferenceEngine.State.Initialized || it is InferenceEngine.State.Error
                        }.first()
                        if (state is InferenceEngine.State.Error) {
                            throw IllegalStateException(
                                state.exception.message ?: "Unable to start the inference engine",
                                state.exception
                            )
                        }
                    }
                }
                engine = initializedEngine
                engineReady = true
                refreshUi()
            } catch (exception: Exception) {
                Log.e(TAG, "Unable to initialize inference engine", exception)
                showToast(getString(R.string.engine_start_failed))
                modelStatusTv.text = getString(R.string.engine_unavailable)
            }
        }
    }

    private fun buildSystemPrompt(): String {
        val parts = mutableListOf<String>()
        parts.add("You are a helpful assistant running on the user's phone. Most work happens on-device; web tools reach the live internet when enabled. Answer normally unless a tool is needed.")

        if (!toolSettings.allToolsEnabled) {
            parts.add("")
            parts.add("Tools are currently disabled. Answer directly from your own knowledge.")
            return parts.joinToString("\n")
        }

        // When the loaded model has a native tool-aware chat template, Jinja injects
        // the tools section and calling instructions in the model's exact native format.
        if (nativeToolsSupported) {
            return parts.joinToString("\n")
        }

        val toolDescs = mutableListOf<String>()
        if (toolSettings.sandboxToolsEnabled) {
            toolDescs.add("FILE TOOLS (paths are relative to the private sandbox):")
            toolDescs.add("""list_files: {"name":"list_files"}""")
            toolDescs.add("""read_file: {"name":"read_file","path":"notes.txt"}""")
            toolDescs.add("""write_file: {"name":"write_file","path":"notes.txt","content":"text"}""")
            toolDescs.add("""append_file: {"name":"append_file","path":"notes.txt","content":"more text"}""")
            toolDescs.add("""edit_file: {"name":"edit_file","path":"notes.txt","old_text":"unique old text","new_text":"new text"}""")
            toolDescs.add("Use one tool at a time. Read before editing when unsure. For edit_file, old_text can be a short distinctive excerpt; matching tolerates case and whitespace differences but must be unique.")
        }
        if (toolSettings.webToolsEnabled) {
            if (toolDescs.isNotEmpty()) toolDescs.add("")
            toolDescs.add("WEB TOOLS (live internet search and page reading):")
            toolDescs.add("""web_search: {"name":"web_search","query":"search terms","limit":5,"offset":0}""")
            toolDescs.add("  - query: search terms (required)")
            toolDescs.add("  - limit: results to return, 1-10 (optional, default: 5)")
            toolDescs.add("  - offset: skip this many results to see more (optional, default: 0)")
            toolDescs.add("""web_fetch: {"name":"web_fetch","url":"https://example.com/page","max_chars":6000}""")
            toolDescs.add("  - url: page to read as filtered text (required)")
            toolDescs.add("  - max_chars: max text chars, 500-20000 (optional, default: 6000)")
            toolDescs.add("Search first, then fetch the most promising result. Use offset for more results.")
        }
        if (toolDescs.isNotEmpty()) {
            parts.add("")
            parts.addAll(toolDescs)
        }

        parts.add("")
        parts.add("To use a tool, output only one call in this exact wrapper:")
        parts.add("""<tool_call>{"name":"read_file","path":"notes.txt"}</tool_call>""")
        parts.add("Rules:")
        parts.add("- One tool call per message. Wait for <tool_result> before continuing.")
        parts.add("- Never invent a tool result. If no tool fits, just answer directly.")
        parts.add("- Answer from the tool result; do not paste raw JSON back to the user.")
        return parts.joinToString("\n")
    }

    private fun showToolSettings() {
        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_tool_settings, null)
        val allToggle = content.findViewById<MaterialCheckBox>(R.id.toggle_all_tools)
        val sandboxToggle = content.findViewById<MaterialCheckBox>(R.id.toggle_sandbox)
        val webToggle = content.findViewById<MaterialCheckBox>(R.id.toggle_web)

        allToggle.isChecked = toolSettings.allToolsEnabled
        sandboxToggle.isChecked = toolSettings.sandboxToolsEnabled
        webToggle.isChecked = toolSettings.webToolsEnabled

        fun applyChildEnabled() {
            val enabled = allToggle.isChecked
            sandboxToggle.isEnabled = enabled
            webToggle.isEnabled = enabled
        }
        applyChildEnabled()

        allToggle.setOnCheckedChangeListener { _, _ -> applyChildEnabled() }

        dialog.setOnDismissListener {
            toolSettings.allToolsEnabled = allToggle.isChecked
            toolSettings.sandboxToolsEnabled = sandboxToggle.isChecked
            toolSettings.webToolsEnabled = webToggle.isChecked
            lifecycleScope.launch {
                val inferenceEngine = engine ?: return@launch
                withContext(Dispatchers.IO) {
                    inferenceEngine.setTools(ToolDefinitions.buildToolsJson(toolSettings))
                    nativeToolsSupported = inferenceEngine.hasNativeToolSupport()
                    inferenceEngine.setSystemPrompt(buildSystemPrompt())
                }
            }
        }

        dialog.setContentView(content)
        dialog.show()
    }

    private fun showInferenceSettings() {
        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_inference_settings, null)
        val ctxSpinner = content.findViewById<Spinner>(R.id.setting_context_size)
        val kvSpinner = content.findViewById<Spinner>(R.id.setting_kv_cache)
        val threadsSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.setting_threads)
        val tempSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.setting_temperature)
        val topPSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.setting_top_p)
        val topKSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.setting_top_k)
        val repeatSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.setting_repeat_penalty)
        val maxTokensSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.setting_max_tokens)
        val tempLabel = content.findViewById<TextView>(R.id.setting_temp_label)
        val topPLabel = content.findViewById<TextView>(R.id.setting_top_p_label)
        val topKLabel = content.findViewById<TextView>(R.id.setting_top_k_label)
        val repeatLabel = content.findViewById<TextView>(R.id.setting_repeat_label)
        val maxTokensLabel = content.findViewById<TextView>(R.id.setting_max_tokens_label)
        val threadsLabel = content.findViewById<TextView>(R.id.setting_threads_label)

        val prevCtx = inferenceSettings.contextSize
        val prevKv = inferenceSettings.kvCacheType
        val prevThreads = inferenceSettings.threads

        ctxSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            InferenceSettings.CONTEXT_OPTIONS.map { it.toString() }
        )
        ctxSpinner.setSelection(
            InferenceSettings.CONTEXT_OPTIONS.indexOf(prevCtx).takeIf { it >= 0 } ?: 1
        )
        kvSpinner.adapter = ArrayAdapter(
            this, android.R.layout.simple_spinner_dropdown_item,
            InferenceSettings.KV_LABELS.toList()
        )
        kvSpinner.setSelection(prevKv.coerceIn(0, InferenceSettings.KV_LABELS.size - 1))
        threadsSlider.valueTo = InferenceSettings.MAX_THREADS.toFloat()

        fun refreshLabels() {
            tempLabel.text = getString(R.string.setting_temperature, tempSlider.value)
            topPLabel.text = getString(R.string.setting_top_p, topPSlider.value)
            topKLabel.text = getString(R.string.setting_top_k, topKSlider.value.toInt())
            repeatLabel.text = getString(R.string.setting_repeat_penalty, repeatSlider.value)
            maxTokensLabel.text = getString(R.string.setting_max_tokens, maxTokensSlider.value.toInt())
            threadsLabel.text = getString(R.string.setting_threads, threadsSlider.value.toInt())
        }
        tempSlider.value = inferenceSettings.temperature
        topPSlider.value = inferenceSettings.topP
        topKSlider.value = inferenceSettings.topK.toFloat()
        repeatSlider.value = inferenceSettings.repeatPenalty
        maxTokensSlider.value = inferenceSettings.maxTokens.toFloat()
        threadsSlider.value = prevThreads.toFloat().coerceIn(1f, InferenceSettings.MAX_THREADS.toFloat())
        refreshLabels()

        val listener = com.google.android.material.slider.Slider.OnChangeListener { _, _, _ -> refreshLabels() }
        tempSlider.addOnChangeListener(listener)
        topPSlider.addOnChangeListener(listener)
        topKSlider.addOnChangeListener(listener)
        repeatSlider.addOnChangeListener(listener)
        maxTokensSlider.addOnChangeListener(listener)
        threadsSlider.addOnChangeListener(listener)

        dialog.setOnDismissListener {
            inferenceSettings.contextSize =
                InferenceSettings.CONTEXT_OPTIONS[ctxSpinner.selectedItemPosition]
            inferenceSettings.kvCacheType = kvSpinner.selectedItemPosition
            inferenceSettings.threads = threadsSlider.value.toInt()
            inferenceSettings.temperature = tempSlider.value
            inferenceSettings.topP = topPSlider.value
            inferenceSettings.topK = topKSlider.value.toInt()
            inferenceSettings.repeatPenalty = repeatSlider.value
            inferenceSettings.maxTokens = maxTokensSlider.value.toInt()
            lifecycleScope.launch {
                runCatching {
                    engine?.updateSampling(
                        inferenceSettings.temperature,
                        inferenceSettings.topK,
                        inferenceSettings.topP,
                        inferenceSettings.repeatPenalty
                    )
                }
                if (isModelReady && (inferenceSettings.contextSize != prevCtx ||
                        inferenceSettings.kvCacheType != prevKv ||
                        inferenceSettings.threads != prevThreads)) {
                    showToast(getString(R.string.reload_model_to_apply))
                }
            }
        }

        dialog.setContentView(content)
        dialog.show()
    }

    private fun restoreConversations() {
        conversations += chatStore.loadConversations()
        val selectedId = chatStore.selectedConversationId()
        val selectedConversation = conversations.firstOrNull { it.id == selectedId }
            ?: conversations.firstOrNull()
            ?: newConversation()
        selectConversation(selectedConversation, saveSelection = false)
    }

    private val getContent = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        uri?.let { handleSelectedModel(it) }
    }

    private fun chooseModel() {
        if (!engineReady) {
            showToast(getString(R.string.engine_starting))
            return
        }
        if (generationJob?.isActive == true) {
            showToast(getString(R.string.wait_for_generation))
            return
        }
        getContent.launch(arrayOf("application/octet-stream", "*/*"))
    }

    private fun handleSelectedModel(uri: Uri) {
        setModelLoadingUi(getString(R.string.reading_model))
        lifecycleScope.launch {
            try {
                val fileName = withContext(Dispatchers.IO) {
                    contentResolver.query(uri, null, null, null, null)?.use { cursor ->
                        if (cursor.moveToFirst()) {
                            val idx = cursor.getColumnIndex("_display_name")
                            if (idx >= 0) cursor.getString(idx) else null
                        } else null
                    }
                } ?: uri.lastPathSegment ?: "model.bin"

                val modelFile = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        ensureModelFile(sanitizeModelName(fileName), input)
                    } ?: error("Unable to copy the selected model")
                }

                loadModel(modelFile.name, modelFile)
            } catch (exception: Exception) {
                Log.e(TAG, "Unable to import model", exception)
                showToast(getString(R.string.model_load_failed))
                isModelReady = false
                refreshUi()
            }
        }
    }

    private fun sanitizeModelName(rawName: String): String {
        var name = rawName.replace(Regex("[^A-Za-z0-9._-]"), "_")
        if (name.isBlank() || name == "." || name == "..") name = "model_${System.currentTimeMillis()}"
        return name
    }

    private suspend fun ensureModelFile(modelName: String, input: InputStream): File =
        withContext(Dispatchers.IO) {
            File(ensureModelsDirectory(), modelName).also { modelFile ->
                if (!modelFile.exists()) {
                    withContext(Dispatchers.Main) { setModelLoadingUi(getString(R.string.copying_model)) }
                    FileOutputStream(modelFile).use { output -> input.copyTo(output) }
                }
            }
        }

    private suspend fun loadModel(modelName: String, modelFile: File) {
        stopGeneration()
        setModelLoadingUi(getString(R.string.loading_model))
        try {
            withContext(Dispatchers.IO) {
                val inferenceEngine = requireNotNull(engine) { "Inference engine is not ready" }
                if (inferenceEngine.state.value.isModelLoaded) {
                    inferenceEngine.cleanUp()
                }
                inferenceEngine.configure(
                    inferenceSettings.contextSize,
                    inferenceSettings.kvCacheType,
                    inferenceSettings.threads
                )
                inferenceEngine.updateSampling(
                    inferenceSettings.temperature,
                    inferenceSettings.topK,
                    inferenceSettings.topP,
                    inferenceSettings.repeatPenalty
                )
                inferenceEngine.loadModel(modelFile.path)
                inferenceEngine.setTools(ToolDefinitions.buildToolsJson(toolSettings))
                nativeToolsSupported = inferenceEngine.hasNativeToolSupport()
                Log.i(TAG, "Model loaded: nativeToolsSupported=$nativeToolsSupported")
                inferenceEngine.setSystemPrompt(buildSystemPrompt())
            }
            selectedModelName = modelName
            chatStore.saveLastModelName(modelName)
            isModelReady = true
            refreshUi()
            updateCtxTracker()
        } catch (exception: Exception) {
            Log.e(TAG, "Unable to load model", exception)
            selectedModelName = null
            isModelReady = false
            chatStore.clearLastModelName()
            refreshUi()
            showToast(getString(R.string.model_load_failed))
        }
    }

    private fun toggleVoiceInput() {
        if (speechJob?.isActive == true || speechRecognizer.isListening) {
            stopVoiceInput()
            return
        }
        if (generationJob?.isActive == true) {
            showToast(getString(R.string.wait_for_generation))
            return
        }
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.RECORD_AUDIO) ==
            PackageManager.PERMISSION_GRANTED
        ) {
            startVoiceInput()
        } else {
            requestMicrophonePermission.launch(Manifest.permission.RECORD_AUDIO)
        }
    }

    private fun startVoiceInput() {
        if (speechJob?.isActive == true) return
        val typedPrefix = userInputEt.text?.toString()?.trimEnd().orEmpty()
        updateVoiceInputUi(active = true, loading = true)

        speechJob = lifecycleScope.launch {
            try {
                val transcript = speechRecognizer.listen { partial ->
                    withContext(Dispatchers.Main) {
                        updateVoiceInputUi(active = true, loading = false)
                        setDictatedText(typedPrefix, partial)
                    }
                }
                if (transcript.isBlank()) {
                    showToast(getString(R.string.stt_no_speech))
                } else {
                    setDictatedText(typedPrefix, transcript)
                }
            } catch (exception: CancellationException) {
                throw exception
            } catch (exception: Exception) {
                Log.e(TAG, "Offline speech recognition failed", exception)
                showToast(getString(R.string.stt_failed))
            } finally {
                updateVoiceInputUi(active = false, loading = false)
                speechJob = null
            }
        }
    }

    private fun stopVoiceInput() {
        speechRecognizer.stop()
        updateVoiceInputUi(active = false, loading = false)
    }

    private fun updateVoiceInputUi(active: Boolean, loading: Boolean) {
        micButton.setIconResource(if (active) R.drawable.outline_stop_24 else R.drawable.outline_mic_24)
        micButton.contentDescription = getString(
            if (active) R.string.stop_voice_input else R.string.start_voice_input
        )
        micButton.isActivated = active
        if (active) {
            micButton.tooltipText = getString(if (loading) R.string.stt_loading else R.string.stt_listening)
        } else {
            micButton.tooltipText = getString(R.string.start_voice_input)
        }
    }

    private fun setDictatedText(prefix: String, rawTranscript: String) {
        val transcript = rawTranscript.trim()
            .lowercase(Locale.US)
            .replaceFirstChar { character ->
                if (prefix.isBlank()) character.titlecase(Locale.US) else character.toString()
            }
        if (transcript.isBlank()) return
        val combined = if (prefix.isBlank()) transcript else "$prefix $transcript"
        userInputEt.setText(combined)
        userInputEt.setSelection(combined.length)
    }

    private fun stopGeneration() {
        engine?.stopGeneration()
        generationJob?.cancel()
    }

    private fun handleUserInput() {
        if (speechJob?.isActive == true) stopVoiceInput()
        val userMessage = userInputEt.text?.toString()?.trim().orEmpty()
        if (userMessage.isBlank()) {
            showToast(getString(R.string.empty_message))
            return
        }
        val inferenceEngine = engine ?: return
        if (generationJob?.isActive == true) return

        userInputEt.text = null
        userInputEt.isEnabled = false
        isGenerating = true
        updateSendButton()
        micButton.isEnabled = false

        appendMessage(Message(UUID.randomUUID().toString(), userMessage, true))
        appendMessage(Message(UUID.randomUUID().toString(), getString(R.string.thinking), false))
        saveConversationHistory()

        generationJob = lifecycleScope.launch(Dispatchers.Default) {
            var failure: Throwable? = null
            var producedOutput = false
            try {
                var toolCallCount = 0
                var streamedTokens = 0
                var isFirstTurn = true
                var lastToolCall: ToolCall? = null
                var lastToolResult: String = ""

                while (true) {
                    val generatedResponse = StringBuilder()
                    var turnEmittedTokens = 0

                    val tokenFlow = if (isFirstTurn || !nativeToolsSupported || lastToolCall == null) {
                        inferenceEngine.sendUserPrompt(
                            if (isFirstTurn) userMessage else toolResultPrompt(lastToolCall!!, lastToolResult),
                            inferenceSettings.maxTokens
                        )
                    } else {
                        val toolName = when (val completedCall = requireNotNull(lastToolCall)) {
                            is ToolCall.Sandbox -> completedCall.call.name
                            is ToolCall.WebSearch -> completedCall.call.action
                        }
                        inferenceEngine.sendToolResponse(
                            toolName,
                            "",
                            lastToolResult.ifEmpty { "(no output)" },
                            inferenceSettings.maxTokens
                        )
                    }
                    isFirstTurn = false

                    tokenFlow.collect { token ->
                        generatedResponse.append(token)
                        producedOutput = true
                        turnEmittedTokens++
                        if (++streamedTokens % 8 == 0) updateCtxTracker()
                        withContext(Dispatchers.Main) {
                            val split = splitResponse(generatedResponse.toString())
                            updateLastAssistantMessage(split.answer, split.thinking, split.tooling)
                            if (split.thinking.isNotBlank()) messageAdapter.expandThinking(messages.last().id)
                            if (split.tooling.isNotBlank()) messageAdapter.expandTooling(messages.last().id)
                        }
                    }

                    val rawResp = generatedResponse.toString()
                    val parsed = inferenceEngine.parseResponse(rawResp, isPartial = false)
                    val toolCall = parseFromNative(parsed) ?: parseAnyToolCall(rawResp)

                    withContext(Dispatchers.Main) {
                        if (parsed.hasParser && (parsed.content.isNotBlank() || parsed.thinking.isNotBlank())) {
                            val tooling = if (toolCall != null) {
                                extractAnyToolJson(rawResp)
                                    ?: parsed.toolCalls.firstOrNull()?.let { "${it.name}(${it.arguments})" }
                                    ?: ""
                            } else ""
                            updateLastAssistantMessage(parsed.content.trim(), parsed.thinking.trim(), tooling)
                        }
                    }

                    if (toolCall == null) {
                        if (turnEmittedTokens == 0 && toolCallCount > 0) {
                            withContext(Dispatchers.Main) {
                                updateLastAssistantMessage(getString(R.string.generation_stopped))
                            }
                        }
                        break
                    }
                    if (toolCallCount >= MAX_TOOL_CALLS_PER_TURN) break

                    val toolResult = withContext(Dispatchers.IO) { executeToolCall(toolCall) }
                    toolCallCount++
                    lastToolCall = toolCall
                    lastToolResult = toolResult
                    withContext(Dispatchers.Main) {
                        appendMessage(
                            Message(UUID.randomUUID().toString(), getString(R.string.thinking), false)
                        )
                    }
                }
            } catch (exception: CancellationException) {
                throw exception
            } catch (exception: Throwable) {
                failure = exception
                Log.e(TAG, "Generation failed", exception)
            } finally {
                updateCtxTracker()
                withContext(NonCancellable + Dispatchers.Main) {
                    if (!producedOutput) {
                        updateLastAssistantMessage(
                            if (failure == null) getString(R.string.generation_stopped)
                            else getString(R.string.generation_failed)
                        )
                    }
                    saveConversationHistory()
                    userInputEt.isEnabled = isModelReady
                    isGenerating = false
                    updateSendButton()
                    micButton.isEnabled = true
                    scrollMessagesToBottom()
                }
            }
        }
    }

    private fun appendMessage(message: Message) {
        messages += message
        activeConversation()?.messages?.add(message)
        messageAdapter.notifyItemInserted(messages.lastIndex)
        updateEmptyState()
        scrollMessagesToBottom()
    }

    private fun updateLastAssistantMessage(content: String, thinking: String = "", tooling: String = "") {
        val index = messages.lastIndex
        if (index < 0 || messages[index].isUser) return
        val updatedMessage = messages[index].copy(content = content, thinking = thinking, tooling = tooling)
        if (thinking.isNotBlank()) {
            Log.d(TAG, "updateLastAssistantMessage idx=$index thinking=${thinking.take(60)} content=${content.take(60)} tooling=${tooling.take(60)}")
        }
        messages[index] = updatedMessage
        activeConversation()?.messages?.let { conversationMessages ->
            val conversationIndex = conversationMessages.indexOfLast { it.id == updatedMessage.id }
            if (conversationIndex >= 0) conversationMessages[conversationIndex] = updatedMessage
        }
        messageAdapter.notifyItemChanged(index)
        scrollMessagesToBottom()
    }

    private fun splitResponse(raw: String): ResponseSplit {
        var working = raw
        var thinking = ""
        val open = working.indexOf(THINK_OPEN)
        if (open >= 0) {
            val prefix = working.substring(0, open)
            val afterOpen = working.substring(open + THINK_OPEN.length)
            val close = afterOpen.indexOf(THINK_CLOSE)
            if (close >= 0) {
                thinking = afterOpen.substring(0, close).trim()
                working = prefix + afterOpen.substring(close + THINK_CLOSE.length)
            } else {
                thinking = afterOpen.trim()
                working = prefix
            }
        } else {
            // Fallback: If <think> was prefilled into the prompt by the model's chat template,
            // the generated response starts directly with thinking and ends with </think>
            val close = working.indexOf(THINK_CLOSE)
            if (close >= 0) {
                thinking = working.substring(0, close).trim()
                working = working.substring(close + THINK_CLOSE.length)
            }
        }
        val tooling = extractAnyToolJson(working) ?: ""
        val answer = if (tooling.isNotBlank()) working.replace(tooling, "").trim() else working.trim()
        if (thinking.isNotBlank() || tooling.isNotBlank()) {
            Log.d(TAG, "splitResponse thinking=${thinking.take(60)} tooling=${tooling.take(60)} answer=${answer.take(60)}")
        }
        return ResponseSplit(thinking = thinking, tooling = tooling, answer = answer)
    }

    private fun createConversation() {
        if (generationJob?.isActive == true) {
            showToast(getString(R.string.wait_for_generation))
            return
        }
        selectConversation(newConversation())
    }

    private fun newConversation(): Conversation {
        return Conversation(
            id = UUID.randomUUID().toString(),
            title = getString(R.string.new_conversation),
            createdAt = System.currentTimeMillis(),
            messages = mutableListOf()
        ).also { conversations.add(0, it) }
    }

    private fun selectConversation(conversation: Conversation, saveSelection: Boolean = true) {
        activeConversationId = conversation.id
        messages.clear()
        messages.addAll(conversation.messages)
        messageAdapter.notifyDataSetChanged()
        updateEmptyState()
        scrollMessagesToBottom()
        if (saveSelection) saveConversationHistory()
    }

    private fun activeConversation(): Conversation? =
        conversations.firstOrNull { it.id == activeConversationId }

    private fun saveConversationHistory() {
        activeConversation()?.let { conversation ->
            conversation.messages.firstOrNull { it.isUser }?.content?.let { firstMessage ->
                conversation.title = firstMessage.take(CONVERSATION_TITLE_LENGTH)
            }
        }
        chatStore.saveConversations(conversations, activeConversationId)
    }

    private fun parseFromNative(parsed: ParsedResponse): ToolCall? {
        if (!toolSettings.allToolsEnabled || !parsed.hasParser) return null
        val nativeCall = parsed.toolCalls.firstOrNull() ?: return null
        val arguments = runCatching {
            if (nativeCall.arguments.isBlank()) JSONObject() else JSONObject(nativeCall.arguments)
        }.getOrDefault(JSONObject())
        arguments.put("name", nativeCall.name)
        if (toolSettings.sandboxToolsEnabled) {
            sandboxTools.parseToolCall(arguments)?.let { return ToolCall.Sandbox(it) }
        }
        if (toolSettings.webToolsEnabled) {
            webSearchTools.parseToolCall(arguments)?.let { return ToolCall.WebSearch(it) }
        }
        return null
    }

    private fun parseAnyToolCall(response: String): ToolCall? {
        if (!toolSettings.allToolsEnabled) return null
        val sandbox = sandboxTools.parseToolCall(response)
        if (sandbox != null) return ToolCall.Sandbox(sandbox)
        if (toolSettings.webToolsEnabled) {
            val web = webSearchTools.parseToolCall(response)
            if (web != null) return ToolCall.WebSearch(web)
        }
        return null
    }

    private fun extractAnyToolJson(response: String): String? {
        if (!toolSettings.allToolsEnabled) return null
        sandboxTools.extractToolJson(response)?.let { return it }
        if (toolSettings.webToolsEnabled) {
            webSearchTools.extractWebJson(response)?.let { return it }
        }
        LfmToolParser.extractToolBlock(response)?.let { return it }
        return null
    }

    private suspend fun executeToolCall(call: ToolCall): String = when (call) {
        is ToolCall.Sandbox -> sandboxTools.execute(call.call)
        is ToolCall.WebSearch -> webSearchTools.execute(call.call)
    }

    private fun toolResultPrompt(call: ToolCall, result: String): String = when (call) {
        is ToolCall.Sandbox -> sandboxTools.toolResultPrompt(call.call, result)
        is ToolCall.WebSearch -> webSearchTools.toolResultPrompt(call.call, result)
    }

    private fun showConversationHistory() {
        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_conversation_history, null)
        val historyRv = content.findViewById<RecyclerView>(R.id.conversation_history)
        val newChatBtn = content.findViewById<MaterialButton>(R.id.new_chat)
        val adapter = ConversationAdapter(conversations, activeConversationId) { conversation ->
            if (generationJob?.isActive == true) {
                showToast(getString(R.string.wait_for_generation))
            } else {
                selectConversation(conversation)
                dialog.dismiss()
            }
        }
        historyRv.layoutManager = LinearLayoutManager(this)
        historyRv.adapter = adapter
        newChatBtn.setOnClickListener {
            createConversation()
            dialog.dismiss()
        }
        dialog.setContentView(content)
        dialog.show()
    }

    private fun showModelManager() {
        if (generationJob?.isActive == true) {
            showToast(getString(R.string.wait_for_generation))
            return
        }

        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_model_manager, null)
        val urlInput = content.findViewById<TextInputEditText>(R.id.hugging_face_url)
        val tokenInput = content.findViewById<TextInputEditText>(R.id.hugging_face_token)
        val browseButton = content.findViewById<MaterialButton>(R.id.browse_hugging_face_repository)
        val importButton = content.findViewById<MaterialButton>(R.id.import_local_model)
        val progress = content.findViewById<ProgressBar>(R.id.download_progress)
        val downloadStatus = content.findViewById<TextView>(R.id.download_status)
        val remoteHeading = content.findViewById<TextView>(R.id.remote_models_heading)
        val remoteModelsRv = content.findViewById<RecyclerView>(R.id.remote_models)
        val modelsRv = content.findViewById<RecyclerView>(R.id.local_models)

        lateinit var modelsAdapter: LocalModelAdapter
        lateinit var remoteModelsAdapter: RemoteGgufAdapter

        fun setRepositoryControlsEnabled(enabled: Boolean) {
            browseButton.isEnabled = enabled
            importButton.isEnabled = enabled
            urlInput.isEnabled = enabled
            tokenInput.isEnabled = enabled
            dialog.setCancelable(enabled)
        }

        fun updateDownloadProgress(downloaded: Long, total: Long) {
            runOnUiThread {
                if (total > 0) {
                    progress.isIndeterminate = false
                    progress.progress = ((downloaded * 100) / total).toInt().coerceIn(0, 100)
                    downloadStatus.text = getString(R.string.download_progress, progress.progress)
                } else {
                    downloadStatus.text = getString(
                        R.string.download_bytes,
                        (downloaded / BYTES_PER_MEGABYTE).toInt()
                    )
                }
            }
        }

        fun downloadRemoteFile(file: HuggingFaceGgufFile) {
            if (modelDownloadJob?.isActive == true || repositoryIndexJob?.isActive == true) return
            val token = tokenInput.text?.toString()?.takeIf { it.isNotBlank() }
            setRepositoryControlsEnabled(false)
            remoteModelsAdapter.setDownloading(file.path)
            progress.visibility = View.VISIBLE
            progress.isIndeterminate = true
            downloadStatus.text = getString(R.string.download_bytes, 0)

            modelDownloadJob = lifecycleScope.launch {
                try {
                    val modelFile = modelRepository.download(file, token, ::updateDownloadProgress)
                    modelsAdapter.replaceModels(modelRepository.localModels())
                    downloadStatus.text = getString(R.string.download_complete, modelFile.name)
                } catch (exception: CancellationException) {
                    throw exception
                } catch (exception: Exception) {
                    Log.e(TAG, "Model download failed", exception)
                    downloadStatus.text = getString(
                        R.string.download_failed,
                        exception.message ?: getString(R.string.model_load_failed)
                    )
                } finally {
                    progress.visibility = View.GONE
                    remoteModelsAdapter.setDownloading(null)
                    setRepositoryControlsEnabled(true)
                }
            }
        }

        remoteModelsAdapter = RemoteGgufAdapter(onDownload = ::downloadRemoteFile)
        remoteModelsRv.layoutManager = LinearLayoutManager(this)
        // Fixed-height lists consume their own scrolls; the panel only
        // scrolls on touches outside them.
        remoteModelsRv.isNestedScrollingEnabled = false
        remoteModelsRv.setOnTouchListener { view, event ->
            if (event.actionMasked == MotionEvent.ACTION_DOWN) {
                view.parent.requestDisallowInterceptTouchEvent(true)
            }
            false
        }
        remoteModelsRv.adapter = remoteModelsAdapter

        modelsAdapter = LocalModelAdapter(
            models = modelRepository.localModels(),
            selectedModelName = { selectedModelName },
            onLoad = { localModel ->
                if (!engineReady) {
                    showToast(getString(R.string.engine_starting))
                    return@LocalModelAdapter
                }
                dialog.dismiss()
                stopGeneration()
                lifecycleScope.launch { loadModel(localModel.file.name, localModel.file) }
            },
            onDelete = { modelFile ->
                confirmDeleteModel(modelFile) {
                    modelsAdapter.replaceModels(modelRepository.localModels())
                }
            }
        )
        modelsRv.layoutManager = LinearLayoutManager(this)
        modelsRv.isNestedScrollingEnabled = false
        modelsRv.setOnTouchListener { view, event ->
            if (event.actionMasked == MotionEvent.ACTION_DOWN) {
                view.parent.requestDisallowInterceptTouchEvent(true)
            }
            false
        }
        modelsRv.adapter = modelsAdapter

        importButton.setOnClickListener {
            dialog.dismiss()
            chooseModel()
        }
        browseButton.setOnClickListener {
            if (modelDownloadJob?.isActive == true || repositoryIndexJob?.isActive == true) {
                return@setOnClickListener
            }
            val source = urlInput.text?.toString().orEmpty()
            val token = tokenInput.text?.toString()?.takeIf { it.isNotBlank() }
            setRepositoryControlsEnabled(false)
            progress.visibility = View.VISIBLE
            progress.isIndeterminate = true
            downloadStatus.text = getString(R.string.repo_indexing)

            repositoryIndexJob = lifecycleScope.launch {
                try {
                    val files = modelRepository.listGgufFiles(source, token)
                    remoteModelsAdapter.replaceFiles(files)
                    remoteHeading.visibility = View.VISIBLE
                    remoteModelsRv.visibility = View.VISIBLE
                    downloadStatus.text = getString(R.string.repo_indexed, files.size)
                } catch (exception: CancellationException) {
                    throw exception
                } catch (exception: Exception) {
                    Log.e(TAG, "Repository indexing failed", exception)
                    downloadStatus.text = getString(
                        R.string.download_failed,
                        exception.message ?: getString(R.string.model_load_failed)
                    )
                } finally {
                    progress.visibility = View.GONE
                    setRepositoryControlsEnabled(true)
                }
            }
        }

        dialog.setContentView(content)
        dialog.show()
    }

    private fun confirmDeleteModel(modelFile: File, onFinished: () -> Unit) {
        MaterialAlertDialogBuilder(this)
            .setMessage(getString(R.string.confirm_delete_model, modelFile.name))
            .setNegativeButton(android.R.string.cancel, null)
            .setPositiveButton(R.string.delete) { _, _ ->
                stopGeneration()
                lifecycleScope.launch {
                    val wasLoaded = modelFile.name == selectedModelName
                    val deleted = runCatching {
                        withContext(Dispatchers.IO) {
                            if (wasLoaded) {
                                if (engine?.state?.value?.isModelLoaded == true) engine?.cleanUp()
                            }
                            modelFile.delete()
                        }
                    }.getOrDefault(false)

                    if (wasLoaded) {
                        selectedModelName = null
                        isModelReady = false
                        chatStore.clearLastModelName()
                        refreshUi()
                    }
                    showToast(
                        getString(if (deleted) R.string.model_deleted else R.string.model_delete_failed)
                    )
                    onFinished()
                }
            }
            .show()
    }

    private fun showSandboxFiles() {
        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_sandbox_files, null)
        val location = content.findViewById<TextView>(R.id.sandbox_location)
        val picker = content.findViewById<AutoCompleteTextView>(R.id.sandbox_file_picker)
        val pathInput = content.findViewById<TextInputEditText>(R.id.sandbox_path)
        val editor = content.findViewById<TextInputEditText>(R.id.sandbox_editor)
        val readButton = content.findViewById<MaterialButton>(R.id.read_sandbox_file)
        val saveButton = content.findViewById<MaterialButton>(R.id.save_sandbox_file)
        val deleteButton = content.findViewById<MaterialButton>(R.id.delete_sandbox_file)

        location.text = getString(R.string.sandbox_location, sandboxTools.rootDirectory.absolutePath)

        fun refreshFilePicker(selectedPath: String? = null) {
            val files = sandboxTools.listFiles()
            picker.setAdapter(ArrayAdapter(this, android.R.layout.simple_dropdown_item_1line, files))
            selectedPath?.let { picker.setText(it, false) }
        }

        fun readFile(relativePath: String) {
            lifecycleScope.launch {
                runCatching { withContext(Dispatchers.IO) { sandboxTools.readFile(relativePath) } }
                    .onSuccess { contents ->
                        pathInput.setText(relativePath)
                        editor.setText(contents)
                        refreshFilePicker(relativePath)
                    }
                    .onFailure { exception ->
                        showToast(getString(R.string.file_operation_failed, exception.message ?: "read failed"))
                    }
            }
        }

        refreshFilePicker()
        picker.setOnItemClickListener { _, _, position, _ ->
            readFile(picker.adapter.getItem(position).toString())
        }
        readButton.setOnClickListener {
            readFile(pathInput.text?.toString()?.trim().orEmpty())
        }
        saveButton.setOnClickListener {
            val path = pathInput.text?.toString()?.trim().orEmpty()
            val contents = editor.text?.toString().orEmpty()
            lifecycleScope.launch {
                runCatching { withContext(Dispatchers.IO) { sandboxTools.writeFile(path, contents) } }
                    .onSuccess {
                        showToast(getString(R.string.file_saved))
                        refreshFilePicker(path)
                    }
                    .onFailure { exception ->
                        showToast(getString(R.string.file_operation_failed, exception.message ?: "save failed"))
                    }
            }
        }
        deleteButton.setOnClickListener {
            val path = pathInput.text?.toString()?.trim().orEmpty()
            MaterialAlertDialogBuilder(this)
                .setMessage(getString(R.string.confirm_delete_file, path))
                .setNegativeButton(android.R.string.cancel, null)
                .setPositiveButton(R.string.delete) { _, _ ->
                    val deleted = runCatching { sandboxTools.deleteFile(path) }.getOrDefault(false)
                    if (deleted) {
                        pathInput.text = null
                        editor.text = null
                        picker.text = null
                        refreshFilePicker()
                        showToast(getString(R.string.file_deleted))
                    } else {
                        showToast(getString(R.string.file_operation_failed, "file not found"))
                    }
                }
                .show()
        }

        dialog.setContentView(content)
        dialog.show()
    }

    private fun setModelLoadingUi(status: String) {
        isModelReady = false
        modelStatusTv.text = status
        userInputEt.isEnabled = false
        userActionBtn.isEnabled = false
        micButton.isEnabled = false
        changeModelBtn.isEnabled = false
        ramTracker.visibility = View.VISIBLE
    }

    private fun updateCtxTracker() {
        val inferenceEngine = engine ?: return
        lifecycleScope.launch(Dispatchers.Default) {
            val usage = runCatching { inferenceEngine.contextUsage() }.getOrNull() ?: return@launch
            withContext(Dispatchers.Main) { renderCtxTracker(usage.first, usage.second) }
        }
    }

    private fun renderCtxTracker(used: Int, total: Int) {
        if (!isModelReady || total <= 0) return
        ctxUsageTv.text = getString(R.string.context_usage, used.coerceAtLeast(0), total)
        ctxBar.progress = ((used.coerceAtLeast(0) * 100L) / total).toInt().coerceIn(0, 100)
        ctxUsageTv.setTextColor(
            if (used * 100L / total >= 90) getColor(android.R.color.holo_red_dark)
            else getColor(R.color.model_status_text)
        )
    }

    private fun startRamTracker() {
        lifecycleScope.launch {
            val actManager = getSystemService(Context.ACTIVITY_SERVICE) as? ActivityManager
            val memInfo = ActivityManager.MemoryInfo()
            while (isActive) {
                updateRamTracker(actManager, memInfo)
                delay(2500)
            }
        }
    }

    private fun updateRamTracker(actManager: ActivityManager?, memInfo: ActivityManager.MemoryInfo) {
        if (actManager == null) return
        actManager.getMemoryInfo(memInfo)
        val totalBytes = memInfo.totalMem
        val availBytes = memInfo.availMem
        val usedBytes = (totalBytes - availBytes).coerceAtLeast(0)
        val totalGB = totalBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
        val usedGB = usedBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
        val freeGB = availBytes.toDouble() / (1024.0 * 1024.0 * 1024.0)
        val usedPercent = if (totalBytes > 0) ((usedBytes * 100L) / totalBytes).toInt().coerceIn(0, 100) else 0

        ramUsageTv.text = getString(R.string.ram_usage, usedGB, totalGB, freeGB)
        ramBar.progress = usedPercent

        if (memInfo.lowMemory || usedPercent >= 90) {
            ramUsageTv.setTextColor(getColor(android.R.color.holo_red_dark))
            ramBar.progressTintList = ColorStateList.valueOf(getColor(android.R.color.holo_red_dark))
        } else if (usedPercent >= 80) {
            ramUsageTv.setTextColor(getColor(android.R.color.holo_orange_dark))
            ramBar.progressTintList = ColorStateList.valueOf(getColor(android.R.color.holo_orange_dark))
        } else {
            ramUsageTv.setTextColor(getColor(R.color.model_status_text))
            ramBar.progressTintList = null
        }
    }

    private fun refreshUi() {
        modelStatusTv.text = when {
            !engineReady -> getString(R.string.engine_starting)
            isModelReady -> getString(R.string.model_ready, selectedModelName ?: "")
            else -> getString(R.string.no_model_selected)
        }
        ctxTracker.visibility = if (isModelReady) View.VISIBLE else View.GONE
        ramTracker.visibility = if (isModelReady) View.VISIBLE else View.GONE
        val generationActive = generationJob?.isActive == true
        userInputEt.isEnabled = engineReady && !generationActive && !isGenerating
        changeModelBtn.isEnabled = engineReady && generationJob?.isActive != true
        micButton.isEnabled = !generationActive && !isGenerating
        updateSendButton()
        updateEmptyState()
    }

    private fun updateSendButton() {
        if (isGenerating) {
            userActionBtn.isEnabled = true
            userActionBtn.text = getString(R.string.stop)
            userActionBtn.setIconResource(R.drawable.outline_stop_24)
        } else if (isModelReady) {
            userActionBtn.isEnabled = engineReady
            userActionBtn.text = getString(R.string.send)
            userActionBtn.setIconResource(R.drawable.outline_send_24)
        } else {
            userActionBtn.isEnabled = engineReady
            userActionBtn.text = null
            userActionBtn.setIconResource(R.drawable.outline_folder_open_24)
        }
    }

    private fun updateEmptyState() {
        emptyStateTv.visibility = if (messages.isEmpty()) android.view.View.VISIBLE else android.view.View.GONE
    }

    private fun scrollMessagesToBottom() {
        if (messages.isNotEmpty()) messagesRv.post { messagesRv.scrollToPosition(messages.lastIndex) }
    }

    private fun setupKeyboardHandling() {
        val root = findViewById<View>(R.id.main)
        val composerCard = findViewById<View>(R.id.composer)
        val baseMargin = (resources.displayMetrics.density * 12).toInt()
        ViewCompat.setOnApplyWindowInsetsListener(root) { _, insets ->
            val systemBars = insets.getInsets(WindowInsetsCompat.Type.systemBars())
            val ime = insets.getInsets(WindowInsetsCompat.Type.ime())
            root.setPadding(root.paddingLeft, systemBars.top, root.paddingRight, systemBars.bottom)
            val params = composerCard.layoutParams as ViewGroup.MarginLayoutParams
            params.bottomMargin = baseMargin + ime.bottom
            composerCard.layoutParams = params
            if (ime.bottom > 0) scrollMessagesToBottom()
            insets
        }
    }

    private fun ensureModelsDirectory(): File = File(filesDir, DIRECTORY_MODELS).also { directory ->
        if (directory.exists() && !directory.isDirectory) directory.delete()
        if (!directory.exists()) directory.mkdirs()
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onStop() {
        saveConversationHistory()
        generationJob?.cancel()
        stopVoiceInput()
        super.onStop()
    }

    override fun onDestroy() {
        modelDownloadJob?.cancel()
        repositoryIndexJob?.cancel()
        speechJob?.cancel()
        generationJob?.cancel()
        stopGeneration()
        speechRecognizer.close()
        // destroy() unloads the model itself, so no separate unload here
        runCatching { engine?.destroy() }
        super.onDestroy()
    }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        when (level) {
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_CRITICAL -> {
                stopGeneration()
                unloadAllModels()
            }
            ComponentCallbacks2.TRIM_MEMORY_RUNNING_LOW -> {
                stopGeneration()
            }
        }
    }

    private fun unloadAllModels() {
        stopGeneration()
        lifecycleScope.launch {
            runCatching { engine?.cleanUp() }
            selectedModelName = null
            isModelReady = false
            refreshUi()
        }
    }

    companion object {
        private val TAG = MainActivity::class.java.simpleName
        private const val DIRECTORY_MODELS = "models"
        private const val FILE_EXTENSION_GGUF = ".gguf"
        private const val CONVERSATION_TITLE_LENGTH = 48
        private const val MAX_TOOL_CALLS_PER_TURN = 3
        private const val BYTES_PER_MEGABYTE = 1024L * 1024L
        private const val THINK_OPEN = "<think>"
        private const val THINK_CLOSE = "</think>"
    }
}

fun GgufMetadata.filename() = when {
    basic.name != null -> {
        basic.name?.let { name ->
            basic.sizeLabel?.let { size -> "$name-$size" } ?: name
        }
    }
    architecture?.architecture != null -> {
        architecture?.architecture?.let { architecture ->
            basic.uuid?.let { uuid -> "$architecture-$uuid" }
                ?: "$architecture-${System.currentTimeMillis()}"
        }
    }
    else -> "model-${System.currentTimeMillis().toHexString()}"
}
