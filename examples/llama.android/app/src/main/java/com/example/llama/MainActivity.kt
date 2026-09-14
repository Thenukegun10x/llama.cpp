package com.example.llama

import android.Manifest
import android.content.ComponentCallbacks2
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.ArrayAdapter
import android.widget.AutoCompleteTextView
import android.widget.ImageView
import android.widget.ProgressBar
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
import com.arm.aichat.ImageGenEngine
import com.arm.aichat.InferenceEngine
import com.arm.aichat.gguf.GgufMetadata
import com.arm.aichat.internal.ImageGenEngineImpl
import com.arm.aichat.isModelLoaded
import com.google.android.material.appbar.MaterialToolbar
import com.google.android.material.bottomsheet.BottomSheetDialog
import com.google.android.material.button.MaterialButton
import com.google.android.material.checkbox.MaterialCheckBox
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.google.android.material.textfield.TextInputEditText
import android.view.ViewGroup

import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.NonCancellable
import kotlinx.coroutines.flow.filter
import kotlinx.coroutines.flow.first
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
    data class ImageGen(val call: ImageGenCall) : ToolCall()
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
    private lateinit var chatStore: ConversationStore
    private lateinit var sandboxTools: SandboxFileTools
    private lateinit var modelRepository: HuggingFaceModelRepository
    private lateinit var speechRecognizer: OnDeviceSpeechRecognizer
    private lateinit var toolSettings: ToolSettings
    private var imageGenEngine: ImageGenEngine? = null
    private var imageGenTool: ImageGenTool? = null

    private var engine: InferenceEngine? = null
    private var engineReady = false
    private var isModelReady = false
    private var isGenerating = false
    private var selectedModelName: String? = null
    private var generationJob: Job? = null
    private var modelDownloadJob: Job? = null
    private var repositoryIndexJob: Job? = null
    private var speechJob: Job? = null

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
        modelRepository = HuggingFaceModelRepository(applicationContext)
        speechRecognizer = OnDeviceSpeechRecognizer(applicationContext)
        toolSettings = ToolSettings(applicationContext)
        initImageGenEngine()
        toolbar = findViewById(R.id.toolbar)
        modelStatusTv = findViewById(R.id.model_status)
        messagesRv = findViewById(R.id.messages)
        emptyStateTv = findViewById(R.id.empty_state)
        userInputEt = findViewById(R.id.user_input)
        userActionBtn = findViewById(R.id.user_action)
        micButton = findViewById(R.id.mic_button)
        changeModelBtn = findViewById(R.id.change_model)

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
                R.id.action_image_gen -> {
                    showImageGeneration()
                    true
                }
                R.id.action_tool_settings -> {
                    showToolSettings()
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
                restoreLastModel()
            } catch (exception: Exception) {
                Log.e(TAG, "Unable to initialize inference engine", exception)
                showToast(getString(R.string.engine_start_failed))
                modelStatusTv.text = getString(R.string.engine_unavailable)
            }
        }
    }

    private fun initImageGenEngine() {
        try {
            val outputDir = File(filesDir, "generated_images").apply { mkdirs() }
            val sdEngine = ImageGenEngineImpl(outputDir)
            if (sdEngine.state.value.mode != ImageGenEngine.Mode.ERROR) {
                imageGenEngine = sdEngine
                imageGenTool = ImageGenTool(sdEngine, outputDir)
                lifecycleScope.launch { sdEngine.init() }
            }
        } catch (e: Exception) {
            Log.w(TAG, "Image gen not available", e)
        }
    }

    private fun buildSystemPrompt(): String {
        val parts = mutableListOf<String>()
        parts.add("You are a helpful private, offline assistant. Answer normally unless a tool is needed.")

        if (!toolSettings.allToolsEnabled) {
            parts.add("")
            parts.add("When using a tool, output only one call in this exact wrapper:")
            parts.add("""<tool_call>{"name":"tool_name"}</tool_call>""")
            parts.add("Use one tool at a time. Wait for <tool_result> before reporting success. Never invent a tool result.")
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
        if (toolSettings.imageGenEnabled && imageGenEngine != null &&
            imageGenEngine?.state?.value?.mode == ImageGenEngine.Mode.READY) {
            if (toolDescs.isNotEmpty()) toolDescs.add("")
            toolDescs.add("IMAGE GENERATION TOOL:")
            toolDescs.add("""generate_image: {"name":"generate_image","prompt":"a detailed description of the image","negative_prompt":"things to avoid","width":512,"height":512,"steps":20,"cfg_scale":7.0,"seed":-1}""")
            toolDescs.add("  - prompt: detailed description of the image to generate (required)")
            toolDescs.add("  - negative_prompt: things to avoid in the image (optional, default: empty)")
            toolDescs.add("  - width/height: 256-2048 pixels (optional, default: 512)")
            toolDescs.add("  - steps: 1-50, higher = more detail but slower (optional, default: 20)")
            toolDescs.add("  - cfg_scale: 1.0-30.0, higher = follows prompt more strictly (optional, default: 7.0)")
            toolDescs.add("  - seed: random seed, -1 for random (optional, default: -1)")
        }

        if (toolDescs.isNotEmpty()) {
            parts.add("")
            parts.addAll(toolDescs)
        }

        parts.add("")
        parts.add("When using a tool, output only one call in this exact wrapper:")
        parts.add("""<tool_call>{"name":"tool_name","prompt":"..."}</tool_call>""")
        parts.add("Use one tool at a time. Wait for <tool_result> before reporting success. Never invent a tool result.")
        return parts.joinToString("\n")
    }

    private fun showToolSettings() {
        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_tool_settings, null)
        val allToggle = content.findViewById<MaterialCheckBox>(R.id.toggle_all_tools)
        val sandboxToggle = content.findViewById<MaterialCheckBox>(R.id.toggle_sandbox)
        val imageGenToggle = content.findViewById<MaterialCheckBox>(R.id.toggle_image_gen)
        val imageGenSection = content.findViewById<View>(R.id.image_gen_section)

        allToggle.isChecked = toolSettings.allToolsEnabled
        sandboxToggle.isChecked = toolSettings.sandboxToolsEnabled
        imageGenToggle.isChecked = toolSettings.imageGenEnabled

        if (imageGenEngine == null || imageGenEngine?.state?.value?.mode == ImageGenEngine.Mode.ERROR) {
            imageGenSection.visibility = View.GONE
        }

        fun applyChildEnabled() {
            val enabled = allToggle.isChecked
            sandboxToggle.isEnabled = enabled
            imageGenToggle.isEnabled = enabled && imageGenSection.visibility == View.VISIBLE
        }
        applyChildEnabled()

        allToggle.setOnCheckedChangeListener { _, _ -> applyChildEnabled() }

        dialog.setOnDismissListener {
            toolSettings.allToolsEnabled = allToggle.isChecked
            toolSettings.sandboxToolsEnabled = sandboxToggle.isChecked
            toolSettings.imageGenEnabled = imageGenToggle.isChecked
        }

        dialog.setContentView(content)
        dialog.show()
    }

    private fun showImageGeneration() {
        val dialog = BottomSheetDialog(this)
        val content = layoutInflater.inflate(R.layout.dialog_image_gen, null)
        val statusTv = content.findViewById<TextView>(R.id.sd_model_status)
        val promptEt = content.findViewById<TextInputEditText>(R.id.image_gen_prompt)
        val negativeEt = content.findViewById<TextInputEditText>(R.id.image_gen_negative)
        val stepsSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.image_gen_steps)
        val cfgSlider = content.findViewById<com.google.android.material.slider.Slider>(R.id.image_gen_cfg)
        val seedEt = content.findViewById<TextInputEditText>(R.id.image_gen_seed)
        val generateBtn = content.findViewById<MaterialButton>(R.id.generate_button)
        val progress = content.findViewById<ProgressBar>(R.id.gen_progress)
        val statusTv2 = content.findViewById<TextView>(R.id.gen_status)
        val genImage = content.findViewById<ImageView>(R.id.gen_image)
        val resultInfo = content.findViewById<TextView>(R.id.gen_result_info)
        val loadBtn = content.findViewById<MaterialButton>(R.id.load_sd_model)
        val sq512 = content.findViewById<MaterialButton>(R.id.size_sq_512)
        val landscape = content.findViewById<MaterialButton>(R.id.size_landscape)
        val portrait = content.findViewById<MaterialButton>(R.id.size_portrait)
        val hd = content.findViewById<MaterialButton>(R.id.size_hd)

        var width = 512
        var height = 512
        var generating = false
        var generationJob: Job? = null

        val sizeLabel = content.findViewById<TextView>(R.id.size_label)!!

        fun setSize(w: Int, h: Int) { width = w; height = h; sizeLabel.text = "Size: ${w}×${h}" }

        sq512.setOnClickListener { setSize(512, 512) }
        landscape.setOnClickListener { setSize(768, 512) }
        portrait.setOnClickListener { setSize(512, 768) }
        hd.setOnClickListener { setSize(1024, 1024) }
        setSize(512, 512)

        stepsSlider.addOnChangeListener { _, value, _ ->
            content.findViewById<TextView>(R.id.steps_label).text = "Steps: ${value.toInt()}"
        }
        cfgSlider.addOnChangeListener { _, value, _ ->
            content.findViewById<TextView>(R.id.cfg_label).text = "CFG Scale: ${String.format(Locale.US, "%.1f", value)}"
        }
        stepsSlider.value = 20f
        cfgSlider.value = 7.0f

        fun refreshStatus() {
            val engine = imageGenEngine
            when {
                engine == null -> {
                    statusTv.text = "Image generation not available"
                    generateBtn.isEnabled = false
                    loadBtn.visibility = View.GONE
                }
                engine.state.value.mode == ImageGenEngine.Mode.ERROR -> {
                    statusTv.text = "Error: ${engine.state.value.error ?: "unknown"}"
                    generateBtn.isEnabled = false
                    loadBtn.visibility = View.GONE
                }
                engine.state.value.mode == ImageGenEngine.Mode.GENERATING -> {
                    statusTv.text = "Generating..."
                    generateBtn.isEnabled = false
                    loadBtn.visibility = View.GONE
                }
                engine.state.value.mode == ImageGenEngine.Mode.READY -> {
                    statusTv.text = "Model ready"
                    generateBtn.isEnabled = !generating
                    loadBtn.visibility = View.GONE
                }
                engine.state.value.mode == ImageGenEngine.Mode.LOADING -> {
                    statusTv.text = "Loading model..."
                    generateBtn.isEnabled = false
                    loadBtn.visibility = View.GONE
                }
                engine.state.value.mode == ImageGenEngine.Mode.UNLOADED -> {
                    statusTv.text = "No image generation model loaded"
                    generateBtn.isEnabled = false
                    loadBtn.visibility = View.VISIBLE
                }
            }
        }
        refreshStatus()

        val stateJob = lifecycleScope.launch {
            imageGenEngine?.state?.collect {
                withContext(Dispatchers.Main) { refreshStatus() }
            }
        }
        dialog.setOnDismissListener {
            stateJob.cancel()
            generationJob?.cancel()
        }

        generateBtn.setOnClickListener {
            val prompt = promptEt.text?.toString()?.trim()
            if (prompt.isNullOrBlank() || generating) return@setOnClickListener

            generating = true
            generateBtn.isEnabled = false
            progress.visibility = View.VISIBLE
            progress.isIndeterminate = true
            statusTv2.visibility = View.VISIBLE
            statusTv2.text = "Generating..."
            genImage.visibility = View.GONE
            resultInfo.visibility = View.GONE

            generationJob = lifecycleScope.launch {
                try {
                    val seed = seedEt.text?.toString()?.trim()?.toLongOrNull() ?: -1L
                    val negative = negativeEt.text?.toString()?.trim() ?: ""
                    val steps = stepsSlider.value.toInt()
                    val cfg = cfgSlider.value

                    val result = withContext(Dispatchers.IO) {
                        imageGenEngine?.generateImage(prompt, negative, width, height, steps, cfg, seed)
                    }

                    if (result != null) {
                        val file = File(result)
                        val bitmap = android.graphics.BitmapFactory.decodeFile(result)
                        if (bitmap != null) {
                            genImage.setImageBitmap(bitmap)
                            genImage.visibility = View.VISIBLE
                            resultInfo.text = "${width}×${height} | ${file.length() / 1024}KB | seed=${seed}"
                            resultInfo.visibility = View.VISIBLE
                            statusTv2.text = "Done"
                        } else {
                            statusTv2.text = "Failed to decode image"
                        }
                    } else {
                        statusTv2.text = "Generation failed"
                    }
                } catch (e: CancellationException) {
                    statusTv2.text = "Cancelled"
                } catch (e: Exception) {
                    statusTv2.text = "Error: ${e.message}"
                } finally {
                    generating = false
                    progress.visibility = View.GONE
                    generateBtn.isEnabled = true
                }
            }
        }

        loadBtn.setOnClickListener { showModelManager() }

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

    private fun restoreLastModel() {
        val modelName = chatStore.lastModelName() ?: return
        val modelFile = File(ensureModelsDirectory(), modelName)
        if (modelFile.exists()) {
            lifecycleScope.launch { loadModel(modelName, modelFile) }
        }
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

                val isSafetensors = fileName.endsWith(".safetensors", ignoreCase = true)

                val modelFile = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { input ->
                        ensureModelFile(sanitizeModelName(fileName), input)
                    } ?: error("Unable to copy the selected model")
                }

                val detectedType = modelRepository.detectModelType(modelFile)
                when (detectedType) {
                    ModelType.IMAGE_GEN -> loadImageGenModel(modelFile)
                    else -> loadModel(modelFile.name, modelFile)
                }
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
        setModelLoadingUi(getString(R.string.loading_model))
        try {
            withContext(Dispatchers.IO) {
                val inferenceEngine = requireNotNull(engine) { "Inference engine is not ready" }
                runCatching { imageGenEngine?.unload() }
                if (inferenceEngine.state.value.isModelLoaded) {
                    inferenceEngine.cleanUp()
                }
                inferenceEngine.loadModel(modelFile.path)
                inferenceEngine.setSystemPrompt(buildSystemPrompt())
            }
            selectedModelName = modelName
            chatStore.saveLastModelName(modelName)
            isModelReady = true
            refreshUi()
        } catch (exception: Exception) {
            Log.e(TAG, "Unable to load model", exception)
            selectedModelName = null
            isModelReady = false
            chatStore.clearLastModelName()
            refreshUi()
            showToast(getString(R.string.model_load_failed))
        }
    }

    private suspend fun loadImageGenModel(modelFile: File) {
        setModelLoadingUi(getString(R.string.tool_image_gen_loading))
        try {
            withContext(Dispatchers.IO) {
                runCatching { engine?.cleanUp() }
                isModelReady = false
                selectedModelName = null
                chatStore.clearLastModelName()
                imageGenEngine?.loadModel(modelFile.path)
            }
            refreshUi()
        } catch (exception: Exception) {
            Log.e(TAG, "Unable to load image gen model", exception)
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
                var nextPrompt = userMessage
                var toolCallCount = 0
                while (true) {
                    val generatedResponse = StringBuilder()
                    inferenceEngine.sendUserPrompt(nextPrompt).collect { token ->
                        generatedResponse.append(token)
                        producedOutput = true
                        withContext(Dispatchers.Main) {
                            val split = splitResponse(generatedResponse.toString())
                            updateLastAssistantMessage(split.answer, split.thinking, split.tooling)
                            if (split.thinking.isNotBlank()) messageAdapter.expandThinking(messages.last().id)
                            if (split.tooling.isNotBlank()) messageAdapter.expandTooling(messages.last().id)
                        }
                    }

                    val toolCall = parseAnyToolCall(generatedResponse.toString()) ?: break
                    if (toolCallCount >= MAX_TOOL_CALLS_PER_TURN) break

                    val toolResult = withContext(Dispatchers.IO) { executeToolCall(toolCall) }
                    toolCallCount++
                    withContext(Dispatchers.Main) {
                        appendMessage(
                            Message(UUID.randomUUID().toString(), getString(R.string.thinking), false)
                        )
                    }
                    nextPrompt = toolResultPrompt(toolCall, toolResult)
                }
            } catch (exception: CancellationException) {
                throw exception
            } catch (exception: Throwable) {
                failure = exception
                Log.e(TAG, "Generation failed", exception)
            } finally {
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

    private fun parseAnyToolCall(response: String): ToolCall? {
        val sandbox = sandboxTools.parseToolCall(response)
        if (sandbox != null) return ToolCall.Sandbox(sandbox)
        val imgTool = imageGenTool
        if (toolSettings.imageGenEnabled && imgTool != null) {
            val imgGen = imgTool.parseToolCall(response)
            if (imgGen != null) return ToolCall.ImageGen(imgGen)
        }
        return null
    }

    private fun extractAnyToolJson(response: String): String? {
        sandboxTools.extractToolJson(response)?.let { return it }
        val imgTool = imageGenTool
        if (toolSettings.imageGenEnabled && imgTool != null) {
            imgTool.extractImageGenJson(response)?.let { return it }
        }
        return null
    }

    private suspend fun executeToolCall(call: ToolCall): String = when (call) {
        is ToolCall.Sandbox -> sandboxTools.execute(call.call)
        is ToolCall.ImageGen -> imageGenTool?.execute(call.call) ?: "ERROR: Image generation is not available"
    }

    private fun toolResultPrompt(call: ToolCall, result: String): String = when (call) {
        is ToolCall.Sandbox -> sandboxTools.toolResultPrompt(call.call, result)
        is ToolCall.ImageGen -> imageGenTool?.toolResultPrompt(call.call, result)
            ?: "<tool_result>$result</tool_result>"
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
        remoteModelsRv.isNestedScrollingEnabled = true
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
                when (localModel.type) {
                    ModelType.IMAGE_GEN -> lifecycleScope.launch { loadImageGenModel(localModel.file) }
                    else -> lifecycleScope.launch { loadModel(localModel.file.name, localModel.file) }
                }
            },
            onDelete = { modelFile ->
                confirmDeleteModel(modelFile) {
                    modelsAdapter.replaceModels(modelRepository.localModels())
                }
            }
        )
        modelsRv.layoutManager = LinearLayoutManager(this)
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
                lifecycleScope.launch {
                    val wasLoaded = modelFile.name == selectedModelName
                    val deleted = runCatching {
                        withContext(Dispatchers.IO) {
                            if (wasLoaded) {
                                if (engine?.state?.value?.isModelLoaded == true) engine?.cleanUp()
                                if (imageGenEngine?.state?.value?.mode == ImageGenEngine.Mode.READY) imageGenEngine?.unload()
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
    }

    private fun refreshUi() {
        modelStatusTv.text = when {
            !engineReady -> getString(R.string.engine_starting)
            isModelReady -> getString(R.string.model_ready, selectedModelName ?: "")
            else -> getString(R.string.no_model_selected)
        }
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
        speechRecognizer.close()
        unloadAllModels()
        runCatching { imageGenEngine?.destroy() }
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
        runCatching {
            imageGenEngine?.cancelGeneration()
            imageGenEngine?.unload()
        }
        runCatching { engine?.cleanUp() }
        selectedModelName = null
        isModelReady = false
        refreshUi()
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
