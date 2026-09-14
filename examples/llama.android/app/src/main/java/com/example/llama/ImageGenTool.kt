package com.example.llama

import org.json.JSONObject
import java.io.File

data class ImageGenCall(
    val prompt: String,
    val negativePrompt: String,
    val width: Int,
    val height: Int,
    val steps: Int,
    val cfgScale: Float,
    val seed: Long
)

class ImageGenTool(private val engine: com.arm.aichat.ImageGenEngine, outputDir: File) {    fun parseToolCall(response: String): ImageGenCall? {
        val payload = extractImageGenJson(response) ?: return null
        return runCatching {
            val json = JSONObject(payload)
            val name = json.firstString("name", "tool", "action") ?: return null
            if (!isImageGenCall(name)) return null
            val prompt = requireNotNull(json.firstString("prompt")) { "prompt is required" }
            ImageGenCall(
                prompt = prompt,
                negativePrompt = json.firstString("negative_prompt", "negative", allowEmpty = true) ?: "",
                width = json.optInt("width", 512).coerceIn(256, 2048),
                height = json.optInt("height", 512).coerceIn(256, 2048),
                steps = json.optInt("steps", 20).coerceIn(1, 50),
                cfgScale = json.optDouble("cfg_scale", 7.0).toFloat().coerceIn(1.0f, 30.0f),
                seed = json.optLong("seed", -1L)
            )
        }.getOrNull()
    }

    suspend fun execute(call: ImageGenCall): String = runCatching {
        val resultPath = engine.generateImage(
            prompt = call.prompt,
            negativePrompt = call.negativePrompt,
            width = call.width,
            height = call.height,
            steps = call.steps,
            cfgScale = call.cfgScale,
            seed = call.seed
        )
        if (resultPath != null) {
            val file = File(resultPath)
            "Image generated successfully. File: ${file.name} (${call.width}x${call.height}, " +
                "${file.length() / 1024}KB), prompt: \"${call.prompt.take(200)}\""
        } else {
            "ERROR: Image generation failed. The model may not support the requested size or parameters."
        }
    }.getOrElse { exception -> "ERROR: ${exception.message ?: "Image generation failed"}" }

    fun toolResultPrompt(call: ImageGenCall, result: String): String {
        val json = JSONObject()
            .put("name", "generate_image")
            .put("prompt", call.prompt)
            .put("result", result.take(MAX_TOOL_RESULT_CHARS))
        return "<tool_result>${json}</tool_result>"
    }

    internal fun extractImageGenJson(response: String): String? {
        TOOL_TAG_REGEX.find(response)?.groupValues?.get(1)?.let { tagged ->
            balancedJsonObject(tagged)?.let { return it }
        }
        return balancedJsonObject(response)
    }

    private fun balancedJsonObject(text: String): String? {
        var searchFrom = 0
        while (searchFrom < text.length) {
            val start = text.indexOf('{', searchFrom)
            if (start < 0) return null
            var depth = 0
            var inString = false
            var escaped = false
            for (index in start until text.length) {
                val character = text[index]
                if (inString) {
                    when {
                        escaped -> escaped = false
                        character == '\\' -> escaped = true
                        character == '"' -> inString = false
                    }
                } else {
                    when (character) {
                        '"' -> inString = true
                        '{' -> depth++
                        '}' -> {
                            depth--
                            if (depth == 0) {
                                val candidate = text.substring(start, index + 1)
                                val looksLikeTool = runCatching {
                                    val json = JSONObject(candidate)
                                    isImageGenCall(json.firstString("name", "tool", "action") ?: "")
                                }.getOrDefault(false)
                                if (looksLikeTool) return candidate
                                break
                            }
                        }
                    }
                }
            }
            searchFrom = start + 1
        }
        return null
    }

    private fun isImageGenCall(rawName: String): Boolean {
        val compact = rawName.lowercase().replace(Regex("[^a-z]"), "")
        return compact.contains("generateimage") || compact.contains("imagegen") ||
            compact.contains("txt2img") || compact.contains("text2image") ||
            compact.contains("texttoimage") || compact.contains("drawimage") ||
            compact.contains("imagine") || compact.contains("diffusion") ||
            compact == "draw"
    }

    private fun JSONObject.firstString(
        vararg keys: String,
        allowEmpty: Boolean = false
    ): String? = keys.firstNotNullOfOrNull { key ->
        if (!has(key) || isNull(key)) return@firstNotNullOfOrNull null
        optString(key).takeIf { value -> allowEmpty || value.isNotBlank() }
    }

    companion object {
        private const val MAX_TOOL_RESULT_CHARS = 4 * 1024
        private val TOOL_TAG_REGEX = Regex(
            """<\s*tool[_ -]?call\s*>(.*?)<\s*/\s*tool[_ -]?call\s*>""",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)
        )

        val IMAGE_GEN_SYSTEM_PROMPT = """
            IMAGE GENERATION TOOL (generate_image):
            generate_image: {"name":"generate_image","prompt":"a detailed description of the image","negative_prompt":"things to avoid","width":512,"height":512,"steps":20,"cfg_scale":7.0,"seed":-1}

            Parameters (all optional except prompt):
            - prompt: detailed description of the image to generate (required)
            - negative_prompt: things to avoid in the image (default: empty)
            - width: image width in pixels, 256-2048 (default: 512)
            - height: image height in pixels, 256-2048 (default: 512)
            - steps: sampling steps, 1-50, higher = more detail but slower (default: 20)
            - cfg_scale: prompt adherence, 1.0-30.0, higher = follows prompt more strictly (default: 7.0)
            - seed: random seed, -1 for random (default: -1)

            When using a tool, output only one call in this exact wrapper:
            <tool_call>{"name":"generate_image","prompt":"describe the image"}</tool_call>

            Use one tool at a time. Wait for <tool_result> before reporting success.
            Never invent a tool result.
        """.trimIndent()
    }
}
