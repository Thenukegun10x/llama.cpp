package com.example.llama

import android.content.Context
import org.json.JSONObject
import java.io.File

data class SandboxToolCall(
    val name: String,
    val path: String?,
    val content: String?,
    val oldText: String?,
    val newText: String?
)

class SandboxFileTools(val rootDirectory: File) {
    constructor(context: Context) : this(
        File(context.filesDir, DIRECTORY_NAME).apply { mkdirs() }.canonicalFile
    )

    fun parseToolCall(response: String): SandboxToolCall? {
        val payload = extractToolJson(response)
        if (payload != null) {
            runCatching {
                parseToolCall(JSONObject(payload))?.let { return it }
            }
        }
        val lfmCalls = LfmToolParser.parseToolCalls(response)
        for (lfmJson in lfmCalls) {
            parseToolCall(lfmJson)?.let { return it }
        }
        return null
    }

    fun parseToolCall(json: JSONObject): SandboxToolCall? {
        return runCatching {
            val name = canonicalToolName(json.firstString("name", "tool", "action") ?: return null)
                ?: return null
            SandboxToolCall(
                name = name,
                path = json.firstString("path", "file", "file_path", "filename"),
                content = json.firstString("content", "text", "data", allowEmpty = true),
                oldText = json.firstString("old_text", "oldText", "old", "find", allowEmpty = true),
                newText = json.firstString("new_text", "newText", "new", "replace", allowEmpty = true)
            )
        }.getOrNull()
    }

    fun execute(call: SandboxToolCall): String = runCatching {
        when (call.name) {
            "list_files" -> listFiles().ifEmpty { listOf("(empty)") }.joinToString("\n")
            "read_file" -> readFile(requirePath(call))
            "write_file" -> writeFile(requirePath(call), requireNotNull(call.content) { "content is required" })
            "append_file" -> appendFile(requirePath(call), requireNotNull(call.content) { "content is required" })
            "edit_file" -> editFile(
                requirePath(call),
                requireNotNull(call.oldText) { "old_text is required" },
                requireNotNull(call.newText) { "new_text is required" }
            )
            else -> error("Unknown tool: ${call.name}")
        }
    }.getOrElse { exception -> "ERROR: ${exception.message ?: "Tool failed"}" }

    fun toolResultPrompt(call: SandboxToolCall, result: String): String {
        val json = JSONObject()
            .put("name", call.name)
            .put("path", call.path ?: "")
            .put("result", result.take(MAX_TOOL_RESULT_CHARS))
        return "<tool_result>${json}</tool_result>\n\nPlease use the above tool results to answer the user's request."
    }

    fun listFiles(): List<String> {
        rootDirectory.mkdirs()
        return rootDirectory.walkTopDown()
            .filter { it.isFile }
            .map { it.relativeTo(rootDirectory).invariantSeparatorsPath }
            .take(MAX_LISTED_FILES)
            .sorted()
            .toList()
    }

    fun readFile(relativePath: String): String {
        val file = resolveFile(relativePath)
        require(file.isFile) { "File does not exist: $relativePath" }
        require(file.length() <= MAX_FILE_BYTES) { "File is larger than $MAX_FILE_BYTES bytes" }
        return file.readText(Charsets.UTF_8)
    }

    fun writeFile(relativePath: String, content: String): String {
        require(content.toByteArray(Charsets.UTF_8).size <= MAX_FILE_BYTES) {
            "Content is larger than $MAX_FILE_BYTES bytes"
        }
        val file = resolveFile(relativePath)
        file.parentFile?.mkdirs()
        file.writeText(content, Charsets.UTF_8)
        return "Wrote ${content.length} characters to ${file.relativeTo(rootDirectory).invariantSeparatorsPath}"
    }

    fun appendFile(relativePath: String, content: String): String {
        val file = resolveFile(relativePath)
        val existing = if (file.exists()) readFile(relativePath) else ""
        val separator = if (existing.isNotEmpty() && content.isNotEmpty() && !existing.endsWith('\n')) "\n" else ""
        return writeFile(relativePath, existing + separator + content).replace("Wrote", "Appended")
    }

    fun editFile(relativePath: String, oldText: String, newText: String): String {
        require(oldText.isNotEmpty()) { "old_text cannot be empty" }
        val original = readFile(relativePath)
        val match = findFriendlyEditMatch(original, oldText)
            ?: error("old_text was not found in $relativePath; read the file and use a distinctive excerpt")
        val edited = original.replaceRange(match, newText)
        return writeFile(relativePath, edited).replace("Wrote", "Edited")
    }

    fun deleteFile(relativePath: String): Boolean {
        val file = resolveFile(relativePath)
        return file.isFile && file.delete()
    }

    private fun findFriendlyEditMatch(original: String, requested: String): IntRange? {
        uniqueLiteralMatch(original, requested, ignoreCase = false)?.let { return it }
        uniqueLiteralMatch(original, requested, ignoreCase = true)?.let { return it }

        val words = requested.trim().split(Regex("\\s+")).filter { it.isNotEmpty() }
        if (words.isEmpty()) return null
        val flexiblePattern = words.joinToString("\\s+") { word -> Regex.escape(word) }
        val matches = Regex(flexiblePattern, setOf(RegexOption.IGNORE_CASE))
            .findAll(original)
            .take(2)
            .toList()
        require(matches.size <= 1) { "old_text matches more than once; include more surrounding text" }
        return matches.singleOrNull()?.range
    }

    private fun uniqueLiteralMatch(text: String, query: String, ignoreCase: Boolean): IntRange? {
        val first = text.indexOf(query, ignoreCase = ignoreCase)
        if (first < 0) return null
        val second = text.indexOf(query, startIndex = first + 1, ignoreCase = ignoreCase)
        require(second < 0) { "old_text matches more than once; include more surrounding text" }
        return first until first + query.length
    }

    internal fun extractToolJson(response: String): String? {
        TOOL_TAG_REGEX.find(response)?.groupValues?.get(1)?.let { tagged ->
            balancedJsonObject(tagged)?.let { return it }
        }
        balancedJsonObject(response)?.let { return it }
        val lfmBlock = LfmToolParser.extractToolBlock(response)
        if (lfmBlock != null) {
            val hasSandboxTool = LfmToolParser.parseToolCalls(lfmBlock).any {
                canonicalToolName(it.firstString("name", "tool", "action") ?: "") != null
            }
            if (hasSandboxTool) return lfmBlock
        }
        return null
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
                                    canonicalToolName(json.firstString("name", "tool", "action") ?: "") != null
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

    private fun canonicalToolName(rawName: String): String? {
        val compact = rawName.lowercase().replace(Regex("[^a-z]"), "")
        return when {
            compact.contains("list") -> "list_files"
            compact.contains("read") || compact.contains("open") || compact.contains("getfile") -> "read_file"
            compact.contains("append") || compact.contains("addto") -> "append_file"
            compact.contains("edit") || compact.contains("replace") || compact.contains("patch") ||
                compact.contains("update") -> "edit_file"
            compact.contains("write") || compact.contains("create") || compact.contains("save") -> "write_file"
            else -> null
        }
    }

    private fun JSONObject.firstString(
        vararg keys: String,
        allowEmpty: Boolean = false
    ): String? = keys.firstNotNullOfOrNull { key ->
        if (!has(key) || isNull(key)) return@firstNotNullOfOrNull null
        optString(key).takeIf { value -> allowEmpty || value.isNotBlank() }
    }

    private fun resolveFile(relativePath: String): File {
        require(relativePath.isNotBlank()) { "path cannot be blank" }
        require(!File(relativePath).isAbsolute) { "Absolute paths are not allowed" }
        val target = File(rootDirectory, relativePath).canonicalFile
        val rootPath = rootDirectory.path + File.separator
        require(target.path.startsWith(rootPath)) { "Path escapes the sandbox" }
        require(target != rootDirectory) { "A file path is required" }
        return target
    }

    private fun requirePath(call: SandboxToolCall): String =
        requireNotNull(call.path) { "path is required" }

    companion object {
        private const val DIRECTORY_NAME = "assistant-sandbox"
        private const val MAX_FILE_BYTES = 256 * 1024L
        private const val MAX_LISTED_FILES = 200
        private const val MAX_TOOL_RESULT_CHARS = 32 * 1024
        private val TOOL_TAG_REGEX = Regex(
            """<\s*tool[_ -]?call\s*>(.*?)<\s*/\s*tool[_ -]?call\s*>""",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)
        )
    }
}
