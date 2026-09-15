package com.example.llama

import org.json.JSONArray
import org.json.JSONObject

object LfmToolParser {
    const val TOOL_CALL_START = "<|tool_call_start|>"
    const val TOOL_CALL_END = "<|tool_call_end|>"

    private val TOOL_TAG_REGEX = Regex(
        """<\|\s*tool_call_start\s*\|>(.*?)<\|\s*tool_call_end\s*\|>""",
        setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)
    )

    private val FUNCTION_CALL_REGEX = Regex(
        """([A-Za-z_][A-Za-z0-9_.]*)\s*\(""",
        RegexOption.IGNORE_CASE
    )

    /**
     * Extracts the raw tool block string from [text] (including tags).
     * Handles both closed `<|tool_call_start|>...<|tool_call_end|>` and
     * streaming/unclosed `<|tool_call_start|>...`.
     */
    fun extractToolBlock(text: String): String? {
        val start = text.indexOf(TOOL_CALL_START)
        if (start >= 0) {
            val end = text.indexOf(TOOL_CALL_END, startIndex = start + TOOL_CALL_START.length)
            return if (end >= 0) {
                text.substring(start, end + TOOL_CALL_END.length)
            } else {
                text.substring(start)
            }
        }
        // Also check regex in case of slight spacing variations e.g. <| tool_call_start |>
        TOOL_TAG_REGEX.find(text)?.let { return it.value }
        return null
    }

    /**
     * Extracts the first parsed tool call as a [JSONObject], or null if none found.
     */
    fun parseFirst(text: String): JSONObject? {
        return parseToolCalls(text).firstOrNull()
    }

    /**
     * Parses all tool calls found in [text].
     * Returns a list of [JSONObject] with "name" and all parsed arguments.
     */
    fun parseToolCalls(text: String): List<JSONObject> {
        val innerContent = extractInnerContent(text) ?: return emptyList()
        val trimmed = innerContent.trim()
        if (trimmed.isEmpty()) return emptyList()

        // Strip outer brackets if present: [func1(...), func2(...)]
        val callsContent = if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
            trimmed.substring(1, trimmed.length - 1).trim()
        } else {
            trimmed
        }

        if (callsContent.isEmpty()) return emptyList()

        val callSnippets = splitTopLevelCalls(callsContent)
        return callSnippets.mapNotNull { parseSingleCall(it) }
    }

    private fun extractInnerContent(text: String): String? {
        // 1. Try explicit start and end tags
        val startIdx = text.indexOf(TOOL_CALL_START)
        if (startIdx >= 0) {
            val afterStart = startIdx + TOOL_CALL_START.length
            val endIdx = text.indexOf(TOOL_CALL_END, startIndex = afterStart)
            return if (endIdx >= 0) {
                text.substring(afterStart, endIdx)
            } else {
                text.substring(afterStart)
            }
        }

        // 2. Try regex match for start/end
        TOOL_TAG_REGEX.find(text)?.groupValues?.get(1)?.let { return it }

        // 3. Fallback: bracketed function call e.g. [web_search(...)] or standalone func(...)
        val trimmed = text.trim()
        if (trimmed.startsWith("[") && trimmed.endsWith("]") && trimmed.contains("(")) {
            return trimmed
        }
        if (FUNCTION_CALL_REGEX.containsMatchIn(trimmed)) {
            return trimmed
        }

        return null
    }

    private fun splitTopLevelCalls(text: String): List<String> {
        val calls = mutableListOf<String>()
        var inSingleQuote = false
        var inDoubleQuote = false
        var escaped = false
        var parenDepth = 0
        var braceDepth = 0
        var bracketDepth = 0
        var sliceStart = 0

        for (i in text.indices) {
            val c = text[i]
            if (escaped) {
                escaped = false
                continue
            }
            if (c == '\\') {
                escaped = true
                continue
            }

            if (inSingleQuote) {
                if (c == '\'') inSingleQuote = false
                continue
            }
            if (inDoubleQuote) {
                if (c == '"') inDoubleQuote = false
                continue
            }

            when (c) {
                '\'' -> inSingleQuote = true
                '"' -> inDoubleQuote = true
                '(' -> parenDepth++
                ')' -> if (parenDepth > 0) parenDepth--
                '{' -> braceDepth++
                '}' -> if (braceDepth > 0) braceDepth--
                '[' -> bracketDepth++
                ']' -> if (bracketDepth > 0) bracketDepth--
                ',' -> {
                    if (parenDepth == 0 && braceDepth == 0 && bracketDepth == 0) {
                        val slice = text.substring(sliceStart, i).trim()
                        if (slice.isNotEmpty()) calls.add(slice)
                        sliceStart = i + 1
                    }
                }
            }
        }

        val remaining = text.substring(sliceStart).trim()
        if (remaining.isNotEmpty()) calls.add(remaining)
        return calls
    }

    private fun parseSingleCall(callText: String): JSONObject? {
        val trimmed = callText.trim().removePrefix("[").removeSuffix("]").trim()
        val openParen = trimmed.indexOf('(')
        if (openParen <= 0) return null

        val funcName = trimmed.substring(0, openParen).trim()
        if (!funcName.matches(Regex("""^[A-Za-z_][A-Za-z0-9_.]*$"""))) {
            return null
        }

        val closeParen = findMatchingCloseParen(trimmed, openParen)
        if (closeParen < 0) return null

        val argsContent = trimmed.substring(openParen + 1, closeParen).trim()
        val argTokens = splitArguments(argsContent)

        val json = JSONObject()
        json.put("name", funcName)
        json.put("tool", funcName)
        json.put("action", funcName)

        for ((index, token) in argTokens.withIndex()) {
            val eqIdx = findTopLevelEquals(token)
            val (key, valueStr) = if (eqIdx >= 0) {
                token.substring(0, eqIdx).trim() to token.substring(eqIdx + 1).trim()
            } else {
                when (index) {
                    0 -> "query"
                    else -> "arg$index"
                } to token.trim()
            }

            if (key.isBlank()) continue
            val parsedValue = parseValue(valueStr)
            json.put(key, parsedValue)
        }

        return json
    }

    private fun findMatchingCloseParen(text: String, openIdx: Int): Int {
        var depth = 0
        var inSingleQuote = false
        var inDoubleQuote = false
        var escaped = false

        for (i in openIdx until text.length) {
            val c = text[i]
            if (escaped) {
                escaped = false
                continue
            }
            if (c == '\\') {
                escaped = true
                continue
            }
            if (inSingleQuote) {
                if (c == '\'') inSingleQuote = false
                continue
            }
            if (inDoubleQuote) {
                if (c == '"') inDoubleQuote = false
                continue
            }

            when (c) {
                '\'' -> inSingleQuote = true
                '"' -> inDoubleQuote = true
                '(' -> depth++
                ')' -> {
                    depth--
                    if (depth == 0) return i
                }
            }
        }
        return -1
    }

    private fun splitArguments(text: String): List<String> {
        if (text.isBlank()) return emptyList()
        val tokens = mutableListOf<String>()
        var inSingleQuote = false
        var inDoubleQuote = false
        var escaped = false
        var parenDepth = 0
        var braceDepth = 0
        var bracketDepth = 0
        var start = 0

        for (i in text.indices) {
            val c = text[i]
            if (escaped) {
                escaped = false
                continue
            }
            if (c == '\\') {
                escaped = true
                continue
            }
            if (inSingleQuote) {
                if (c == '\'') inSingleQuote = false
                continue
            }
            if (inDoubleQuote) {
                if (c == '"') inDoubleQuote = false
                continue
            }

            when (c) {
                '\'' -> inSingleQuote = true
                '"' -> inDoubleQuote = true
                '(' -> parenDepth++
                ')' -> if (parenDepth > 0) parenDepth--
                '{' -> braceDepth++
                '}' -> if (braceDepth > 0) braceDepth--
                '[' -> bracketDepth++
                ']' -> if (bracketDepth > 0) bracketDepth--
                ',' -> {
                    if (parenDepth == 0 && braceDepth == 0 && bracketDepth == 0) {
                        val token = text.substring(start, i).trim()
                        if (token.isNotEmpty()) tokens.add(token)
                        start = i + 1
                    }
                }
            }
        }

        val remaining = text.substring(start).trim()
        if (remaining.isNotEmpty()) tokens.add(remaining)
        return tokens
    }

    private fun findTopLevelEquals(token: String): Int {
        var inSingleQuote = false
        var inDoubleQuote = false
        var escaped = false
        var parenDepth = 0
        var braceDepth = 0
        var bracketDepth = 0

        for (i in token.indices) {
            val c = token[i]
            if (escaped) {
                escaped = false
                continue
            }
            if (c == '\\') {
                escaped = true
                continue
            }
            if (inSingleQuote) {
                if (c == '\'') inSingleQuote = false
                continue
            }
            if (inDoubleQuote) {
                if (c == '"') inDoubleQuote = false
                continue
            }

            when (c) {
                '\'' -> inSingleQuote = true
                '"' -> inDoubleQuote = true
                '(' -> parenDepth++
                ')' -> if (parenDepth > 0) parenDepth--
                '{' -> braceDepth++
                '}' -> if (braceDepth > 0) braceDepth--
                '[' -> bracketDepth++
                ']' -> if (bracketDepth > 0) bracketDepth--
                '=' -> {
                    if (parenDepth == 0 && braceDepth == 0 && bracketDepth == 0) {
                        return i
                    }
                }
            }
        }
        return -1
    }

    private fun parseValue(raw: String): Any {
        val trimmed = raw.trim()
        if (trimmed.isEmpty()) return ""

        // Single quoted string
        if (trimmed.startsWith("'") && trimmed.endsWith("'") && trimmed.length >= 2) {
            return unescapeString(trimmed.substring(1, trimmed.length - 1))
        }

        // Double quoted string
        if (trimmed.startsWith("\"") && trimmed.endsWith("\"") && trimmed.length >= 2) {
            return unescapeString(trimmed.substring(1, trimmed.length - 1))
        }

        // Booleans
        if (trimmed.equals("true", ignoreCase = true)) return true
        if (trimmed.equals("false", ignoreCase = true)) return false

        // Null / None
        if (trimmed.equals("none", ignoreCase = true) || trimmed.equals("null", ignoreCase = true)) {
            return JSONObject.NULL
        }

        // Numbers: Integer / Long
        if (trimmed.matches(Regex("""^-?\d+$"""))) {
            return trimmed.toIntOrNull() ?: trimmed.toLongOrNull() ?: trimmed
        }

        // Numbers: Double
        if (trimmed.matches(Regex("""^-?\d+\.\d+(?:[eE][+-]?\d+)?$"""))) {
            return trimmed.toDoubleOrNull() ?: trimmed
        }

        // JSON Object / Python Dict
        if (trimmed.startsWith("{") && trimmed.endsWith("}")) {
            runCatching {
                return JSONObject(trimmed)
            }
            val normalized = normalizePythonJson(trimmed)
            runCatching {
                return JSONObject(normalized)
            }
        }

        // JSON Array / Python List
        if (trimmed.startsWith("[") && trimmed.endsWith("]")) {
            runCatching {
                return JSONArray(trimmed)
            }
            val normalized = normalizePythonJson(trimmed)
            runCatching {
                return JSONArray(normalized)
            }
        }

        return trimmed
    }

    private fun normalizePythonJson(text: String): String {
        return text.replace(Regex("""\bTrue\b"""), "true")
            .replace(Regex("""\bFalse\b"""), "false")
            .replace(Regex("""\bNone\b"""), "null")
    }

    fun unescapeString(raw: String): String {
        val sb = StringBuilder()
        var i = 0
        while (i < raw.length) {
            val c = raw[i]
            if (c == '\\' && i + 1 < raw.length) {
                val next = raw[i + 1]
                when (next) {
                    'n' -> sb.append('\n')
                    'r' -> sb.append('\r')
                    't' -> sb.append('\t')
                    'b' -> sb.append('\b')
                    'f' -> sb.append('\u000c')
                    '\'', '"', '\\' -> sb.append(next)
                    'u' -> {
                        if (i + 5 < raw.length) {
                            val hex = raw.substring(i + 2, i + 6)
                            val code = hex.toIntOrNull(16)
                            if (code != null) {
                                sb.append(code.toChar())
                                i += 6
                                continue
                            }
                        }
                        sb.append(next)
                    }
                    else -> sb.append(next)
                }
                i += 2
            } else {
                sb.append(c)
                i++
            }
        }
        return sb.toString()
    }
}
