package com.example.llama

import org.json.JSONArray
import org.json.JSONObject

object ToolDefinitions {

    fun buildToolsJson(settings: ToolSettings): String {
        if (!settings.allToolsEnabled) return "[]"
        val tools = JSONArray()

        if (settings.sandboxToolsEnabled) {
            tools.put(createTool(
                name = "list_files",
                description = "List all files currently stored in the sandbox filesystem.",
                properties = emptyMap(),
                required = emptyList()
            ))

            tools.put(createTool(
                name = "read_file",
                description = "Read the text content of a file in the sandbox filesystem.",
                properties = mapOf(
                    "path" to prop("string", "Relative path of the file to read")
                ),
                required = listOf("path")
            ))

            tools.put(createTool(
                name = "write_file",
                description = "Create a new file or overwrite an existing file in the sandbox filesystem.",
                properties = mapOf(
                    "path" to prop("string", "Relative path of the file to write"),
                    "content" to prop("string", "Text content to write into the file")
                ),
                required = listOf("path", "content")
            ))

            tools.put(createTool(
                name = "append_file",
                description = "Append text content to the end of an existing file in the sandbox filesystem.",
                properties = mapOf(
                    "path" to prop("string", "Relative path of the file to append to"),
                    "content" to prop("string", "Text content to append")
                ),
                required = listOf("path", "content")
            ))

            tools.put(createTool(
                name = "edit_file",
                description = "Replace a unique excerpt in a file with new text.",
                properties = mapOf(
                    "path" to prop("string", "Relative path of the file to edit"),
                    "old_text" to prop("string", "Unique excerpt of existing text to find and replace"),
                    "new_text" to prop("string", "New replacement text")
                ),
                required = listOf("path", "old_text", "new_text")
            ))
        }

        if (settings.webToolsEnabled) {
            tools.put(createTool(
                name = "web_search",
                description = "Search the live web using DuckDuckGo and return titled snippets with URLs.",
                properties = mapOf(
                    "query" to prop("string", "Search terms to look up on the web"),
                    "limit" to prop("integer", "Maximum number of results to return (1-10, default 5)"),
                    "offset" to prop("integer", "Pagination offset to skip results (default 0)")
                ),
                required = listOf("query")
            ))

            tools.put(createTool(
                name = "web_fetch",
                description = "Fetch a webpage by URL and return its text content cleaned to Markdown.",
                properties = mapOf(
                    "url" to prop("string", "The HTTP or HTTPS URL of the web page to fetch"),
                    "max_chars" to prop("integer", "Maximum text characters to return (500-20000, default 6000)")
                ),
                required = listOf("url")
            ))
        }

        return tools.toString()
    }

    private fun prop(type: String, description: String): JSONObject =
        JSONObject().put("type", type).put("description", description)

    private fun createTool(
        name: String,
        description: String,
        properties: Map<String, JSONObject>,
        required: List<String>
    ): JSONObject {
        val propsObj = JSONObject()
        for ((k, v) in properties) {
            propsObj.put(k, v)
        }
        val params = JSONObject()
            .put("type", "object")
            .put("properties", propsObj)
            .put("required", JSONArray(required))

        val func = JSONObject()
            .put("name", name)
            .put("description", description)
            .put("parameters", params)

        return JSONObject()
            .put("type", "function")
            .put("function", func)
    }
}
