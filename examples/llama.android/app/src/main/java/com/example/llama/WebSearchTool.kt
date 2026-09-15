package com.example.llama

import org.json.JSONObject
import org.jsoup.Jsoup
import org.jsoup.nodes.Document
import java.net.InetAddress
import java.net.URL
import java.net.URLDecoder

data class WebSearchCall(
    val action: String,
    val query: String,
    val url: String?,
    val limit: Int,
    val offset: Int,
    val maxChars: Int
)

data class WebResult(
    val title: String,
    val url: String,
    val snippet: String
)

class WebSearchTool {
    fun parseToolCall(response: String): WebSearchCall? {
        val payload = extractWebJson(response)
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

    fun parseToolCall(json: JSONObject): WebSearchCall? {
        return runCatching {
            val action = canonicalToolName(json.firstString("name", "tool", "action") ?: return null)
                ?: return null
            WebSearchCall(
                action = action,
                query = json.firstString("query", "q", "prompt") ?: "",
                url = json.firstString("url", "link", "href"),
                limit = json.optInt("limit", DEFAULT_LIMIT).coerceIn(1, MAX_LIMIT),
                offset = json.optInt("offset", 0).coerceAtLeast(0),
                maxChars = json.optInt("max_chars", DEFAULT_MAX_CHARS).coerceIn(MIN_MAX_CHARS, MAX_MAX_CHARS)
            ).also {
                if (action == ACTION_SEARCH) require(it.query.isNotBlank()) { "query is required" }
                else require(!it.url.isNullOrBlank()) { "url is required" }
            }
        }.getOrNull()
    }

    // Blocking network IO; callers already run this off the main thread.
    fun execute(call: WebSearchCall): String = runCatching {
        when (call.action) {
            ACTION_SEARCH -> {
                val results = search(call.query, call.limit, call.offset)
                if (results.isEmpty()) "No results found."
                else results.mapIndexed { index, result ->
                    "${call.offset + index + 1}. ${result.title}\n${result.url}\n${result.snippet}"
                }.joinToString("\n\n")
            }
            else -> fetchPageText(
                URL(normalizeHttpUrl(requireNotNull(call.url) { "url is required" })),
                call.maxChars
            ).ifBlank { "Page had no readable text." }
        }
    }.getOrElse { exception -> "ERROR: ${exception.message ?: "Web tool failed"}" }

    fun toolResultPrompt(call: WebSearchCall, result: String): String {
        val json = JSONObject()
            .put("name", call.action)
            .put("query", if (call.action == ACTION_SEARCH) call.query else (call.url ?: ""))
            .put("result", result.take(MAX_TOOL_RESULT_CHARS))
        return "<tool_result>${json}</tool_result>\n\nPlease use the above search results to answer the user's request."
    }

    internal fun extractWebJson(response: String): String? {
        TOOL_TAG_REGEX.find(response)?.groupValues?.get(1)?.let { tagged ->
            balancedJsonObject(tagged)?.let { return it }
        }
        balancedJsonObject(response)?.let { return it }
        val lfmBlock = LfmToolParser.extractToolBlock(response)
        if (lfmBlock != null) {
            val hasWebTool = LfmToolParser.parseToolCalls(lfmBlock).any {
                canonicalToolName(it.firstString("name", "tool", "action") ?: "") != null
            }
            if (hasWebTool) return lfmBlock
        }
        return null
    }

    // Pages through DDG until limit is filled, offset is honored, or pages end.
    internal fun search(query: String, limit: Int, offset: Int): List<WebResult> {
        require(query.isNotBlank()) { "query is blank" }
        val out = mutableListOf<WebResult>()
        var pageStart = (offset / RESULTS_PER_PAGE) * RESULTS_PER_PAGE
        var skip = offset - pageStart
        repeat(MAX_PAGES) {
            val page = fetchResultPage(query, pageStart)
            if (page.isEmpty()) return out
            for (result in page.drop(skip)) {
                if (out.size >= limit) return out
                out += result
            }
            skip = 0
            if (page.size < RESULTS_PER_PAGE) return out
            pageStart += RESULTS_PER_PAGE
        }
        return out
    }

    internal fun fetchResultPage(query: String, start: Int): List<WebResult> {
        val doc = Jsoup.connect(SEARCH_URL)
            .data("q", query, "s", start.toString())
            .userAgent(USER_AGENT)
            .timeout(HTTP_TIMEOUT_MS)
            .maxBodySize(MAX_BODY_BYTES)
            .post()
        return parseResults(doc)
    }

    // Pure: DDG html into results; unwraps /l/?uddg= redirect links and skips ads.
    internal fun parseResults(doc: Document): List<WebResult> =
        doc.select(".result").mapNotNull { element ->
            if (element.hasClass("result--ad") ||
                element.hasClass("result--sponsored") ||
                element.selectFirst(".badge--ad") != null
            ) {
                return@mapNotNull null
            }
            val link = element.selectFirst("a.result__a") ?: return@mapNotNull null
            val rawHref = link.attr("abs:href").ifBlank { link.attr("href") }
            if (rawHref.contains("duckduckgo.com/y.js") || rawHref.contains("ad_provider=")) {
                return@mapNotNull null
            }
            val url = unwrapResultUrl(rawHref) ?: return@mapNotNull null
            val title = link.text().trim()
            if (title.isEmpty()) return@mapNotNull null
            WebResult(title, url, element.selectFirst(".result__snippet")?.text()?.trim().orEmpty())
        }

    internal fun unwrapResultUrl(href: String): String? {
        if (href.isBlank()) return null
        return runCatching {
            val url = URL(href)
            if (url.host.endsWith("duckduckgo.com")) {
                if (url.path == "/l/") {
                    url.query?.split("&")
                        ?.firstOrNull { it.startsWith("uddg=") }
                        ?.substringAfter("=")
                        ?.let { URLDecoder.decode(it, "UTF-8") }
                        ?.takeIf { it.startsWith("http://") || it.startsWith("https://") }
                } else {
                    null
                }
            } else if (href.startsWith("http://") || href.startsWith("https://")) {
                if (href.contains("bing.com/aclick") || href.contains("ad_provider=")) null
                else href
            } else null
        }.getOrNull()
    }

    internal fun fetchPageText(pageUrl: URL, maxChars: Int): String {
        requirePublicHttpUrl(pageUrl)
        val doc = Jsoup.connect(pageUrl.toString())
            .userAgent(USER_AGENT)
            .timeout(HTTP_TIMEOUT_MS)
            .maxBodySize(MAX_BODY_BYTES)
            .followRedirects(true)
            .get()
        return extractArticleText(doc).take(maxChars)
    }

    // Pure: drop boilerplate, prefer article/main, squash whitespace.
    internal fun extractArticleText(doc: Document): String {
        doc.select("script, style, noscript, header, footer, nav, aside, form, iframe").remove()
        val scope = doc.selectFirst("article, main, [role=main]") ?: doc.body() ?: return ""
        return scope.text().replace(Regex("\\s+"), " ").trim()
    }

    internal fun normalizeHttpUrl(raw: String): String {
        val trimmed = raw.trim()
        require(trimmed.isNotEmpty()) { "url is blank" }
        val withScheme = if (trimmed.contains("://")) trimmed else "https://$trimmed"
        val url = runCatching { URL(withScheme) }.getOrNull()
        require(url != null && (url.protocol == "http" || url.protocol == "https")) {
            "Only http(s) URLs are allowed"
        }
        return url.toString()
    }

    // Fail closed on non-routable targets before any socket opens.
    internal fun requirePublicHttpUrl(url: URL) {
        require(url.protocol == "http" || url.protocol == "https") { "Only http(s) URLs are allowed" }
        val host = url.host.lowercase()
        require(host != "localhost") { "Host is not allowed" }
        val address = runCatching { InetAddress.getByName(host) }.getOrNull() ?: return
        require(!address.isLoopbackAddress && !address.isLinkLocalAddress &&
            !address.isSiteLocalAddress && !address.isMulticastAddress &&
            !address.isAnyLocalAddress) { "Host is not allowed" }
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

    internal fun canonicalToolName(rawName: String): String? {
        val compact = rawName.lowercase().replace(Regex("[^a-z]"), "")
        return when {
            compact.contains("fetch") || compact.contains("openurl") || compact.contains("readpage") -> ACTION_FETCH
            compact.contains("search") || compact.contains("web") || compact.contains("query") ||
                compact.contains("google") || compact.contains("lookup") -> ACTION_SEARCH
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

    companion object {
        const val ACTION_SEARCH = "web_search"
        const val ACTION_FETCH = "web_fetch"
        private const val SEARCH_URL = "https://html.duckduckgo.com/html/"
        private const val USER_AGENT =
            "Mozilla/5.0 (Linux; Android 14) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
        private const val HTTP_TIMEOUT_MS = 20_000
        private const val MAX_BODY_BYTES = 2 * 1024 * 1024
        private const val RESULTS_PER_PAGE = 30
        private const val MAX_PAGES = 3
        private const val DEFAULT_LIMIT = 5
        private const val MAX_LIMIT = 10
        private const val DEFAULT_MAX_CHARS = 6_000
        private const val MIN_MAX_CHARS = 500
        private const val MAX_MAX_CHARS = 20_000
        private const val MAX_TOOL_RESULT_CHARS = 8 * 1024
        private val TOOL_TAG_REGEX = Regex(
            """<\s*tool[_ -]?call\s*>(.*?)<\s*/\s*tool[_ -]?call\s*>""",
            setOf(RegexOption.IGNORE_CASE, RegexOption.DOT_MATCHES_ALL)
        )
    }
}
