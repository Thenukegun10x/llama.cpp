package com.example.llama

import org.jsoup.Jsoup
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Assert.fail
import org.junit.Test
import java.net.URL

class WebSearchToolTest {
    private val tool = WebSearchTool()

    private val sampleHtml = """
        <html><body>
        <div class="result">
          <h2 class="result__title"><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=aaa">First Result</a></h2>
          <a class="result__snippet" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&amp;rut=aaa">First snippet text.</a>
        </div>
        <div class="result">
          <h2 class="result__title"><a rel="nofollow" class="result__a" href="https://example.org/b">Second Result</a></h2>
        </div>
        <div class="result"><span class="no-link">sponsored</span></div>
        </body></html>
    """.trimIndent()

    @Test
    fun parseResultsUnwrapsRedirectsAndSkipsAds() {
        val doc = Jsoup.parse(sampleHtml, "https://html.duckduckgo.com/html/")
        val results = tool.parseResults(doc)
        assertEquals(2, results.size)
        assertEquals("First Result", results[0].title)
        assertEquals("https://example.com/a", results[0].url)
        assertEquals("First snippet text.", results[0].snippet)
        assertEquals("Second Result", results[1].title)
        assertEquals("https://example.org/b", results[1].url)
        assertEquals("", results[1].snippet)
    }

    @Test
    fun unwrapResultUrlFiltersSchemes() {
        assertEquals(
            "https://example.com/a",
            tool.unwrapResultUrl("https://duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa&rut=1")
        )
        assertEquals("https://example.org/b", tool.unwrapResultUrl("https://example.org/b"))
        assertNull(tool.unwrapResultUrl("javascript:void(0)"))
        assertNull(tool.unwrapResultUrl(""))
        assertNull(tool.unwrapResultUrl("https://duckduckgo.com/l/?rut=1"))
    }

    @Test
    fun canonicalToolNameMatchesWebActions() {
        assertEquals(WebSearchTool.ACTION_SEARCH, tool.canonicalToolName("web_search"))
        assertEquals(WebSearchTool.ACTION_SEARCH, tool.canonicalToolName("Web Search"))
        assertEquals(WebSearchTool.ACTION_FETCH, tool.canonicalToolName("fetch page"))
        assertEquals(WebSearchTool.ACTION_SEARCH, tool.canonicalToolName("google"))
        assertNull(tool.canonicalToolName("read_file"))
    }

    @Test
    fun parseToolCallClampsLimitAndOffset() {
        val call = tool.parseToolCall(
            """<tool_call>{"name":"web_search","query":"kotlin","limit":99,"offset":-3}</tool_call>"""
        )
        assertNotNull(call)
        assertEquals(WebSearchTool.ACTION_SEARCH, call!!.action)
        assertEquals(10, call.limit)
        assertEquals(0, call.offset)
    }

    @Test
    fun parseToolCallRejectsMissingFields() {
        assertNull(tool.parseToolCall("""{"name":"web_search"}"""))
        assertNull(tool.parseToolCall("""{"name":"web_fetch"}"""))
        assertNull(tool.parseToolCall("just chatting, no tool here"))
    }

    @Test
    fun parseToolCallSupportsLfmSearchCall() {
        val call = tool.parseToolCall(
            "<|tool_call_start|>[web_search(query='web search tool demonstration', limit=5)]<|tool_call_end|>"
        )
        assertNotNull(call)
        assertEquals(WebSearchTool.ACTION_SEARCH, call!!.action)
        assertEquals("web search tool demonstration", call.query)
        assertEquals(5, call.limit)
    }

    @Test
    fun parseToolCallSupportsLfmFetchCall() {
        val call = tool.parseToolCall(
            """<|tool_call_start|>[web_fetch(url="https://example.com/test", max_chars=1500)]<|tool_call_end|>"""
        )
        assertNotNull(call)
        assertEquals(WebSearchTool.ACTION_FETCH, call!!.action)
        assertEquals("https://example.com/test", call.url)
        assertEquals(1500, call.maxChars)
    }

    @Test
    fun extractArticleTextDropsBoilerplate() {
        val doc = Jsoup.parse(
            "<html><body><nav>menu</nav><script>var x = 1;</script>" +
                "<article><h1>Headline</h1><p>Body text here.</p></article>" +
                "<footer>copy</footer></body></html>"
        )
        assertEquals("Headline Body text here.", tool.extractArticleText(doc))
    }

    @Test
    fun requirePublicHttpUrlBlocksLocalTargets() {
        try {
            tool.requirePublicHttpUrl(URL("http://127.0.0.1/"))
            fail("loopback should be rejected")
        } catch (expected: IllegalArgumentException) {
        }
        try {
            tool.requirePublicHttpUrl(URL("http://localhost/"))
            fail("localhost should be rejected")
        } catch (expected: IllegalArgumentException) {
        }
        try {
            tool.requirePublicHttpUrl(URL("file:///etc/passwd"))
            fail("file scheme should be rejected")
        } catch (expected: IllegalArgumentException) {
        }
        // IP literal needs no DNS and is publicly routable.
        tool.requirePublicHttpUrl(URL("https://93.184.216.0/"))
    }

    @Test
    fun normalizeHttpUrlAddsSchemeAndRejectsOthers() {
        assertEquals("https://example.com", tool.normalizeHttpUrl("example.com"))
        assertEquals("http://example.com/x", tool.normalizeHttpUrl("http://example.com/x"))
        try {
            tool.normalizeHttpUrl("ftp://example.com/x")
            fail("ftp should be rejected")
        } catch (expected: IllegalArgumentException) {
        }
    }

    @Test
    fun toolResultPromptWrapsJson() {
        val call = WebSearchCall(WebSearchTool.ACTION_SEARCH, "kotlin", null, 5, 0, 6000)
        val prompt = tool.toolResultPrompt(call, "some results")
        assertTrue(prompt.startsWith("<tool_result>"))
        assertTrue(prompt.contains("</tool_result>"))
        assertTrue(prompt.contains("web_search"))
    }
}
