package com.example.llama

import org.json.JSONObject
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test

class LfmToolParserTest {

    @Test
    fun testParseLfmSearchCall() {
        val raw = "<|tool_call_start|>[web_search(query='web search tool demonstration', limit=5)]<|tool_call_end|>"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("web_search", call!!.getString("name"))
        assertEquals("web search tool demonstration", call.getString("query"))
        assertEquals(5, call.getInt("limit"))
    }

    @Test
    fun testParseWithDoubleQuotesAndEscapes() {
        val raw = """<|tool_call_start|>[web_search(query="hello \"world\"", limit=10)]<|tool_call_end|>"""
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("web_search", call!!.getString("name"))
        assertEquals("hello \"world\"", call.getString("query"))
        assertEquals(10, call.getInt("limit"))
    }

    @Test
    fun testParseSingleQuoteEscapes() {
        val raw = """<|tool_call_start|>[web_search(query='don\'t stop\nnew line', limit=3)]<|tool_call_end|>"""
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("don't stop\nnew line", call!!.getString("query"))
        assertEquals(3, call.getInt("limit"))
    }

    @Test
    fun testParseBooleansAndNumbers() {
        val raw = "<|tool_call_start|>[toggle(enabled=True, count=-3, factor=1.5, flag=false)]<|tool_call_end|>"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals(true, call!!.getBoolean("enabled"))
        assertEquals(-3, call.getInt("count"))
        assertEquals(1.5, call.getDouble("factor"), 0.001)
        assertEquals(false, call.getBoolean("flag"))
    }

    @Test
    fun testParseNoneAndNull() {
        val raw = "<|tool_call_start|>[set_nullable(value=None, extra=null)]<|tool_call_end|>"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertTrue(call!!.isNull("value") || call.get("value") == JSONObject.NULL)
        assertTrue(call.isNull("extra") || call.get("extra") == JSONObject.NULL)
    }

    @Test
    fun testParseEmptyArgs() {
        val raw = "<|tool_call_start|>[list_files()]<|tool_call_end|>"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("list_files", call!!.getString("name"))
    }

    @Test
    fun testParseMultipleCalls() {
        val raw = "<|tool_call_start|>[web_search(query='cats'), web_search(query='dogs')]<|tool_call_end|>"
        val calls = LfmToolParser.parseToolCalls(raw)
        assertEquals(2, calls.size)
        assertEquals("cats", calls[0].getString("query"))
        assertEquals("dogs", calls[1].getString("query"))
    }

    @Test
    fun testParseWithoutBrackets() {
        val raw = "<|tool_call_start|>web_search(query='llama')<|tool_call_end|>"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("web_search", call!!.getString("name"))
        assertEquals("llama", call.getString("query"))
    }

    @Test
    fun testParseWithoutTags() {
        val raw = "[web_search(query='llama')]"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("web_search", call!!.getString("name"))
        assertEquals("llama", call.getString("query"))
    }

    @Test
    fun testExtractToolBlock() {
        val closed = "Some text\n<|tool_call_start|>[web_search(query='test')]<|tool_call_end|>\nMore text"
        val block = LfmToolParser.extractToolBlock(closed)
        assertEquals("<|tool_call_start|>[web_search(query='test')]<|tool_call_end|>", block)

        val unclosed = "Some text\n<|tool_call_start|>[web_search(query='streaming"
        val streamingBlock = LfmToolParser.extractToolBlock(unclosed)
        assertEquals("<|tool_call_start|>[web_search(query='streaming", streamingBlock)

        assertNull(LfmToolParser.extractToolBlock("Regular text with no tools"))
    }

    @Test
    fun testCommasAndParensInsideStringValues() {
        val raw = "<|tool_call_start|>[web_search(query='cats, dogs, and (birds) = awesome', limit=5)]<|tool_call_end|>"
        val call = LfmToolParser.parseFirst(raw)
        assertNotNull(call)
        assertEquals("cats, dogs, and (birds) = awesome", call!!.getString("query"))
        assertEquals(5, call.getInt("limit"))
    }
}
