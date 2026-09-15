package com.example.llama

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import java.io.File

class SandboxFileToolsTest {

    private fun createTools(): SandboxFileTools {
        val tempDir = File(System.getProperty("java.io.tmpdir"), "test-sandbox-${System.currentTimeMillis()}").apply { mkdirs() }
        return SandboxFileTools(tempDir)
    }

    @Test
    fun parseToolCallSupportsJson() {
        val tools = createTools()
        val call = tools.parseToolCall("""<tool_call>{"name":"read_file","path":"notes.txt"}</tool_call>""")
        assertNotNull(call)
        assertEquals("read_file", call!!.name)
        assertEquals("notes.txt", call.path)
    }

    @Test
    fun parseToolCallSupportsLfm() {
        val tools = createTools()
        val call = tools.parseToolCall(
            "<|tool_call_start|>[write_file(path='notes.txt', content='hello world')]<|tool_call_end|>"
        )
        assertNotNull(call)
        assertEquals("write_file", call!!.name)
        assertEquals("notes.txt", call.path)
        assertEquals("hello world", call.content)
    }

    @Test
    fun parseToolCallSupportsLfmListFiles() {
        val tools = createTools()
        val call = tools.parseToolCall("<|tool_call_start|>[list_files()]<|tool_call_end|>")
        assertNotNull(call)
        assertEquals("list_files", call!!.name)
    }
}
