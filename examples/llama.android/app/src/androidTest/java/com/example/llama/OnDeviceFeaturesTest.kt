package com.example.llama

import androidx.test.ext.junit.runners.AndroidJUnit4
import androidx.test.platform.app.InstrumentationRegistry
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Test
import org.junit.runner.RunWith

@RunWith(AndroidJUnit4::class)
class OnDeviceFeaturesTest {
    private val context = InstrumentationRegistry.getInstrumentation().targetContext

    @Test
    fun bundledSpeechModelLoadsAndReturnsResult() {
        val modelDirectory = "stt"
        val modelConfig = OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = "$modelDirectory/encoder-epoch-99-avg-1.int8.onnx",
                decoder = "$modelDirectory/decoder-epoch-99-avg-1.onnx",
                joiner = "$modelDirectory/joiner-epoch-99-avg-1.int8.onnx"
            ),
            tokens = "$modelDirectory/tokens.txt",
            numThreads = 2,
            provider = "cpu"
        )
        val config = OnlineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = 16_000, featureDim = 80, dither = 0.0f),
            modelConfig = modelConfig,
            enableEndpoint = true,
            decodingMethod = "greedy_search"
        )
        val recognizer = OnlineRecognizer(context.assets, config)
        val stream = recognizer.createStream()
        try {
            stream.acceptWaveform(FloatArray(16_000), 16_000)
            stream.inputFinished()
            while (recognizer.isReady(stream)) recognizer.decode(stream)
            assertNotNull(recognizer.getResult(stream))
        } finally {
            stream.release()
            recognizer.release()
        }
    }

    @Test
    fun smallModelToolCallsAreParsedLeniently() {
        val tools = SandboxFileTools(context)

        val tagged = tools.parseToolCall(
            "<TOOL CALL>\n{\"tool\":\"open file\",\"file_path\":\"notes.txt\"}\n</TOOL CALL>"
        )
        assertNotNull(tagged)
        assertEquals("read_file", tagged?.name)
        assertEquals("notes.txt", tagged?.path)

        val bare = tools.parseToolCall(
            "I will do that now: {\"action\":\"save\",\"filename\":\"todo.txt\",\"text\":\"one\"}"
        )
        assertEquals("write_file", bare?.name)
        assertEquals("one", bare?.content)
    }

    @Test
    fun editMatchingToleratesCaseAndWhitespace() {
        val tools = SandboxFileTools(context)
        val path = "codex-test/friendly-edit.txt"
        try {
            tools.writeFile(path, "Heading\n\nKeep   this line\nFooter")
            tools.editFile(path, "keep this LINE", "Updated line")
            assertEquals("Heading\n\nUpdated line\nFooter", tools.readFile(path))
        } finally {
            tools.deleteFile(path)
        }
    }
}
