package com.example.llama

import android.annotation.SuppressLint
import android.content.Context
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import com.k2fsa.sherpa.onnx.FeatureConfig
import com.k2fsa.sherpa.onnx.OnlineModelConfig
import com.k2fsa.sherpa.onnx.OnlineRecognizer
import com.k2fsa.sherpa.onnx.OnlineRecognizerConfig
import com.k2fsa.sherpa.onnx.OnlineTransducerModelConfig
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.isActive
import kotlinx.coroutines.withContext
import kotlin.math.max

class OnDeviceSpeechRecognizer(private val context: Context) : AutoCloseable {
    @Volatile
    private var listening = false

    @Volatile
    private var audioRecord: AudioRecord? = null

    private var recognizer: OnlineRecognizer? = null

    val isListening: Boolean
        get() = listening

    @SuppressLint("MissingPermission")
    suspend fun listen(onPartialResult: suspend (String) -> Unit): String =
        withContext(Dispatchers.IO) {
            check(!listening) { "Speech recognition is already active" }
            listening = true

            val onlineRecognizer = recognizer ?: createRecognizer().also { recognizer = it }
            if (!listening) return@withContext ""

            val stream = onlineRecognizer.createStream()
            val minimumBufferBytes = AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )
            check(minimumBufferBytes > 0) { "This device cannot record 16 kHz mono audio" }

            val recorder = AudioRecord.Builder()
                .setAudioSource(MediaRecorder.AudioSource.VOICE_RECOGNITION)
                .setAudioFormat(
                    AudioFormat.Builder()
                        .setEncoding(AudioFormat.ENCODING_PCM_16BIT)
                        .setSampleRate(SAMPLE_RATE)
                        .setChannelMask(AudioFormat.CHANNEL_IN_MONO)
                        .build()
                )
                .setBufferSizeInBytes(max(minimumBufferBytes, SAMPLE_RATE * 2))
                .build()

            check(recorder.state == AudioRecord.STATE_INITIALIZED) {
                "The microphone could not be initialized"
            }

            audioRecord = recorder
            val pcm = ShortArray(AUDIO_CHUNK_SAMPLES)
            var latestResult = ""

            try {
                recorder.startRecording()
                while (listening && currentCoroutineContext().isActive) {
                    val sampleCount = recorder.read(
                        pcm,
                        0,
                        pcm.size,
                        AudioRecord.READ_BLOCKING
                    )
                    if (sampleCount <= 0) {
                        if (!listening) break
                        check(sampleCount != AudioRecord.ERROR_DEAD_OBJECT) {
                            "The microphone disconnected"
                        }
                        continue
                    }

                    val samples = FloatArray(sampleCount) { index -> pcm[index] / 32768.0f }
                    stream.acceptWaveform(samples, SAMPLE_RATE)
                    while (onlineRecognizer.isReady(stream)) onlineRecognizer.decode(stream)

                    val partial = onlineRecognizer.getResult(stream).text.trim()
                    if (partial.isNotBlank() && partial != latestResult) {
                        latestResult = partial
                        onPartialResult(partial)
                    }
                    if (onlineRecognizer.isEndpoint(stream)) break
                }

                stream.inputFinished()
                while (onlineRecognizer.isReady(stream)) onlineRecognizer.decode(stream)
                onlineRecognizer.getResult(stream).text.trim().ifBlank { latestResult }
            } finally {
                listening = false
                audioRecord = null
                runCatching {
                    if (recorder.recordingState == AudioRecord.RECORDSTATE_RECORDING) recorder.stop()
                }
                recorder.release()
                stream.release()
            }
        }

    fun stop() {
        listening = false
        runCatching { audioRecord?.stop() }
    }

    override fun close() {
        stop()
        recognizer?.release()
        recognizer = null
    }

    private fun createRecognizer(): OnlineRecognizer {
        val modelConfig = OnlineModelConfig(
            transducer = OnlineTransducerModelConfig(
                encoder = "$MODEL_DIRECTORY/$ENCODER_MODEL",
                decoder = "$MODEL_DIRECTORY/$DECODER_MODEL",
                joiner = "$MODEL_DIRECTORY/$JOINER_MODEL"
            ),
            tokens = "$MODEL_DIRECTORY/$TOKENS_FILE",
            numThreads = 2,
            provider = "cpu"
        )
        val config = OnlineRecognizerConfig(
            featConfig = FeatureConfig(sampleRate = SAMPLE_RATE, featureDim = 80, dither = 0.0f),
            modelConfig = modelConfig,
            enableEndpoint = true,
            decodingMethod = "greedy_search"
        )
        return OnlineRecognizer(context.assets, config)
    }

    companion object {
        private const val SAMPLE_RATE = 16_000
        private const val AUDIO_CHUNK_SAMPLES = 2_048
        private const val MODEL_DIRECTORY = "stt"
        private const val ENCODER_MODEL = "encoder-epoch-99-avg-1.int8.onnx"
        private const val DECODER_MODEL = "decoder-epoch-99-avg-1.onnx"
        private const val JOINER_MODEL = "joiner-epoch-99-avg-1.int8.onnx"
        private const val TOKENS_FILE = "tokens.txt"
    }
}
