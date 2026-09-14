package com.example.llama

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.withContext
import org.json.JSONArray
import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLDecoder
import java.net.URLEncoder

enum class ModelType { LLM, IMAGE_GEN, UNKNOWN }

data class HuggingFaceGgufFile(
    val repositoryId: String,
    val revision: String,
    val path: String,
    val sizeBytes: Long,
    val downloadUrl: String,
    val autoType: ModelType = ModelType.UNKNOWN
)

data class LocalModel(
    val file: File,
    val type: ModelType
)

class HuggingFaceModelRepository(context: Context) {
    val modelsDirectory: File = File(context.filesDir, "models").apply { mkdirs() }

    fun localModels(): List<LocalModel> = modelsDirectory.listFiles()
        ?.filter { file ->
            when {
                file.extension.equals("gguf", ignoreCase = true) -> true
                file.extension.equals("safetensors", ignoreCase = true) &&
                    !file.name.lowercase().contains("text_encoder") &&
                    !file.name.lowercase().contains("vae") &&
                    !file.name.lowercase().contains("clip") -> true
                else -> false
            }
        }
        ?.map { LocalModel(it, detectModelType(it)) }
        ?.sortedByDescending { it.file.lastModified() }
        .orEmpty()

    fun detectModelType(file: File): ModelType = when {
        file.extension.equals("gguf", ignoreCase = true) -> {
            val arch = readGgufArchitecture(file)
            if (arch == null) ModelType.UNKNOWN
            else if (isSdArchitecture(arch)) ModelType.IMAGE_GEN
            else ModelType.LLM
        }
        file.extension.equals("safetensors", ignoreCase = true) -> ModelType.IMAGE_GEN
        else -> ModelType.UNKNOWN
    }

    private fun isSdArchitecture(arch: String): Boolean {
        val lower = arch.lowercase()
        return lower.contains("sd1") || lower.contains("sd2") || lower.contains("sdxl") ||
            lower.contains("sd3") || lower.contains("flux") || lower.contains("wan") ||
            lower.contains("hunyuan") || lower.contains("stable-diffusion") ||
            lower.contains("sana") || lower.contains("pixart") || lower.contains("lumina") ||
            lower.contains("kolors") || lower.contains("auraflow") || lower.contains("playground")
    }

    private fun readGgufArchitecture(file: File): String? = runCatching {
        if (file.length() < 16 || file.length() > MAX_GGUF_FILE_SIZE) return null
        java.io.RandomAccessFile(file, "r").use { raf ->
            val buf = ByteArray(8)
            raf.readFully(buf, 0, 4)
            if (buf[0] != 'G'.code.toByte() || buf[1] != 'G'.code.toByte() ||
                buf[2] != 'U'.code.toByte() || buf[3] != 'F'.code.toByte()) return null
            raf.readFully(buf, 0, 4)
            val version = readLe32(buf, 0)
            raf.skipBytes(8) // tensor_count
            raf.readFully(buf)
            val kvCount = readLe64(buf, 0)
            if (kvCount <= 0 || kvCount > 100_000L) return null
            for (i in 0 until kvCount.toInt()) {
                val key = readGiString(raf)
                raf.readFully(buf, 0, 4)
                val valueType = readLe32(buf, 0)
                if (key == "general.architecture") {
                    require(valueType == GGUF_TYPE_STRING)
                    return readGiString(raf)
                }
                skipValue(raf, valueType)
            }
            null
        }
    }.getOrNull()

    private fun readGiString(raf: java.io.RandomAccessFile): String {
        val len = readLe64(ByteArray(8).also { raf.readFully(it) }, 0)
        require(len >= 0 && len <= 10_000_000)
        val bytes = ByteArray(len.toInt())
        raf.readFully(bytes)
        return String(bytes, Charsets.UTF_8)
    }

    private fun skipValue(raf: java.io.RandomAccessFile, type: Int) {
        val buf = ByteArray(8)
        when (type) {
            GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL -> raf.skipBytes(1)
            GGUF_TYPE_UINT16, GGUF_TYPE_INT16 -> raf.skipBytes(2)
            GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32 -> raf.skipBytes(4)
            GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64 -> raf.skipBytes(8)
            GGUF_TYPE_STRING -> {
                val len = readLe64(ByteArray(8).also { raf.readFully(it) }, 0)
                require(len >= 0 && len <= 10_000_000)
                raf.skipBytes(len.toInt())
            }
            GGUF_TYPE_ARRAY -> {
                raf.readFully(buf, 0, 4)
                val innerType = readLe32(buf, 0)
                raf.readFully(buf)
                val arrLen = readLe64(buf, 0)
                require(arrLen >= 0 && arrLen <= 10_000_000)
                val elSize = when (innerType) {
                    GGUF_TYPE_UINT8, GGUF_TYPE_INT8, GGUF_TYPE_BOOL -> 1
                    GGUF_TYPE_UINT16, GGUF_TYPE_INT16 -> 2
                    GGUF_TYPE_UINT32, GGUF_TYPE_INT32, GGUF_TYPE_FLOAT32 -> 4
                    GGUF_TYPE_UINT64, GGUF_TYPE_INT64, GGUF_TYPE_FLOAT64 -> 8
                    else -> 0
                }
                raf.skipBytes((arrLen * elSize).toInt())
            }
            else -> {}
        }
    }

    private fun readLe32(buf: ByteArray, offset: Int): Int =
        ((buf[offset + 3].toInt() and 0xFF) shl 24) or
        ((buf[offset + 2].toInt() and 0xFF) shl 16) or
        ((buf[offset + 1].toInt() and 0xFF) shl 8) or
        (buf[offset].toInt() and 0xFF)

    private fun readLe64(buf: ByteArray, offset: Int): Long =
        ((buf[offset + 7].toLong() and 0xFF) shl 56) or
        ((buf[offset + 6].toLong() and 0xFF) shl 48) or
        ((buf[offset + 5].toLong() and 0xFF) shl 40) or
        ((buf[offset + 4].toLong() and 0xFF) shl 32) or
        ((buf[offset + 3].toLong() and 0xFF) shl 24) or
        ((buf[offset + 2].toLong() and 0xFF) shl 16) or
        ((buf[offset + 1].toLong() and 0xFF) shl 8) or
        (buf[offset].toLong() and 0xFF)

    suspend fun listModelFiles(source: String, token: String?, filterType: ModelType? = null): List<HuggingFaceGgufFile> =
        withContext(Dispatchers.IO) {
            val repository = parseRepository(source)
            val files = mutableListOf<HuggingFaceGgufFile>()
            var pageUrl: URL? = treeUrl(repository)
            var pagesRead = 0

            while (pageUrl != null && pagesRead < MAX_TREE_PAGES && files.size < MAX_INDEXED_GGUF_FILES) {
                currentCoroutineContext().ensureActive()
                val connection = openConnection(pageUrl, token)
                try {
                    connection.connect()
                    require(connection.responseCode in 200..299) {
                        "Hugging Face returned HTTP ${connection.responseCode} while indexing ${repository.id}"
                    }
                    val entries = connection.inputStream.bufferedReader().use { reader ->
                        JSONArray(reader.readText())
                    }
                    for (index in 0 until entries.length()) {
                        currentCoroutineContext().ensureActive()
                        val entry = entries.optJSONObject(index) ?: continue
                        val path = entry.optString("path")
                        if (entry.optString("type") != "file") continue

                        val isGguf = path.endsWith(".gguf", ignoreCase = true)
                        val isSafetensor = path.endsWith(".safetensors", ignoreCase = true)
                        if (!isGguf && !isSafetensor) continue

                        val autoType = when {
                            isSafetensor -> ModelType.IMAGE_GEN
                            else -> ModelType.UNKNOWN
                        }
                        if (filterType != null && autoType != ModelType.UNKNOWN && autoType != filterType) continue

                        val lfsSize = entry.optJSONObject("lfs")?.optLong("size", -1L) ?: -1L
                        val size = entry.optLong("size", lfsSize).takeIf { it >= 0 } ?: 0L
                        files += HuggingFaceGgufFile(
                            repositoryId = repository.id,
                            revision = repository.revision,
                            path = path,
                            sizeBytes = size,
                            downloadUrl = resolveUrl(repository, path).toString(),
                            autoType = autoType
                        )
                        if (files.size >= MAX_INDEXED_GGUF_FILES) break
                    }
                    pageUrl = nextPageUrl(connection.getHeaderField("Link"))
                    pagesRead++
                } finally {
                    connection.disconnect()
                }
            }

            require(files.isNotEmpty()) { "No model files found in ${repository.id}" }
            files.sortedWith(compareBy<HuggingFaceGgufFile> { it.path.lowercase() }.thenBy { it.sizeBytes })
        }

    suspend fun listGgufFiles(source: String, token: String?): List<HuggingFaceGgufFile> =
        listModelFiles(source, token)

    suspend fun download(
        source: String,
        token: String?,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val url = normalizeUrl(source)
        val modelName = sanitizeFileName(URLDecoder.decode(url.path.substringAfterLast('/'), "UTF-8"))
        require(modelName.endsWith(".gguf", ignoreCase = true) ||
            modelName.endsWith(".safetensors", ignoreCase = true)) {
            "The URL must point to a .gguf or .safetensors file"
        }
        download(url, modelName, token, onProgress)
    }

    suspend fun download(
        file: HuggingFaceGgufFile,
        token: String?,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit
    ): File = withContext(Dispatchers.IO) {
        val localName = sanitizeFileName(file.path.replace('/', '_'))
        download(normalizeUrl(file.downloadUrl), localName, token, onProgress)
    }

    private suspend fun download(
        url: URL,
        modelName: String,
        token: String?,
        onProgress: (downloadedBytes: Long, totalBytes: Long) -> Unit
    ): File {
        require(modelName.endsWith(".gguf", ignoreCase = true) ||
            modelName.endsWith(".safetensors", ignoreCase = true)) {
            "The URL must point to a .gguf or .safetensors file"
        }

        val destination = File(modelsDirectory, modelName)
        require(!destination.exists()) { "$modelName already exists" }
        val partial = File(modelsDirectory, "$modelName.part")

        // Resume from a previous partial download so a dropped connection does not
        // restart the whole transfer (GGUF files can be several gigabytes).
        var downloadedBytes = partial.length().coerceAtLeast(0L)
        var totalBytes = -1L
        var completed = false
        var lastAttemptBytes = downloadedBytes

        return try {
            for (attempt in 0 until MAX_DOWNLOAD_ATTEMPTS) {
                currentCoroutineContext().ensureActive()
                if (attempt > 0) {
                    delay(RETRY_BACKOFF_MS * attempt)
                    downloadedBytes = partial.length().coerceAtLeast(0L)
                }
                lastAttemptBytes = downloadedBytes

                val connection = openConnection(url, token)
                if (downloadedBytes > 0) {
                    connection.setRequestProperty("Range", "bytes=$downloadedBytes-")
                }

                try {
                    connection.connect()
                    val code = connection.responseCode
                    require(code in 200..299) {
                        "Hugging Face returned HTTP $code"
                    }

                    val contentRange = connection.getHeaderField("Content-Range")
                    val lengthHeader = connection.contentLengthLong

                    if (code == HTTP_PARTIAL && contentRange != null) {
                        // "bytes start-end/total" -> recover the real total size.
                        val rangePart = contentRange.substringAfter("bytes ").substringBefore('/')
                        val start = rangePart.substringBefore('-').toLongOrNull() ?: downloadedBytes
                        if (start != downloadedBytes) {
                            // Server did not resume where we expected; restart cleanly.
                            partial.delete()
                            downloadedBytes = 0L
                        }
                        val slash = contentRange.substringAfter('/')
                        totalBytes = slash.toLongOrNull() ?: lengthHeader
                    } else if (downloadedBytes > 0 && code == HTTP_OK) {
                        // Server ignored the Range header; start over.
                        partial.delete()
                        downloadedBytes = 0L
                        totalBytes = lengthHeader
                    } else {
                        totalBytes = lengthHeader
                    }

                    if (totalBytes > 0) {
                        require(modelsDirectory.usableSpace + downloadedBytes > totalBytes + MIN_FREE_SPACE_BYTES) {
                            "Not enough free space for this model"
                        }
                    }

                    connection.inputStream.use { input ->
                        FileOutputStream(partial, downloadedBytes > 0L).use { output ->
                            val buffer = ByteArray(BUFFER_SIZE)
                            while (true) {
                                currentCoroutineContext().ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                output.write(buffer, 0, count)
                                downloadedBytes += count
                                onProgress(downloadedBytes, totalBytes)
                            }
                            output.fd.sync()
                        }
                    }
                    connection.disconnect()

                    if (totalBytes <= 0 || downloadedBytes >= totalBytes) {
                        require(partial.renameTo(destination)) { "Unable to finalize downloaded model" }
                        completed = true
                        break
                    }
                    // Loop and resume from where we left off.
                } catch (e: Exception) {
                    connection.disconnect()
                    if (attempt == MAX_DOWNLOAD_ATTEMPTS - 1) throw e
                    // Otherwise fall through and retry.
                }
            }

            require(completed) {
                "Download ended before the full file arrived (got $lastAttemptBytes of $totalBytes bytes)"
            }
            destination
        } finally {
            if (!completed) partial.delete()
        }
    }

    private fun parseRepository(source: String): HuggingFaceRepository {
        val trimmed = source.trim()
        require(trimmed.isNotEmpty()) { "Paste a Hugging Face repository URL or owner/repo" }
        val pathSegments = when {
            trimmed.startsWith("https://") || trimmed.startsWith("huggingface.co/") -> {
                val url = URL(if (trimmed.startsWith("https://")) trimmed else "https://$trimmed")
                require(isHuggingFaceHost(url)) { "Only huggingface.co repositories are allowed" }
                url.path.trim('/').split('/').filter { it.isNotBlank() }
            }
            else -> trimmed.trim('/').split('/').filter { it.isNotBlank() }
        }
        require(pathSegments.size >= 2) { "Use owner/repo or https://huggingface.co/owner/repo" }
        require(pathSegments[0] != "api" && pathSegments[0] != "datasets" && pathSegments[0] != "spaces") {
            "Only model repositories are supported"
        }
        val owner = decodePathPart(pathSegments[0])
        val name = decodePathPart(pathSegments[1])
        require(owner.matches(REPOSITORY_PART) && name.matches(REPOSITORY_PART)) {
            "Invalid Hugging Face repository name"
        }
        val revision = if (pathSegments.size >= 4 &&
            pathSegments[2] in setOf("tree", "blob", "resolve")
        ) {
            decodePathPart(pathSegments[3])
        } else {
            DEFAULT_REVISION
        }
        require(revision.isNotBlank() && !revision.contains("..")) { "Invalid repository revision" }
        return HuggingFaceRepository("$owner/$name", revision)
    }

    private fun treeUrl(repository: HuggingFaceRepository): URL = URL(
        "$HUGGING_FACE_ROOT/api/models/${encodePathPart(repository.id.substringBefore('/'))}/" +
            "${encodePathPart(repository.id.substringAfter('/'))}/tree/" +
            "${encodePathPart(repository.revision)}?recursive=true&expand=false&limit=$TREE_PAGE_SIZE"
    )

    private fun resolveUrl(repository: HuggingFaceRepository, path: String): URL {
        val encodedPath = path.split('/').joinToString("/") { encodePathPart(it) }
        return URL(
            "$HUGGING_FACE_ROOT/${encodePathPart(repository.id.substringBefore('/'))}/" +
                "${encodePathPart(repository.id.substringAfter('/'))}/resolve/" +
                "${encodePathPart(repository.revision)}/$encodedPath"
        )
    }

    private fun nextPageUrl(linkHeader: String?): URL? {
        val next = NEXT_PAGE_REGEX.find(linkHeader.orEmpty())?.groupValues?.getOrNull(1) ?: return null
        val url = URL(next)
        require(isHuggingFaceHost(url)) { "Unexpected Hugging Face pagination host" }
        return url
    }

    private fun openConnection(url: URL, token: String?): HttpURLConnection {
        require(url.protocol == "https" && isHuggingFaceHost(url)) {
            "Only huggingface.co downloads are allowed"
        }
        return (url.openConnection() as HttpURLConnection).apply {
            connectTimeout = CONNECT_TIMEOUT_MS
            readTimeout = READ_TIMEOUT_MS
            instanceFollowRedirects = true
            setRequestProperty("User-Agent", "LlamaChat-Android")
            setRequestProperty("Accept", "application/json, */*")
            token?.trim()?.takeIf { it.isNotEmpty() }?.let {
                require(!it.contains('\n') && !it.contains('\r')) { "Invalid token" }
                setRequestProperty("Authorization", "Bearer $it")
            }
        }
    }

    private fun normalizeUrl(source: String): URL {
        val trimmed = source.trim()
        require(trimmed.isNotEmpty()) { "Paste a Hugging Face GGUF URL" }
        val normalized = when {
            trimmed.startsWith("https://") -> trimmed
            trimmed.startsWith("huggingface.co/") -> "https://$trimmed"
            else -> error("Use a full huggingface.co URL")
        }.replace("/blob/", "/resolve/")
        val url = URL(normalized)
        require(url.protocol == "https") { "Only HTTPS downloads are allowed" }
        require(isHuggingFaceHost(url)) {
            "Only huggingface.co downloads are allowed"
        }
        return url
    }

    private fun isHuggingFaceHost(url: URL): Boolean =
        url.host == "huggingface.co" || url.host.endsWith(".huggingface.co")

    private fun encodePathPart(part: String): String =
        URLEncoder.encode(part, "UTF-8").replace("+", "%20")

    private fun decodePathPart(part: String): String = URLDecoder.decode(part, "UTF-8")

    private fun sanitizeFileName(fileName: String): String {
        val safeName = fileName.substringBefore('?').replace(Regex("[^A-Za-z0-9._-]"), "_")
        require(safeName.isNotBlank() && safeName != "." && safeName != "..") { "Invalid file name" }
        return safeName
    }

    companion object {
        private const val HUGGING_FACE_ROOT = "https://huggingface.co"
        private const val DEFAULT_REVISION = "main"
        private const val CONNECT_TIMEOUT_MS = 30_000
        private const val READ_TIMEOUT_MS = 60_000
        private const val BUFFER_SIZE = 1024 * 1024
        private const val MIN_FREE_SPACE_BYTES = 64L * 1024 * 1024
        private const val TREE_PAGE_SIZE = 1_000
        private const val MAX_TREE_PAGES = 20
        private const val MAX_INDEXED_GGUF_FILES = 500
        private const val HTTP_OK = 200
        private const val HTTP_PARTIAL = 206
        private const val MAX_DOWNLOAD_ATTEMPTS = 8
        private const val RETRY_BACKOFF_MS = 1_000L
        private val REPOSITORY_PART = Regex("[A-Za-z0-9][A-Za-z0-9._-]*")
        private val NEXT_PAGE_REGEX = Regex("""<([^>]+)>\s*;\s*rel="?next"?""")

        // GGUF value types for lightweight header scanning
        private const val GGUF_TYPE_UINT8   = 0
        private const val GGUF_TYPE_INT8    = 1
        private const val GGUF_TYPE_UINT16  = 2
        private const val GGUF_TYPE_INT16   = 3
        private const val GGUF_TYPE_UINT32  = 4
        private const val GGUF_TYPE_INT32   = 5
        private const val GGUF_TYPE_FLOAT32 = 6
        private const val GGUF_TYPE_BOOL    = 7
        private const val GGUF_TYPE_STRING  = 8
        private const val GGUF_TYPE_ARRAY   = 9
        private const val GGUF_TYPE_UINT64  = 10
        private const val GGUF_TYPE_INT64   = 11
        private const val GGUF_TYPE_FLOAT64 = 12
        private const val MAX_GGUF_FILE_SIZE = 8L * 1024 * 1024 * 1024
    }

    private data class HuggingFaceRepository(val id: String, val revision: String)
}
