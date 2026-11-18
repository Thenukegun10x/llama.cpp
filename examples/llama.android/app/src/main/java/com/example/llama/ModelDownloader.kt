package com.example.llama

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.delay
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import java.io.File

data class DownloadProgress(
    val downloadedBytes: Long,
    val totalBytes: Long,
    val status: Int
)

data class ActiveDownload(
    val id: Long,
    val fileName: String,
    val downloadedBytes: Long,
    val totalBytes: Long
)

// Wraps the system DownloadManager so transfers survive backgrounding and
// process death. Progress is polled; cancellation of the observer does not
// cancel the download itself.
class ModelDownloader(context: Context) {
    private val appContext = context.applicationContext
    private val manager = appContext.getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager

    fun enqueue(url: String, fileName: String, token: String?): Long {
        val request = DownloadManager.Request(Uri.parse(url))
            .setTitle(fileName)
            .setMimeType("application/octet-stream")
            .setNotificationVisibility(DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED)
            .setAllowedOverMetered(true)
            .setAllowedOverRoaming(true)
            .setDestinationInExternalFilesDir(appContext, null, "$DOWNLOAD_SUBDIR/$fileName")
        token?.trim()?.takeIf { it.isNotEmpty() }?.let {
            require(!it.contains('\n') && !it.contains('\r')) { "Invalid token" }
            request.addRequestHeader("Authorization", "Bearer $it")
        }
        return manager.enqueue(request)
    }

    fun observe(id: Long): Flow<DownloadProgress> = flow {
        while (true) {
            currentCoroutineContext().ensureActive()
            val progress = progressOf(id) ?: error("Download was removed")
            emit(progress)
            when (progress.status) {
                DownloadManager.STATUS_SUCCESSFUL -> return@flow
                DownloadManager.STATUS_FAILED -> error("Download failed (reason ${failureReason(id)})")
            }
            delay(POLL_INTERVAL_MS)
        }
    }

    fun progressOf(id: Long): DownloadProgress? {
        manager.query(DownloadManager.Query().setFilterById(id)).use { cursor ->
            if (!cursor.moveToFirst()) return null
            return DownloadProgress(
                downloadedBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)),
                totalBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES)),
                status = cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))
            )
        }
    }

    fun activeDownload(): ActiveDownload? {
        val query = DownloadManager.Query().setFilterByStatus(
            DownloadManager.STATUS_PENDING or DownloadManager.STATUS_RUNNING or DownloadManager.STATUS_PAUSED
        )
        val prefix = Uri.fromFile(downloadDir()).toString()
        manager.query(query).use { cursor ->
            while (cursor.moveToNext()) {
                val localUri = cursor.getString(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_LOCAL_URI)) ?: continue
                if (!localUri.startsWith(prefix)) continue
                return ActiveDownload(
                    id = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_ID)),
                    fileName = cursor.getString(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TITLE)).orEmpty(),
                    downloadedBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)),
                    totalBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                )
            }
        }
        return null
    }

    fun cancel(id: Long) = manager.remove(id)

    private fun failureReason(id: Long): Int =
        manager.query(DownloadManager.Query().setFilterById(id)).use { cursor ->
            if (cursor.moveToFirst()) cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_REASON)) else -1
        }

    private fun downloadDir(): File =
        File(appContext.getExternalFilesDir(null) ?: appContext.filesDir, DOWNLOAD_SUBDIR)

    companion object {
        const val DOWNLOAD_SUBDIR = "models"
        private const val POLL_INTERVAL_MS = 500L
    }
}
