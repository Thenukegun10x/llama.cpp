package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import java.util.Locale

class RemoteGgufAdapter(
    private var files: List<HuggingFaceGgufFile> = emptyList(),
    private val onDownload: (HuggingFaceGgufFile) -> Unit
) : RecyclerView.Adapter<RemoteGgufAdapter.GgufViewHolder>() {

    private var downloadingPath: String? = null

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): GgufViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_remote_gguf, parent, false)
        return GgufViewHolder(view)
    }

    override fun onBindViewHolder(holder: GgufViewHolder, position: Int) {
        val file = files[position]
        holder.name.text = file.path.substringAfterLast('/')
        holder.details.text = buildString {
            append(file.path.substringBeforeLast('/', missingDelimiterValue = ""))
                .takeIf { it.isNotBlank() }
                ?.let { append("$it  -  ") }
            append(formatBytes(file.sizeBytes))
        }
        val downloading = file.path == downloadingPath
        holder.downloadButton.isEnabled = downloadingPath == null
        holder.downloadButton.text = if (downloading) {
            holder.itemView.context.getString(R.string.downloading)
        } else {
            holder.itemView.context.getString(R.string.download)
        }
        holder.downloadButton.setOnClickListener { onDownload(file) }
    }

    override fun getItemCount(): Int = files.size

    fun replaceFiles(updatedFiles: List<HuggingFaceGgufFile>) {
        files = updatedFiles
        downloadingPath = null
        notifyDataSetChanged()
    }

    fun setDownloading(path: String?) {
        downloadingPath = path
        notifyDataSetChanged()
    }

    private fun formatBytes(bytes: Long): String {
        if (bytes <= 0) return "Size unavailable"
        if (bytes < 1024) return "$bytes B"
        val units = arrayOf("KB", "MB", "GB", "TB")
        var value = bytes.toDouble()
        var unit = -1
        while (value >= 1024 && unit < units.lastIndex) {
            value /= 1024
            unit++
        }
        return String.format(Locale.US, "%.1f %s", value, units[unit])
    }

    class GgufViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val name: TextView = view.findViewById(R.id.remote_gguf_name)
        val details: TextView = view.findViewById(R.id.remote_gguf_details)
        val downloadButton: MaterialButton = view.findViewById(R.id.download_remote_gguf)
    }
}
