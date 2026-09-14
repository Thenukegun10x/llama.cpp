package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import com.google.android.material.button.MaterialButton
import com.google.android.material.chip.Chip
import java.io.File
import java.util.Locale

class LocalModelAdapter(
    private var models: List<LocalModel>,
    private val selectedModelName: () -> String?,
    private val onLoad: (LocalModel) -> Unit,
    private val onDelete: (File) -> Unit
) : RecyclerView.Adapter<LocalModelAdapter.ModelViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ModelViewHolder {
        val view = LayoutInflater.from(parent.context).inflate(R.layout.item_local_model, parent, false)
        return ModelViewHolder(view)
    }

    override fun onBindViewHolder(holder: ModelViewHolder, position: Int) {
        val model = models[position]
        holder.name.text = model.file.name
        holder.details.text = if (model.file.name == selectedModelName()) {
            holder.itemView.context.getString(
                R.string.loaded_model_size, formatBytes(model.file.length())
            )
        } else {
            formatBytes(model.file.length())
        }

        when (model.type) {
            ModelType.LLM -> {
                holder.typeChip.visibility = View.VISIBLE
                holder.typeChip.text = "LLM"
            }
            ModelType.IMAGE_GEN -> {
                holder.typeChip.visibility = View.VISIBLE
                holder.typeChip.text = "Image"
            }
            ModelType.UNKNOWN -> {
                holder.typeChip.visibility = View.GONE
            }
        }

        holder.loadButton.isEnabled = model.file.name != selectedModelName()
        holder.loadButton.setOnClickListener { onLoad(model) }
        holder.deleteButton.setOnClickListener { onDelete(model.file) }
    }

    override fun getItemCount(): Int = models.size

    fun replaceModels(updatedModels: List<LocalModel>) {
        models = updatedModels
        notifyDataSetChanged()
    }

    private fun formatBytes(bytes: Long): String {
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

    class ModelViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val name: TextView = view.findViewById(R.id.model_name)
        val details: TextView = view.findViewById(R.id.model_details)
        val typeChip: Chip = view.findViewById(R.id.model_type_chip)
        val loadButton: MaterialButton = view.findViewById(R.id.load_model)
        val deleteButton: MaterialButton = view.findViewById(R.id.delete_model)
    }
}
