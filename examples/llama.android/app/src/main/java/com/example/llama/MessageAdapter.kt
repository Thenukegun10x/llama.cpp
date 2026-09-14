package com.example.llama

import android.util.Log
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.ImageView
import android.widget.LinearLayout
import android.widget.TextView
import androidx.recyclerview.widget.RecyclerView
import java.text.DateFormat
import java.util.Date

data class Message(
    val id: String,
    val content: String,
    val isUser: Boolean,
    val thinking: String = "",
    val tooling: String = "",
    val timestamp: Long = System.currentTimeMillis()
)

class MessageAdapter(
    private val messages: List<Message>
) : RecyclerView.Adapter<RecyclerView.ViewHolder>() {

    private val expandedThinking = mutableSetOf<String>()
    private val expandedTooling = mutableSetOf<String>()

    companion object {
        private const val VIEW_TYPE_USER = 1
        private const val VIEW_TYPE_ASSISTANT = 2
    }

    override fun getItemViewType(position: Int): Int {
        return if (messages[position].isUser) VIEW_TYPE_USER else VIEW_TYPE_ASSISTANT
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): RecyclerView.ViewHolder {
        val layoutInflater = LayoutInflater.from(parent.context)
        return if (viewType == VIEW_TYPE_USER) {
            val view = layoutInflater.inflate(R.layout.item_message_user, parent, false)
            UserMessageViewHolder(view)
        } else {
            val view = layoutInflater.inflate(R.layout.item_message_assistant, parent, false)
            AssistantMessageViewHolder(view)
        }
    }

    override fun onBindViewHolder(holder: RecyclerView.ViewHolder, position: Int) {
        val message = messages[position]
        if (holder is UserMessageViewHolder) {
            holder.itemView.findViewById<TextView>(R.id.msg_content).text = MarkdownRenderer.render(message.content)
            holder.itemView.findViewById<TextView>(R.id.msg_time).text =
                DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(message.timestamp))
        } else if (holder is AssistantMessageViewHolder) {
            holder.msgContent.text = MarkdownRenderer.render(message.content)
            holder.msgContent.visibility = if (message.content.isBlank()) View.GONE else View.VISIBLE
            holder.msgTime.text =
                DateFormat.getTimeInstance(DateFormat.SHORT).format(Date(message.timestamp))

            if (message.thinking.isNotBlank()) {
                Log.d("MsgAdapter", "bind thinking id=${message.id} len=${message.thinking.length}")
                holder.thinkingContainer.visibility = View.VISIBLE
                holder.thinkingText.text = MarkdownRenderer.render(message.thinking)
                val isExpanded = expandedThinking.contains(message.id)
                holder.thinkingText.visibility = if (isExpanded) View.VISIBLE else View.GONE
                holder.thinkingChevron.rotation = if (isExpanded) 180f else 0f
                holder.thinkingHeader.setOnClickListener {
                    if (expandedThinking.contains(message.id)) {
                        expandedThinking.remove(message.id)
                        holder.thinkingText.visibility = View.GONE
                        holder.thinkingChevron.rotation = 0f
                    } else {
                        expandedThinking.add(message.id)
                        holder.thinkingText.visibility = View.VISIBLE
                        holder.thinkingChevron.rotation = 180f
                    }
                }
            } else {
                holder.thinkingContainer.visibility = View.GONE
                expandedThinking.remove(message.id)
            }

            if (message.tooling.isNotBlank()) {
                holder.toolingContainer.visibility = View.VISIBLE
                holder.toolingText.text = MarkdownRenderer.render(message.tooling)
                val isExpanded = expandedTooling.contains(message.id)
                holder.toolingText.visibility = if (isExpanded) View.VISIBLE else View.GONE
                holder.toolingChevron.rotation = if (isExpanded) 180f else 0f
                holder.toolingHeader.setOnClickListener {
                    if (expandedTooling.contains(message.id)) {
                        expandedTooling.remove(message.id)
                        holder.toolingText.visibility = View.GONE
                        holder.toolingChevron.rotation = 0f
                    } else {
                        expandedTooling.add(message.id)
                        holder.toolingText.visibility = View.VISIBLE
                        holder.toolingChevron.rotation = 180f
                    }
                }
            } else {
                holder.toolingContainer.visibility = View.GONE
                expandedTooling.remove(message.id)
            }

            if (message.content.isBlank() && message.thinking.isBlank() && message.tooling.isBlank()) {
                holder.msgContent.visibility = View.VISIBLE
                holder.msgContent.text = holder.itemView.context.getString(R.string.thinking)
            }
        }
    }

    override fun getItemCount(): Int = messages.size

    fun expandThinking(id: String) {
        expandedThinking.add(id)
    }

    fun collapseAllThinking() {
        expandedThinking.clear()
    }

    fun expandTooling(id: String) {
        expandedTooling.add(id)
    }

    class UserMessageViewHolder(view: View) : RecyclerView.ViewHolder(view)

    class AssistantMessageViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val msgContent: TextView = view.findViewById(R.id.msg_content)
        val msgTime: TextView = view.findViewById(R.id.msg_time)
        val thinkingContainer: LinearLayout = view.findViewById(R.id.thinking_container)
        val thinkingHeader: LinearLayout = view.findViewById(R.id.thinking_header)
        val thinkingText: TextView = view.findViewById(R.id.thinking_text)
        val thinkingChevron: ImageView = view.findViewById(R.id.thinking_chevron)
        val toolingContainer: LinearLayout = view.findViewById(R.id.tooling_container)
        val toolingHeader: LinearLayout = view.findViewById(R.id.tooling_header)
        val toolingText: TextView = view.findViewById(R.id.tooling_text)
        val toolingChevron: ImageView = view.findViewById(R.id.tooling_chevron)
    }
}
