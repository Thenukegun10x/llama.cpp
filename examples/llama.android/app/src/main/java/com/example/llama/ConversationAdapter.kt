package com.example.llama

import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import android.widget.TextView
import androidx.core.view.isVisible
import androidx.recyclerview.widget.RecyclerView

class ConversationAdapter(
    private val conversations: List<Conversation>,
    private var selectedConversationId: String?,
    private val onConversationSelected: (Conversation) -> Unit
) : RecyclerView.Adapter<ConversationAdapter.ConversationViewHolder>() {

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): ConversationViewHolder {
        val view = LayoutInflater.from(parent.context)
            .inflate(R.layout.item_conversation, parent, false)
        return ConversationViewHolder(view)
    }

    override fun onBindViewHolder(holder: ConversationViewHolder, position: Int) {
        val conversation = conversations[position]
        holder.title.text = conversation.title
        holder.preview.text = conversation.messages.lastOrNull()?.content ?: "No messages yet"
        holder.selected.isVisible = conversation.id == selectedConversationId
        holder.itemView.setOnClickListener { onConversationSelected(conversation) }
    }

    override fun getItemCount(): Int = conversations.size

    fun setSelectedConversation(conversationId: String) {
        selectedConversationId = conversationId
        notifyDataSetChanged()
    }

    class ConversationViewHolder(view: View) : RecyclerView.ViewHolder(view) {
        val title: TextView = view.findViewById(R.id.conversation_title)
        val preview: TextView = view.findViewById(R.id.conversation_preview)
        val selected: View = view.findViewById(R.id.conversation_selected)
    }
}
