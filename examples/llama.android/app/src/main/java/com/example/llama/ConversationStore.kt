package com.example.llama

import android.content.Context
import org.json.JSONArray
import org.json.JSONObject
import java.util.UUID

data class Conversation(
    val id: String,
    var title: String,
    val createdAt: Long,
    val messages: MutableList<Message>
)

class ConversationStore(context: Context) {
    private val preferences = context.getSharedPreferences(PREFERENCES_NAME, Context.MODE_PRIVATE)

    fun loadConversations(): MutableList<Conversation> {
        val storedConversations = preferences.getString(KEY_CONVERSATIONS, null) ?: return mutableListOf()
        return runCatching {
            val conversations = mutableListOf<Conversation>()
            val conversationArray = JSONArray(storedConversations)
            for (index in 0 until conversationArray.length()) {
                val item = conversationArray.getJSONObject(index)
                val messages = mutableListOf<Message>()
                val messageArray = item.optJSONArray("messages") ?: JSONArray()
                for (messageIndex in 0 until messageArray.length()) {
                    val message = messageArray.getJSONObject(messageIndex)
                    messages += Message(
                        id = message.optString("id", UUID.randomUUID().toString()),
                        content = message.optString("content"),
                        isUser = message.optBoolean("isUser"),
                        thinking = message.optString("thinking"),
                        tooling = message.optString("tooling"),
                        timestamp = message.optLong("timestamp", System.currentTimeMillis())
                    )
                }
                conversations += Conversation(
                    id = item.optString("id", UUID.randomUUID().toString()),
                    title = item.optString("title", "New conversation"),
                    createdAt = item.optLong("createdAt", System.currentTimeMillis()),
                    messages = messages
                )
            }
            conversations
        }.getOrDefault(mutableListOf())
    }

    fun saveConversations(conversations: List<Conversation>, selectedConversationId: String?) {
        val conversationArray = JSONArray()
        conversations.forEach { conversation ->
            val messageArray = JSONArray()
            conversation.messages.forEach { message ->
                messageArray.put(
                    JSONObject()
                        .put("id", message.id)
                        .put("content", message.content)
                        .put("isUser", message.isUser)
                        .put("thinking", message.thinking)
                        .put("tooling", message.tooling)
                        .put("timestamp", message.timestamp)
                )
            }
            conversationArray.put(
                JSONObject()
                    .put("id", conversation.id)
                    .put("title", conversation.title)
                    .put("createdAt", conversation.createdAt)
                    .put("messages", messageArray)
            )
        }
        preferences.edit()
            .putString(KEY_CONVERSATIONS, conversationArray.toString())
            .putString(KEY_SELECTED_CONVERSATION, selectedConversationId)
            .apply()
    }

    fun selectedConversationId(): String? = preferences.getString(KEY_SELECTED_CONVERSATION, null)

    fun lastModelName(): String? = preferences.getString(KEY_MODEL_NAME, null)

    fun saveLastModelName(modelName: String) {
        preferences.edit().putString(KEY_MODEL_NAME, modelName).apply()
    }

    fun clearLastModelName() {
        preferences.edit().remove(KEY_MODEL_NAME).apply()
    }

    companion object {
        private const val PREFERENCES_NAME = "llama_chat_history"
        private const val KEY_CONVERSATIONS = "conversations"
        private const val KEY_SELECTED_CONVERSATION = "selected_conversation"
        private const val KEY_MODEL_NAME = "last_model"
    }
}
