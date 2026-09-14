package com.example.llama

import android.graphics.Typeface
import android.text.Spannable
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.style.BackgroundColorSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.text.style.TypefaceSpan

object MarkdownRenderer {

    fun render(text: String): Spannable {
        val sb = SpannableStringBuilder()
        val lines = text.split("\n")
        var inCodeBlock = false

        for (line in lines) {
            if (inCodeBlock) {
                if (line.trim() == "```") {
                    inCodeBlock = false
                    continue
                }
                val start = sb.length
                sb.append(line)
                sb.setSpan(TypefaceSpan("monospace"), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                continue
            }

            val trimmed = line.trim()
            if (trimmed.startsWith("```")) {
                inCodeBlock = true
                continue
            }

            if (trimmed.startsWith("### ")) {
                appendHeading(sb, trimmed.substring(4), 1.20f)
            } else if (trimmed.startsWith("## ")) {
                appendHeading(sb, trimmed.substring(3), 1.35f)
            } else if (trimmed.startsWith("# ")) {
                appendHeading(sb, trimmed.substring(2), 1.50f)
            } else if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                sb.append("  \u2022 ")
                appendInline(sb, trimmed.substring(2))
            } else if (trimmed.isEmpty()) {
                sb.append("\n")
            } else {
                appendInline(sb, trimmed)
            }
        }

        return SpannableString.valueOf(sb)
    }

    private fun appendHeading(sb: SpannableStringBuilder, text: String, scale: Float) {
        val start = sb.length
        sb.append(text.trim())
        sb.setSpan(StyleSpan(Typeface.BOLD), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        sb.setSpan(RelativeSizeSpan(scale), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
        sb.append("\n")
    }

    private fun appendInline(sb: SpannableStringBuilder, text: String) {
        var i = 0
        while (i < text.length) {
            if (text.startsWith("**", i)) {
                val end = text.indexOf("**", i + 2)
                if (end >= 0) {
                    val start = sb.length
                    sb.append(text, i + 2, end)
                    sb.setSpan(StyleSpan(Typeface.BOLD), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                    i = end + 2
                    continue
                }
            }
            if (text.startsWith("*", i) && (i + 1 >= text.length || text[i + 1] != '*')) {
                val end = text.indexOf("*", i + 1)
                if (end >= 0 && (end + 1 >= text.length || text[end + 1] != '*')) {
                    val start = sb.length
                    sb.append(text, i + 1, end)
                    sb.setSpan(StyleSpan(Typeface.ITALIC), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                    i = end + 1
                    continue
                }
            }
            if (text.startsWith("`", i)) {
                val end = text.indexOf('`', i + 1)
                if (end >= 0) {
                    val start = sb.length
                    sb.append(text, i + 1, end)
                    sb.setSpan(TypefaceSpan("monospace"), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                    sb.setSpan(BackgroundColorSpan(0x1A000000), start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                    i = end + 1
                    continue
                }
            }
            sb.append(text[i])
            i++
        }
        sb.append("\n")
    }
}
