package com.example.llama

import android.content.Intent
import android.graphics.Typeface
import android.net.Uri
import android.text.Spannable
import android.text.SpannableString
import android.text.SpannableStringBuilder
import android.text.TextPaint
import android.text.style.BackgroundColorSpan
import android.text.style.ClickableSpan
import android.text.style.RelativeSizeSpan
import android.text.style.StyleSpan
import android.text.style.TypefaceSpan
import android.view.View
import android.widget.Toast

object MarkdownRenderer {

    fun render(text: String, linkColor: Int = DEFAULT_LINK_COLOR): Spannable {
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
                appendInline(sb, trimmed.substring(2), linkColor)
            } else if (trimmed.isEmpty()) {
                sb.append("\n")
            } else {
                appendInline(sb, trimmed, linkColor)
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

    private fun appendInline(sb: SpannableStringBuilder, text: String, linkColor: Int) {
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
            if (text[i] == '[') {
                val parsed = parseLink(text, i, linkColor)
                if (parsed != null) {
                    val start = sb.length
                    sb.append(parsed.label)
                    sb.setSpan(parsed.span, start, sb.length, Spannable.SPAN_EXCLUSIVE_EXCLUSIVE)
                    i = parsed.end
                    continue
                }
            }
            sb.append(text[i])
            i++
        }
        sb.append("\n")
    }

    private fun parseLink(text: String, start: Int, linkColor: Int): ParsedLink? {
        val closeBracket = text.indexOf(']', start + 1)
        if (closeBracket <= start + 1) return null
        if (closeBracket + 1 >= text.length || text[closeBracket + 1] != '(') return null
        val closeParen = text.indexOf(')', closeBracket + 2)
        if (closeParen <= closeBracket + 1) return null
        val label = text.substring(start + 1, closeBracket)
        if (label.isBlank()) return null
        val target = normalizeUrl(text.substring(closeBracket + 2, closeParen).trim()) ?: return null
        return ParsedLink(label, LinkSpan(target, linkColor), closeParen + 1)
    }

    private fun normalizeUrl(raw: String): String? = when {
        raw.startsWith("http://") || raw.startsWith("https://") -> raw
        raw.startsWith("www.") -> "https://$raw"
        else -> null
    }

    private data class ParsedLink(val label: String, val span: LinkSpan, val end: Int)

    private class LinkSpan(private val url: String, private val color: Int) : ClickableSpan() {
        override fun onClick(widget: View) {
            val intent = Intent(Intent.ACTION_VIEW, Uri.parse(url))
            runCatching { widget.context.startActivity(intent) }
                .onFailure {
                    Toast.makeText(widget.context, R.string.link_open_failed, Toast.LENGTH_SHORT).show()
                }
        }

        override fun updateDrawState(ds: TextPaint) {
            ds.color = color
            ds.isUnderlineText = true
        }
    }

    private const val DEFAULT_LINK_COLOR = 0xFF315F9A.toInt()
}
