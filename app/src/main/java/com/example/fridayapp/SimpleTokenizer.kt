package com.example.fridayapp

import android.content.Context
import java.util.Locale

class SimpleTokenizer(context: Context) {
    private val vocab = HashMap<String, Long>()
    private val unkId: Long
    private val clsId: Long
    private val sepId: Long

    init {
        // Load vocab.txt from assets
        // Make sure 'vocab.txt' is in app/src/main/assets/
        try {
            context.assets.open("vocab.txt").bufferedReader().useLines { lines ->
                lines.forEachIndexed { index, line ->
                    if (line.isNotBlank()) {
                        vocab[line.trim()] = index.toLong()
                    }
                }
            }
        } catch (e: Exception) {
            e.printStackTrace()
        }

        // Standard BERT IDs
        unkId = vocab["[UNK]"] ?: 100
        clsId = vocab["[CLS]"] ?: 101
        sepId = vocab["[SEP]"] ?: 102
    }

    fun tokenize(text: String): LongArray {
        val tokens = ArrayList<Long>()
        tokens.add(clsId) // Start of Sequence

        // 1. Basic Normalization: Lowercase + Punctuation Split
        // This regex puts spaces around anything that isn't a letter or number
        val normalized = text.lowercase(Locale.ROOT)
            .replace(Regex("([^a-z0-9\\s])"), " $1 ")

        // 2. Split by whitespace to get "Basic Tokens"
        val basicTokens = normalized.split("\\s+".toRegex()).filter { it.isNotEmpty() }

        // 3. WordPiece Tokenization (The greedy match loop)
        for (token in basicTokens) {
            // If the whole word exists, use it immediately
            if (vocab.containsKey(token)) {
                tokens.add(vocab[token]!!)
                continue
            }

            // Otherwise, break it down
            var start = 0
            var isBad = false
            val subTokens = ArrayList<Long>()

            while (start < token.length) {
                var end = token.length
                var curSubTokenStr = ""
                var found = false

                // Greedy Longest-Match: Try to find the longest substring from 'start'
                while (end > start) {
                    val subStr = token.substring(start, end)
                    // If it's not the start of the word, prepend "##"
                    val lookUp = if (start > 0) "##$subStr" else subStr

                    if (vocab.containsKey(lookUp)) {
                        curSubTokenStr = lookUp
                        found = true
                        break
                    }
                    end--
                }

                if (found) {
                    subTokens.add(vocab[curSubTokenStr]!!)
                    start = end
                } else {
                    isBad = true
                    break
                }
            }

            if (isBad) {
                tokens.add(unkId)
            } else {
                tokens.addAll(subTokens)
            }
        }

        tokens.add(sepId) // End of Sequence
        return tokens.toLongArray()
    }
}