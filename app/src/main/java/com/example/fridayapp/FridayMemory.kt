package com.example.fridayapp

import android.content.Context
import android.database.sqlite.SQLiteDatabase
import java.io.File
import java.io.FileOutputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import kotlin.math.sqrt

class FridayMemory(private val context: Context) {
    private val dbName = "friday_tools_android.db"
    private var db: SQLiteDatabase? = null

    init {
        val dbFile = context.getDatabasePath(dbName)
        if (!dbFile.exists()) {
            dbFile.parentFile?.mkdirs()
            context.assets.open(dbName).use { inputStream ->
                FileOutputStream(dbFile).use { outputStream ->
                    inputStream.copyTo(outputStream)
                }
            }
        }
        db = SQLiteDatabase.openDatabase(dbFile.absolutePath, null, SQLiteDatabase.OPEN_READWRITE)

    }

    fun findRelevantData(userVector: FloatArray, userText: String): String? {
        val database = db ?: return null
        var bestContent: String? = null
        var highestScore = -1.0f

        // 1. HYBRID FILTER STEP
        // Check if the user's text contains any of our "Hard Tags"
        // This helps disambiguate similar concepts (Time vs Timer)
        val lowerText = userText.lowercase()

        // We will build a dynamic query.
        // If we find a tag, we ONLY select rows matching that tag.
        // If no tag is found, we select ALL rows (pure vector search).

        var selectionQuery = "SELECT content, embedding FROM tools"
        var selectionArgs: Array<String>? = null

        // Simple map of known tags (Hardcoded for speed, or query DB for unique tags)
        val knownTags = listOf("timer", "alarm", "weather", "camera", "flashlight", "battery", "wiki", "google", "url")

        for (tag in knownTags) {
            if (lowerText.contains(tag)) {
                // Found a specific intent! Narrow the search.
                selectionQuery = "SELECT content, embedding FROM tools WHERE filter_tag = ?"
                selectionArgs = arrayOf(tag)
                break // Stop after finding the first strong match
            }
        }

        // 2. RUN QUERY (Either filtered or full)
        val cursor = database.rawQuery(selectionQuery, selectionArgs)

        if (cursor.moveToFirst()) {
            do {
                val content = cursor.getString(0)
                val blob = cursor.getBlob(1)
                val dbVector = blobToFloats(blob)

                val score = cosineSimilarity(userVector, dbVector)

                // Lower the threshold slightly since we might have already filtered relevant ones
                if (score > 0.35f && score > highestScore) {
                    highestScore = score
                    bestContent = content
                }
            } while (cursor.moveToNext())
        }
        cursor.close()

        return bestContent
    }

    private fun cosineSimilarity(v1: FloatArray, v2: FloatArray): Float {
        var dot = 0f
        var mag1 = 0f
        var mag2 = 0f

        for (i in v1.indices) {
            dot += v1[i] * v2[i]
            mag1 += v1[i] * v1[i]
            mag2 += v2[i] * v2[i]
        }
        return dot / (sqrt(mag1) * sqrt(mag2))
    }

    private fun blobToFloats(blob: ByteArray): FloatArray {
        val buffer = ByteBuffer.wrap(blob).order(ByteOrder.LITTLE_ENDIAN)
        val arr = FloatArray(blob.size / 4)
        buffer.asFloatBuffer().get(arr)
        return arr
    }

}
