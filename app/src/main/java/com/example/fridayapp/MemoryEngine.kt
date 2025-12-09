package com.example.fridayapp

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.util.Collections
import kotlin.math.sqrt

class EmbeddingEngine(context: Context) {
    private var env: OrtEnvironment = OrtEnvironment.getEnvironment()
    private var session: OrtSession? = null
    private val tokenizer = SimpleTokenizer(context)

    // Simple Vocabulary (You should actually load vocab.txt from assets for accuracy)
    // For this example, we will assume you can load a vocab map or hash user input
    // To make this work 100% effectively, you need to copy 'vocab.txt' to assets too.

    init {
        // Load the model from raw/assets
        val modelBytes = context.assets.open("all_MiniLM_L6_v2.onnx").readBytes()
        session = env.createSession(modelBytes)
    }

    fun getEmbedding(text: String): FloatArray {

        // 1. Tokenize (Placeholder for brevity - usually requires vocab lookup)
        // In a real app, use a dedicated tokenizer library.
        // Here we create a dummy input to prevent crash if you don't have the tokenizer code yet.
        // YOU MUST IMPLEMENT A PROPER TOKENIZER OR PASS RAW IDS FROM PYTHON FOR EXACT MATCH

        // Input shapes for MiniLM: input_ids, attention_mask, token_type_ids
        // Sizes must match. For now, we return a zero vector to show pipeline connection
        // unless you implement the WordPiece tokenizer.

        // TODO: Plug in a Kotlin BERT Tokenizer here.
        // For now, let's assume we send a dummy request just to show ONNX works.

        val tokenIds = tokenizer.tokenize(text)
        val seqLen = tokenIds.size.toLong()
        val shape = longArrayOf(1, seqLen)

        val inputIdsFn = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(tokenIds), shape)
        val maskFn = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(LongArray(tokenIds.size) { 1 }), shape)
        val typeIdsFn = OnnxTensor.createTensor(env, java.nio.LongBuffer.wrap(LongArray(tokenIds.size) { 0 }), shape)

        val inputs = mapOf(
            "input_ids" to inputIdsFn,
            "attention_mask" to maskFn,
            "token_type_ids" to typeIdsFn
        )

        val output = session?.run(inputs)

        // Extract 384-dim vector (Mean pooling usually required if model doesn't do it)
        // Assuming your ONNX model output[0] is [1, 384]
        @Suppress("UNCHECKED_CAST")
        val outputTensor = output?.get(0)?.value as Array<Array<FloatArray>>
        val tokenVectors = outputTensor[0]// Adjust based on your specific ONNX export shape

        val averagedVector = FloatArray(384)

        for (vector in tokenVectors) {
            for (i in 0 until 384) {
                averagedVector[i] += vector[i]
            }
        }

        // Divide by number of tokens
        val count = tokenVectors.size.toFloat()
        for (i in 0 until 384) {
            averagedVector[i] /= count
        }

        // 6. NORMALIZE (Crucial for Cosine Similarity)
        // Python's encode(normalize_embeddings=True) does this.
        return normalize(averagedVector)
    }

    private fun normalize(v: FloatArray): FloatArray {
        var magnitude = 0f
        for (x in v) {
            magnitude += x * x
        }
        magnitude = sqrt(magnitude)

        if (magnitude > 1e-9) {
            for (i in v.indices) {
                v[i] /= magnitude
            }
        }
        return v
    }
}