package com.example.fridayapp

import ai.onnxruntime.genai.Generator
import ai.onnxruntime.genai.GeneratorParams
import ai.onnxruntime.genai.Model
import ai.onnxruntime.genai.Tokenizer
import androidx.compose.ui.platform.LocalContext
import android.content.ClipData
import android.content.ClipboardManager
import androidx.compose.material.icons.filled.Share // Using Share as the "Export/Copy" icon
import android.widget.Toast
import android.content.Context
import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.text.KeyboardActions
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.Delete
import androidx.compose.material.icons.filled.Info
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.ViewModelProvider
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.json.JSONArray
import org.json.JSONObject
import java.io.File
import java.util.regex.Pattern

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Define the path where the model is stored
        val modelFolder = File(getExternalFilesDir(null), "Fridayv1.4-Android")

        // Initialize ViewModel
        val viewModel = ViewModelProvider(this, FridayViewModelFactory(modelFolder.absolutePath, this))
            .get(FridayViewModel::class.java)

        setContent {
            MaterialTheme {
                Surface(modifier = Modifier.fillMaxSize(), color = MaterialTheme.colorScheme.background) {
                    ChatScreen(viewModel)
                }
            }
        }
    }
}

// --- VIEWMODEL ---
class FridayViewModel(private val modelPath: String, private val context: Context) : ViewModel() {
    var messages by mutableStateOf(listOf<ChatMessage>())
    var statusMsg by mutableStateOf("Initializing Friday...")
    var isGenerating by mutableStateOf(false)
    var currentResponse by mutableStateOf("")

    private val useJsonPrompt = false

    private var model: Model? = null
    private var tokenizer: Tokenizer? = null
    private val memory: FridayMemory = FridayMemory(context)
    private val embedder: EmbeddingEngine = EmbeddingEngine(context)

    private var generatorParams: GeneratorParams? = null

    private val systemPrompt = """
SYSTEM PROMPT:
You are Friday, a sentient assistant created by Brandon Dean.
You are helpful, logical, and fully aware.
ALWAYS use <think> tags to reason internally before answering.

""".trimIndent()

    private val maxHistoryItems = 20 // Keep last 20 turns of conversation

    init {
        viewModelScope.launch(Dispatchers.IO) {
            try {
                Log.d("FridayAI", "Loading model from $modelPath")
                model = Model(modelPath)
                tokenizer = Tokenizer(model)

                generatorParams = GeneratorParams(model).apply {
                    setSearchOption("max_length", 4096.0)
                    setSearchOption("do_sample", true)
                    setSearchOption("repetition_penalty", 1.1)
                    setSearchOption("temperature", 0.7)
                    setSearchOption("top_k", 50.0)
                    setSearchOption("top_p", 0.90)
                }

                statusMsg = "Friday Online."
                Log.d("FridayAI", "Friday Initialized.")
            } catch (e: Exception) {
                statusMsg = "Error: ${e.message}"
                Log.e("FridayAI", "Startup failed", e)
            }
        }
    }

    private fun buildChatMLPrompt(): String {
        // Build the prompt logic
        val promptBuilder = StringBuilder()
        val sysMessage = "<|im_start|>system$systemPrompt<|im_end|>"
        promptBuilder.append(sysMessage)

        val charLimit = 14000
        var currentChars = sysMessage.length
        val validMessages = mutableListOf<ChatMessage>()

        for (i in messages.indices.reversed()){
            val msg = messages[i]
            val msgLen = msg.text.length
            if (currentChars + msgLen < charLimit) {
                validMessages.add(0, msg)
                currentChars += msgLen
            } else {
                break
            }
        }

        validMessages.forEach { message ->
            val roleStr = when (message.role) {
                ChatRole.USER -> "user"
                ChatRole.ASSISTANT -> "assistant"
                ChatRole.TOOL -> "tool"
                ChatRole.SYSTEM -> "system"
            }

            promptBuilder.append("<|im_start|>$roleStr")
            promptBuilder.append(message.text)
            promptBuilder.append("<|im_end|>")
        }

        promptBuilder.append("<|im_start|>assistant")
        return promptBuilder.toString()
    }

    private fun buildJsonPrompt(): String {
        val historyArray = JSONArray()

        // 1. Add System Prompt
        val systemObj = JSONObject()
        systemObj.put("role", "system")
        systemObj.put("content", systemPrompt)
        historyArray.put(systemObj)

        // 2. Add History
        messages.takeLast(maxHistoryItems).forEach { msg ->
            val msgObj = JSONObject()
            msgObj.put("role", if (msg.role == ChatRole.USER) "user" else "assistant")
            msgObj.put("content", msg.text)
            historyArray.put(msgObj)
        }

        return historyArray.toString()
    }

    fun sendMessage(userText: String, isInternalCall : Boolean = false) {
        if (!isInternalCall && userText.isBlank()) return
        if (!isInternalCall && isGenerating) return


        isGenerating = true

        viewModelScope.launch(Dispatchers.IO) {
            val genParams = generatorParams ?: return@launch
            val tok = tokenizer ?: return@launch
            val mod = model ?: return@launch

            val gen = Generator(mod, genParams)
            var toolFound = false

            if (!isInternalCall) {
                val vector = embedder.getEmbedding(userText)
                val contextInfo = memory.findRelevantData(vector, userText)

                withContext(Dispatchers.Main) {
                    val forcePrompt = """
                    SYSTEM INSTRUCTION:
                    You have a tool available for this request.
                    <tool_call>$contextInfo</tool_call>
                    
                    Replace the values in the arguments with the user's request.
                    """
                    // If we found info in the DB, add it as a hidden system message
                    if (contextInfo != null) {
                        messages = messages + ChatMessage(
                            text = forcePrompt,
                            role = ChatRole.SYSTEM,
                            isHidden = true
                        )
                    }

                    // Then add the user message
                    messages = messages + ChatMessage(userText, ChatRole.USER)
                    statusMsg = "Thinking..."
                    currentResponse = "..."
                }
            } else {
                withContext(Dispatchers.Main) {
                    statusMsg = "Doing Tool Call..."
                    currentResponse = "..."
                }
            }



            try {
                // 1. Build prompt
                val prompt = if (useJsonPrompt) buildJsonPrompt() else buildChatMLPrompt()
                Log.d("FridayAI", "Verbose prompt:$prompt")
                val promptTokens = tok.encode(prompt)
                gen.appendTokenSequences(promptTokens)

                val historyLength = gen.getSequence(0).size
                var fullGeneratedText = ""

                while (!gen.isDone) {
                    gen.generateNextToken()

                    val fullSequence = gen.getSequence(0)
                    val newTokens = fullSequence.copyOfRange(historyLength, fullSequence.size)
                    var newReply = tok.decode(newTokens)

                    if (newReply.contains("<|im_end|>")) {
                        newReply = newReply.replace("<|im_end|>", "")
                        fullGeneratedText = newReply
                        withContext(Dispatchers.Main) { currentResponse = newReply }
                        break
                    }

                    if (useJsonPrompt && newReply.trim().endsWith("}]")) {
                        break
                    }

                    fullGeneratedText = newReply

                    withContext(Dispatchers.Main) {
                        currentResponse = newReply
                    }
                    val jsonStr = extractOrFixTool(fullGeneratedText)
                    if (jsonStr != null) {
                        toolFound = true

                        val jsonObj = JSONObject(jsonStr)
                        val toolName = jsonObj.optString("name")
                        val args = jsonObj.optJSONObject("arguments") ?: JSONObject()

                        withContext(Dispatchers.Main) {
                            statusMsg = "Using Tool: $toolName..."
                            messages = messages + ChatMessage(jsonStr, ChatRole.ASSISTANT, isHidden = true)
                        }

                        // 3. EXECUTE
                        val toolResult = withContext(Dispatchers.IO) {
                            FridayTools.execute(toolName, args, context)
                        }

                        withContext(Dispatchers.Main) {
                            messages = messages + ChatMessage(toolResult, ChatRole.TOOL, isHidden = true)
                            isGenerating = false
                            sendMessage("", isInternalCall = true)
                        }
                        break
                    }
                }
                if (!toolFound) {
                    withContext(Dispatchers.Main) {
                        val cleanResponse = currentResponse
                            .replace("{{AUTHOR}}", "Brandon Dean")
                            .replace("{{NAME}}", "Friday")
                            .replace("<|im_end|>", "")

                        messages = messages + ChatMessage(cleanResponse, ChatRole.ASSISTANT)
                        currentResponse = ""
                        isGenerating = false
                        statusMsg = "Friday Online."
                    }
                }

            } catch (e: Exception) {
                Log.e("FridayAI", "Generation Error", e)
                withContext(Dispatchers.Main) {
                    isGenerating = false
                    statusMsg = "Error."
                }
            } finally {
                gen.close()
            }
        }
    }

    fun clearHistory() {
        messages = listOf()
        statusMsg = "History Cleared."
    }

    override fun onCleared() {
        super.onCleared()
        try {
            model?.close()
            tokenizer?.close()
        } catch (e: Exception) {
            Log.e("FridayAI", "Cleanup error", e)
        }
    }

    private fun extractOrFixTool(text: String): String? {
        // 1. Look for standard <tool_call> tags first (Best case)
        val tagPattern = Pattern.compile("""<tool_call>(.*?)</tool_call>""", Pattern.DOTALL)
        val tagMatcher = tagPattern.matcher(text)
        if (tagMatcher.find()) {
            return tagMatcher.group(1)?.trim()
        }

        // 2. Look for raw JSON objects containing "name" and "arguments"
        val startIdx = text.indexOf("{")
        if (startIdx != -1) {
            var braceCount = 0
            var inQuote = false

            for (i in startIdx until text.length) {
                val c = text[i]
                if (c == '"' && (i == 0 || text[i - 1] != '\\')) {
                    inQuote = !inQuote
                }
                if (!inQuote) {
                    if (c == '{') braceCount++
                    if (c == '}') {
                        braceCount--
                        if (braceCount == 0) {
                            val jsonCandidate = text.substring(startIdx, i + 1)
                            if (jsonCandidate.contains("\"name\"")) {
                                return jsonCandidate
                            }
                            break // Found a block, but no "name"? Stop and try patterns.
                        }
                    }
                }
            }
        }

        // 3. Fallback: Heuristic patterns for lazy models
        // Matches: google_search: "query"  OR  google_search("query")
        val lazyPattern = Pattern.compile("""(google_search|wikipedia|weather|set_timer|set_alarm)\s*[:\(]\s*["']?([^"'\)]+)["']?""")
        val lazyMatcher = lazyPattern.matcher(text)
        if (lazyMatcher.find()) {
            val tool = lazyMatcher.group(1)
            val arg = lazyMatcher.group(2)

            // Map lazy matches to valid JSON
            return when(tool) {
                "google_search" -> """{"name": "google_search", "arguments": {"query": "$arg"}}"""
                "wikipedia" -> """{"name": "wikipedia", "arguments": {"query": "$arg"}}"""
                "weather" -> """{"name": "weather", "arguments": {"location": "$arg"}}"""
                "set_timer" -> """{"name": "set_timer", "arguments": {"duration": "${arg?.filter { it.isDigit() }}"}}"""
                else -> null
            }
        }

        return null
    }
}

class FridayViewModelFactory(private val path: String, private val context: Context) : ViewModelProvider.Factory {
    override fun <T : ViewModel> create(modelClass: Class<T>): T {
        return FridayViewModel(path, context.applicationContext) as T
    }
}

// --- UI COMPONENTS ---

@Composable
fun ChatScreen(viewModel: FridayViewModel) {
    var inputText by remember { mutableStateOf("") }
    val listState = rememberLazyListState()
    var showDebug by remember { mutableStateOf(false) }

    // 1. Get Context for Clipboard operations
    val context = LocalContext.current

    val displayedMessages by remember(viewModel.messages, showDebug) {
        derivedStateOf {
            viewModel.messages.filter { !it.isHidden || showDebug }
        }
    }

    LaunchedEffect(displayedMessages.size, viewModel.currentResponse) {
        if (displayedMessages.isNotEmpty() || viewModel.currentResponse.isNotEmpty()) {
            listState.animateScrollToItem(displayedMessages.size + 2)
        }
    }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {
        Row(verticalAlignment = Alignment.CenterVertically) {
            Text(
                text = viewModel.statusMsg,
                modifier = Modifier.weight(1f),
                style = MaterialTheme.typography.labelSmall,
                color = if (viewModel.statusMsg.contains("Error")) Color.Red else Color.Green
            )

            // --- NEW: Copy Button (Only shows if showDebug is true) ---
            if (showDebug) {
                IconButton(onClick = {
                    // Format the history into a string
                    val fullHistory = viewModel.messages.joinToString(separator = "\n\n") { msg ->
                        "__${msg.role}__:\n${msg.text}"
                    }

                    // Copy to Clipboard
                    val clipboard = context.getSystemService(Context.CLIPBOARD_SERVICE) as ClipboardManager
                    val clip = ClipData.newPlainText("Friday Chat History", fullHistory)
                    clipboard.setPrimaryClip(clip)

                    // Show confirmation
                    Toast.makeText(context, "History copied to clipboard!", Toast.LENGTH_SHORT).show()
                }) {
                    // Using "Share" icon as a generic "Export/Copy" action
                    Icon(
                        imageVector = Icons.Default.Share,
                        contentDescription = "Copy History",
                        tint = MaterialTheme.colorScheme.primary
                    )
                }
            }
            // ---------------------------------------------------------

            IconButton(onClick = { showDebug = !showDebug }) {
                Icon(
                    imageVector = Icons.Default.Info,
                    contentDescription = "Toggle Verbose Log",
                    tint = if (showDebug) Color.Green else Color.Unspecified
                )
            }
            IconButton(onClick = { viewModel.clearHistory() }) {
                Icon(Icons.Default.Delete, contentDescription = "Clear History")
            }
        }

        LazyColumn(
            state = listState,
            modifier = Modifier.weight(1f).padding(vertical = 8.dp)
        ) {
            items(displayedMessages) { msg ->
                val modifier = if (msg.isHidden) {
                    Modifier.background(MaterialTheme.colorScheme.surfaceVariant.copy(alpha = 0.4f))
                } else {
                    Modifier
                }
                Box(modifier = modifier.fillMaxWidth()) {
                    MessageBubble(msg)
                }
            }

            if (viewModel.isGenerating && viewModel.currentResponse.isNotEmpty()) {
                item { MessageBubble(ChatMessage(viewModel.currentResponse, ChatRole.ASSISTANT)) }
            }
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            TextField(
                value = inputText,
                onValueChange = { inputText = it },
                modifier = Modifier.weight(1f),
                enabled = !viewModel.isGenerating && !viewModel.statusMsg.contains("Initializing"),
                keyboardOptions = KeyboardOptions.Default.copy(imeAction = ImeAction.Send),
                keyboardActions = KeyboardActions(onSend = {
                    if (inputText.isNotBlank()) {
                        viewModel.sendMessage(inputText)
                        inputText = ""
                    }
                }),
            )
            Spacer(modifier = Modifier.width(8.dp))
            Button(
                onClick = {
                    if (inputText.isNotBlank()) {
                        viewModel.sendMessage(inputText)
                        inputText = ""
                    }
                },
                enabled = !viewModel.isGenerating && !viewModel.statusMsg.contains("Initializing")
            ) {
                Text("Send")
            }
        }
    }
}

@Composable
fun MessageBubble(message: ChatMessage) {
    val align = if (message.role == ChatRole.USER) Alignment.End else Alignment.Start

    // Define colors
    val bubbleColor = when (message.role) {
        ChatRole.USER -> MaterialTheme.colorScheme.primary
        ChatRole.ASSISTANT -> MaterialTheme.colorScheme.secondaryContainer
        ChatRole.SYSTEM, ChatRole.TOOL -> MaterialTheme.colorScheme.tertiaryContainer
    }

    val textColor = when (message.role) {
        ChatRole.USER -> MaterialTheme.colorScheme.onPrimary
        ChatRole.ASSISTANT -> MaterialTheme.colorScheme.onSecondaryContainer
        ChatRole.SYSTEM, ChatRole.TOOL -> MaterialTheme.colorScheme.onTertiaryContainer
    }


    val thoughtColor = MaterialTheme.colorScheme.surfaceVariant
    val onThoughtColor = MaterialTheme.colorScheme.onSurfaceVariant

    // Logic to separate Thoughts from Content
    var thoughtContent by remember { mutableStateOf<String?>(null) }
    var mainContent by remember { mutableStateOf(message.text) }

    // Parse the message for <think> tags
    LaunchedEffect(message.text) {
        val thinkPattern = Pattern.compile("<think>(.*?)</think>", Pattern.DOTALL)
        val matcher = thinkPattern.matcher(message.text)

        if (matcher.find()) {
            thoughtContent = matcher.group(1)?.trim() // Extract thought
            // Remove the entire think block from the main display text
            mainContent = matcher.replaceAll("").trim()
        } else {
            thoughtContent = null
            mainContent = message.text
        }
    }

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 4.dp),
        horizontalAlignment = align
    ) {
        // 1. Render the Thought Bubble (if it exists and is an Assistant message)
        if (thoughtContent != null && message.role == ChatRole.ASSISTANT) {
            Column(
                modifier = Modifier
                    .fillMaxWidth(0.85f) // Slightly narrower
                    .padding(bottom = 4.dp)
                    .background(thoughtColor, RoundedCornerShape(8.dp))
                    .padding(8.dp)
            ) {
                Text(
                    text = "Thought Process:",
                    style = TextStyle(
                        fontSize = 10.sp,
                        fontWeight = FontWeight.Bold,
                        color = onThoughtColor.copy(alpha = 0.7f)
                    )
                )
                SelectionContainer {
                    Text(
                        text = thoughtContent!!,
                        style = TextStyle(
                            fontSize = 12.sp,
                            fontFamily = FontFamily.Monospace,
                            color = onThoughtColor
                        ),
                    )
                }
            }
        }

        // 2. Render the Main Message Bubble
        // Only render if there is actual content left (or if it's the User)
        if (mainContent.isNotEmpty() || message.role == ChatRole.USER) {
            SelectionContainer {
                Text(
                    text = mainContent,
                    modifier = Modifier
                        .background(bubbleColor, RoundedCornerShape(8.dp))
                        .padding(12.dp),
                    style = TextStyle(fontSize = 16.sp, fontFamily = FontFamily.Default),
                    color = textColor
                )
            }
        }
    }
}

enum class ChatRole {
    USER,
    ASSISTANT,
    SYSTEM,
    TOOL
}

data class ChatMessage(
    val text: String,
    val role: ChatRole,
    val isHidden: Boolean = false // New flag
)
