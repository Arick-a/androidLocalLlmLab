package com.arick.androidlocalllmlab

import android.app.Application
import android.net.Uri
import android.os.Debug
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

// Compose 只渲染这个状态，不直接管理 Native Runtime 或文件 I/O。
sealed interface ChatUiState {
    data object Idle : ChatUiState

    data class Preparing(
        val message: String
    ) : ChatUiState

    data class Ready(
        val modelFile: File,
        val nCtx: Int,
        val modelDescription: String,
        val modelName: String
    ) : ChatUiState

    data class Error(
        val message: String
    ) : ChatUiState
}

enum class MessageRole {
    System,
    User,
    Assistant
}

data class ChatMessage(
    val role: MessageRole,
    val content: String,
    // reasoning 不直接混入普通回答；展示层可将它渲染为可折叠区域。
    val reasoningContent: String = "",
    // 展开状态属于会话 UI 数据，不能交给 Composable remember，否则流式内容变化或页面切换会丢失。
    val isReasoningExpanded: Boolean = true,
    // 从首个思考 Token 到正文开始的真实流式耗时。
    val reasoningDurationMillis: Long? = null,
    // 工具执行记录只来自 Kotlin 的实际调用，不展示模型自行生成的“思考”。
    val agentTrace: List<AgentTraceItem> = emptyList(),
    val isAgentTraceExpanded: Boolean = false,
    val agentDurationMillis: Long? = null,
    // 搜索来源由工具结果提供，不能依赖模型在正文中自行拼接超长 URL。
    val sourceLinks: List<SourceLink> = emptyList()
)

data class AgentTraceItem(val text: String)

data class MemorySnapshot(
    val javaUsedBytes: Long,
    val javaMaxBytes: Long,
    val nativeAllocatedBytes: Long,
    val totalPssKb: Int,
    val nativePssKb: Int
)

data class RuntimeMemorySnapshots(
    val beforeLoad: MemorySnapshot? = null,
    val afterModelLoaded: MemorySnapshot? = null,
    val afterContextCreated: MemorySnapshot? = null,
    val afterGeneration: MemorySnapshot? = null,
    val afterModelReleased: MemorySnapshot? = null
) {
    val current: MemorySnapshot?
        get() = afterModelReleased ?: afterGeneration ?: afterContextCreated ?: afterModelLoaded ?: beforeLoad
}

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private companion object {
        val DEFAULT_CPU_THREADS = Runtime.getRuntime().availableProcessors().coerceIn(1, 8)
    }

    // Runtime 持有 JNI 返回的 nativeHandle，生命周期与 ViewModel 一致。
    private val runtime = LocalLlmRuntime()
    private val toolExecutor = AndroidToolExecutor(application)
    private val modelFileImporter = ModelFileImporter(application)
    private val selectedModelStore = SelectedModelStore(application)

    private val _uiState = MutableStateFlow<ChatUiState>(ChatUiState.Idle)
    val uiState: StateFlow<ChatUiState> = _uiState.asStateFlow()

    private val _messages = MutableStateFlow<List<ChatMessage>>(emptyList())
    val messages: StateFlow<List<ChatMessage>> = _messages.asStateFlow()

    // System Prompt 是当前会话配置，不作为普通聊天气泡渲染。
    private val _systemPrompt = MutableStateFlow("")
    val systemPrompt: StateFlow<String> = _systemPrompt.asStateFlow()

    // 由 Native 在应用 Chat Template 后回调，反映模型真正收到的文本。
    private val _finalPrompt = MutableStateFlow("")
    val finalPrompt: StateFlow<String> = _finalPrompt.asStateFlow()

    private val _promptTokenCount = MutableStateFlow(0)
    val promptTokenCount: StateFlow<Int> = _promptTokenCount.asStateFlow()

    private val _trimmedHistoryTurnCount = MutableStateFlow(0)
    val trimmedHistoryTurnCount: StateFlow<Int> = _trimmedHistoryTurnCount.asStateFlow()

    // 4096 是允许 2048 输出的最低实用档；更长多轮对话由设置页按设备内存升到 8K 以上。
    private val _requestedNctx = MutableStateFlow(4096)
    val requestedNctx: StateFlow<Int> = _requestedNctx.asStateFlow()

    // 当前实验真机已验证 8 线程更快；仍限制为设备实际可用核心数，设置页可继续做对照实验。
    private val _cpuThreads = MutableStateFlow(DEFAULT_CPU_THREADS)
    val cpuThreads: StateFlow<Int> = _cpuThreads.asStateFlow()

    private val _inferenceConfig = MutableStateFlow(InferenceConfig())
    val inferenceConfig: StateFlow<InferenceConfig> = _inferenceConfig.asStateFlow()

    private val _thinkingEnabled = MutableStateFlow(true)
    val thinkingEnabled: StateFlow<Boolean> = _thinkingEnabled.asStateFlow()

    private val _generationProgress = MutableStateFlow<String?>(null)
    val generationProgress: StateFlow<String?> = _generationProgress.asStateFlow()

    private val _generationMetrics = MutableStateFlow<GenerationMetrics?>(null)
    val generationMetrics: StateFlow<GenerationMetrics?> = _generationMetrics.asStateFlow()

    private val _modelPreparationMetrics = MutableStateFlow<ModelPreparationMetrics?>(null)
    val modelPreparationMetrics: StateFlow<ModelPreparationMetrics?> = _modelPreparationMetrics.asStateFlow()

    private val _memorySnapshots = MutableStateFlow(RuntimeMemorySnapshots())
    val memorySnapshots: StateFlow<RuntimeMemorySnapshots> = _memorySnapshots.asStateFlow()

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

    private val _isStopRequested = MutableStateFlow(false)
    val isStopRequested: StateFlow<Boolean> = _isStopRequested.asStateFlow()

    private val _generationError = MutableStateFlow<String?>(null)
    val generationError: StateFlow<String?> = _generationError.asStateFlow()

    init {
        // filesDir 中的模型在 App 重启后仍存在，因此可以在新页面自动恢复。
        restoreLastModel()
    }

    fun loadModel(uri: Uri) {
        if (_uiState.value is ChatUiState.Preparing) {
            return
        }

        viewModelScope.launch {
            try {
                _uiState.value = ChatUiState.Preparing("正在导入模型文件…")
                val modelFile = modelFileImporter.importModel(uri)

                // 只有模型和 Context 都成功创建后，才记录为下次自动恢复的模型。
                prepareModel(modelFile)
                selectedModelStore.save(modelFile)
            } catch (error: Throwable) {
                _uiState.value = ChatUiState.Error(
                    error.message ?: "模型加载失败"
                )
            }
        }
    }

    private fun restoreLastModel() {
        val modelFile = selectedModelStore.selectedModelFile() ?: return
        if (!modelFile.isFile) {
            // App 数据被清理或文件被删除时，避免每次进入页面都尝试无效路径。
            selectedModelStore.clear()
            return
        }

        viewModelScope.launch {
            try {
                _uiState.value = ChatUiState.Preparing("正在恢复上次模型…")
                prepareModel(modelFile)
            } catch (error: Throwable) {
                _uiState.value = ChatUiState.Error(
                    error.message ?: "恢复上次模型失败"
                )
            }
        }
    }

    private suspend fun prepareModel(modelFile: File) {
        _uiState.value = ChatUiState.Preparing("正在加载模型并创建 Context…")
        val beforeLoad = takeMemorySnapshot()
        var afterModelLoaded: MemorySnapshot? = null
        val preparationMetrics = withContext(Dispatchers.Default) {
            // Native 模型加载与 Context 分配可能耗时，不能阻塞 Compose 主线程。
            runtime.prepareModel(
                modelPath = modelFile.absolutePath,
                requestedNctx = _requestedNctx.value,
                cpuThreads = _cpuThreads.value,
                onModelLoaded = { afterModelLoaded = takeMemorySnapshot() }
            )
        }

        check(preparationMetrics.actualNctx > 0) { "模型或 Context 创建失败" }
        _modelPreparationMetrics.value = preparationMetrics
        _memorySnapshots.value = RuntimeMemorySnapshots(
            beforeLoad = beforeLoad,
            afterModelLoaded = afterModelLoaded,
            afterContextCreated = takeMemorySnapshot()
        )
        val modelDescription = withContext(Dispatchers.Default) {
            runtime.modelDescription()
        }
        val modelName = withContext(Dispatchers.Default) {
            runtime.modelMetadata("general.name")
        }
        // Qwen3-Instruct-2507 官方仅支持非思考模式，不能沿用上个模型的开关状态。
        if ("$modelName $modelDescription".contains("instruct-2507", ignoreCase = true)) {
            _thinkingEnabled.value = false
        }
        _messages.value = emptyList()
        _finalPrompt.value = ""
        _promptTokenCount.value = 0
        _trimmedHistoryTurnCount.value = 0
        _generationProgress.value = null
        _generationMetrics.value = null
        _generationError.value = null
        _uiState.value = ChatUiState.Ready(
            modelFile,
            preparationMetrics.actualNctx,
            modelDescription,
            modelName
        )
    }

    fun send(prompt: String) {
        if (
            prompt.isBlank() ||
            _uiState.value !is ChatUiState.Ready ||
            _isGenerating.value
        ) {
            return
        }

        // 消息历史是产品数据源；当前轮的空 AI 消息只用于接收流式输出，
        // 因而不会传给 Native 作为 prompt 的一部分。
        val visibleHistory = _messages.value.filter { it.content.isNotBlank() }
        val actualNctx = (_uiState.value as? ChatUiState.Ready)?.nCtx ?: return
        val systemPrompt = buildSystemPrompt(_systemPrompt.value.trim().takeIf { it.isNotEmpty() })
        val inferenceConfig = _inferenceConfig.value
        val nativeUserPrompt = prompt.withQwen3ThinkingInstruction()
        _messages.value = visibleHistory + ChatMessage(
            MessageRole.User,
            prompt
        ) + listOf(
            ChatMessage(MessageRole.Assistant, "")
        )
        _generationError.value = null
        _finalPrompt.value = ""
        _promptTokenCount.value = 0
        _trimmedHistoryTurnCount.value = 0
        _generationProgress.value = "正在准备 Prompt…"
        _generationMetrics.value = null
        // 必须在启动协程前清除上次的停止标记，否则下一轮会立即被旧请求取消。
        runtime.resetStopRequest()
        _isStopRequested.value = false
        _isGenerating.value = true
        val generationStartedAtNanos = System.nanoTime()

        viewModelScope.launch {
            try {
                val answer = withContext(Dispatchers.Default) {
                    val trimmedHistory = buildHistoryWithinContextBudget(
                        systemPrompt = systemPrompt,
                        visibleHistory = visibleHistory,
                        currentUserPrompt = nativeUserPrompt,
                        nCtx = actualNctx,
                        reservedOutputTokens = inferenceConfig.maxTokens
                    )
                    _trimmedHistoryTurnCount.value = trimmedHistory.droppedTurnCount
                    runFunctionCallingLoop(
                        initialMessages = trimmedHistory.messages,
                        inferenceConfig = inferenceConfig,
                        nCtx = actualNctx
                    )
                }
                if (answer.isBlank() && !_isStopRequested.value) {
                    error("模型没有返回内容")
                }
            } catch (error: Throwable) {
                _generationError.value = error.message ?: "生成失败"
            } finally {
                finishLatestAssistantAgentTrace(generationStartedAtNanos)
                _generationProgress.value = null
                _isGenerating.value = false
                _isStopRequested.value = false
            }
        }
    }

    private fun runFunctionCallingLoop(
        initialMessages: List<ChatMessage>,
        inferenceConfig: InferenceConfig,
        nCtx: Int
    ): String {
        var messagesForModel = initialMessages
        repeat(FunctionCalling.maxToolCalls + 1) { attempt ->
            val promptTokenCount = runtime.countChatPromptTokens(messagesForModel)
            val availableOutputTokens = nCtx - promptTokenCount
            check(availableOutputTokens >= 128) {
                "工具结果加入后可用输出空间仅剩 $availableOutputTokens tokens，请缩短问题或清空部分历史"
            }
            // 首轮历史裁剪仍按用户设置的 maxTokens 留足空间；工具结果加入后，
            // 最终回答按 Context 剩余空间自动收缩，不能因固定预留 2048 而直接失败。
            val effectiveInferenceConfig = inferenceConfig.copy(
                maxTokens = minOf(inferenceConfig.maxTokens, availableOutputTokens)
            )
            val generatedAnswer = generateForCurrentTurn(
                messages = messagesForModel,
                inferenceConfig = effectiveInferenceConfig
            )
            if (_isStopRequested.value) {
                return generatedAnswer
            }

            val toolCall = FunctionCalling.parseToolCall(generatedAnswer) ?: return generatedAnswer
            if (attempt == FunctionCalling.maxToolCalls) {
                error("工具调用次数达到上限")
            }

            // 工具 JSON 只是中间协议，不应留在聊天页；下一次生成继续使用同一个占位 AI 消息。
            replaceLatestAssistantMessage(AssistantResponseParts(reasoning = "", answer = ""), null)
            appendLatestAssistantAgentTrace(toolCall.traceDescription())
            _generationProgress.value = "正在执行工具：${toolCall.name}…"
            val toolResult = toolExecutor.execute(toolCall) { progress ->
                _generationProgress.value = progress
                appendLatestAssistantAgentTrace(progress)
            }
            val toolResultText = when (toolResult) {
                is ToolResult.Success -> {
                    appendLatestAssistantAgentTrace("工具执行成功：${toolCall.name}")
                    appendLatestAssistantSources(toolResult.sources)
                    toolResult.content
                }
                is ToolResult.Failure -> {
                    appendLatestAssistantAgentTrace("工具执行失败：${toolResult.message}")
                    "工具执行失败：${toolResult.message}"
                }
            }
            val generatedParts = splitAssistantResponse(generatedAnswer, templateStartsThinking = false)
            messagesForModel = messagesForModel + listOf(
                ChatMessage(
                    role = MessageRole.Assistant,
                    content = generatedParts.answer,
                    reasoningContent = generatedParts.reasoning
                ),
                ChatMessage(
                    MessageRole.User,
                    "工具执行结果：$toolResultText。请基于该结果正常回答用户；除非确有必要，否则不要再次调用工具。"
                )
            )
        }
        error("工具调用流程异常结束")
    }

    private fun generateForCurrentTurn(
        messages: List<ChatMessage>,
        inferenceConfig: InferenceConfig
    ): String {
        var assistantRawContent = ""
        var templateStartsThinking = false
        var shouldRenderAssistant = false
        var reasoningStartedAtNanos: Long? = null
        var reasoningFinishedAtNanos: Long? = null
        val generatedAnswer = runtime.generate(
            messages = messages,
            inferenceConfig = inferenceConfig,
            onPrompt = { finalPrompt, tokenCount ->
                _finalPrompt.value = finalPrompt
                _promptTokenCount.value = tokenCount
                _generationProgress.value = "正在理解 $tokenCount tokens…"
                templateStartsThinking = finalPrompt.trimEnd().endsWith("<think>")
            },
            onToken = { piece ->
                _generationProgress.value = "正在生成回复…"
                assistantRawContent += piece
                if (!shouldRenderAssistant) {
                    if (FunctionCalling.shouldDeferToolRendering(assistantRawContent)) {
                        return@generate
                    }
                    // 普通回答已被识别：把此前暂存的片段一次补上，之后继续流式刷新。
                    shouldRenderAssistant = true
                }
                val assistantParts = splitAssistantResponse(assistantRawContent, templateStartsThinking)
                val nowNanos = System.nanoTime()
                val durationMillis = assistantParts.reasoning.takeIf { it.isNotBlank() }?.let {
                    val startedAt = reasoningStartedAtNanos ?: nowNanos.also { timestamp ->
                        reasoningStartedAtNanos = timestamp
                    }
                    if (assistantParts.answer.isNotBlank() && reasoningFinishedAtNanos == null) {
                        reasoningFinishedAtNanos = nowNanos
                    }
                    ((reasoningFinishedAtNanos ?: nowNanos) - startedAt) / 1_000_000
                }
                replaceLatestAssistantMessage(assistantParts, durationMillis)
            },
            onMetrics = { metrics ->
                _generationMetrics.value = metrics
                _memorySnapshots.value = _memorySnapshots.value.copy(
                    afterGeneration = takeMemorySnapshot(),
                    afterModelReleased = null
                )
            }
        )
        // Native 返回完整 UTF-8；普通回答以它回填，避免临时流式片段遗留乱码。
        // 工具 JSON 只作为协议，不写入气泡，随后由 runFunctionCallingLoop 执行。
        if (FunctionCalling.parseToolCall(generatedAnswer) == null) {
            replaceLatestAssistantMessage(
                splitAssistantResponse(generatedAnswer, templateStartsThinking),
                null
            )
        }
        return generatedAnswer
    }

    private fun replaceLatestAssistantMessage(
        parts: AssistantResponseParts,
        reasoningDurationMillis: Long?
    ) {
        val currentMessages = _messages.value
        _messages.value = currentMessages.mapIndexed { index, message ->
            if (index == currentMessages.lastIndex) {
                message.copy(
                    content = parts.answer,
                    reasoningContent = parts.reasoning,
                    reasoningDurationMillis = reasoningDurationMillis ?: message.reasoningDurationMillis
                )
            } else {
                message
            }
        }
    }

    private fun appendLatestAssistantAgentTrace(text: String) {
        val currentMessages = _messages.value
        _messages.value = currentMessages.mapIndexed { index, message ->
            if (index == currentMessages.lastIndex) {
                message.copy(agentTrace = message.agentTrace + AgentTraceItem(text))
            } else {
                message
            }
        }
    }

    private fun finishLatestAssistantAgentTrace(startedAtNanos: Long) {
        val currentMessages = _messages.value
        _messages.value = currentMessages.mapIndexed { index, message ->
            if (index == currentMessages.lastIndex && message.agentTrace.isNotEmpty()) {
                message.copy(agentDurationMillis = (System.nanoTime() - startedAtNanos) / 1_000_000)
            } else {
                message
            }
        }
    }

    private fun appendLatestAssistantSources(sources: List<SourceLink>) {
        if (sources.isEmpty()) {
            return
        }
        val currentMessages = _messages.value
        _messages.value = currentMessages.mapIndexed { index, message ->
            if (index == currentMessages.lastIndex) {
                message.copy(sourceLinks = (message.sourceLinks + sources).distinctBy(SourceLink::url))
            } else {
                message
            }
        }
    }

    private fun buildSystemPrompt(userSystemPrompt: String?): String = listOfNotNull(
        userSystemPrompt,
        FunctionCalling.systemInstruction(toolExecutor.modelInstruction())
    ).joinToString("\n\n")

    /**
     * 从最近的完整 user/assistant 轮次开始向前保留。每次尝试加入一轮时，都让
     * Native 用真实 Chat Template 重新计数；超预算后不再加入更早的历史。
     */
    private fun buildHistoryWithinContextBudget(
        systemPrompt: String?,
        visibleHistory: List<ChatMessage>,
        currentUserPrompt: String,
        nCtx: Int,
        reservedOutputTokens: Int
    ): TrimmedHistory {
        // 生成区必须提前留出空间；否则 Prompt 即使刚好填满 n_ctx，Decode 也无法继续。
        val promptBudget = nCtx - reservedOutputTokens
        check(promptBudget > 0) { "上下文窗口不足以预留回复空间" }

        val baseMessages = buildList {
            systemPrompt?.let { add(ChatMessage(MessageRole.System, it)) }
            add(ChatMessage(MessageRole.User, currentUserPrompt))
        }
        val baseTokenCount = runtime.countChatPromptTokens(baseMessages)
        check(baseTokenCount in 1..promptBudget) {
            "System Prompt 和本轮问题已占用 $baseTokenCount tokens，超过可用预算 $promptBudget"
        }

        val completedTurns = visibleHistory.toCompletedTurns()
        var keptTurns: List<List<ChatMessage>> = emptyList()
        for (turn in completedTurns.asReversed()) {
            val candidate = buildList {
                systemPrompt?.let { add(ChatMessage(MessageRole.System, it)) }
                addAll(turn)
                keptTurns.asReversed().forEach { keptTurn ->
                    addAll(keptTurn)
                }
                add(ChatMessage(MessageRole.User, currentUserPrompt))
            }

            if (runtime.countChatPromptTokens(candidate) > promptBudget) {
                break
            }
            keptTurns = keptTurns + listOf(turn)
        }

        val messagesForNative = buildList {
            systemPrompt?.let { add(ChatMessage(MessageRole.System, it)) }
            keptTurns.asReversed().forEach { keptTurn ->
                addAll(keptTurn)
            }
            add(ChatMessage(MessageRole.User, currentUserPrompt))
        }
        return TrimmedHistory(
            messages = messagesForNative,
            droppedTurnCount = completedTurns.size - keptTurns.size
        )
    }

    private fun List<ChatMessage>.toCompletedTurns(): List<List<ChatMessage>> {
        val completedTurns = mutableListOf<List<ChatMessage>>()
        var index = 0
        while (index + 1 < size) {
            val user = this[index]
            val assistant = this[index + 1]
            if (user.role == MessageRole.User && assistant.role == MessageRole.Assistant) {
                completedTurns += listOf(user, assistant)
                index += 2
            } else {
                // 被停止的空回答等异常记录不作为“完整轮次”传入下一轮。
                index += 1
            }
        }
        return completedTurns
    }

    private data class TrimmedHistory(
        val messages: List<ChatMessage>,
        val droppedTurnCount: Int
    )

    fun updateSystemPrompt(value: String) {
        if (!_isGenerating.value) {
            _systemPrompt.value = value
        }
    }

    fun toggleReasoning(messageIndex: Int) {
        if (messageIndex !in _messages.value.indices) {
            return
        }
        _messages.value = _messages.value.mapIndexed { index, message ->
            if (index == messageIndex && message.reasoningContent.isNotBlank()) {
                message.copy(isReasoningExpanded = !message.isReasoningExpanded)
            } else {
                message
            }
        }
    }

    fun toggleAgentTrace(messageIndex: Int) {
        if (messageIndex !in _messages.value.indices) {
            return
        }
        _messages.value = _messages.value.mapIndexed { index, message ->
            if (index == messageIndex && message.agentTrace.isNotEmpty()) {
                message.copy(isAgentTraceExpanded = !message.isAgentTraceExpanded)
            } else {
                message
            }
        }
    }

    fun updateContextWindow(requestedNctx: Int) {
        if (requestedNctx <= 0 || _isGenerating.value || _uiState.value is ChatUiState.Preparing) {
            return
        }

        val currentState = _uiState.value as? ChatUiState.Ready
        if (currentState == null) {
            // 还未加载模型时只记录选择；下次加载模型时会使用它创建 Context。
            _requestedNctx.value = requestedNctx
            return
        }
        if (currentState.nCtx == requestedNctx) {
            return
        }

        viewModelScope.launch {
            try {
                _uiState.value = ChatUiState.Preparing("正在重建 Context…")
                val actualNctx = withContext(Dispatchers.Default) {
                    runtime.recreateContext(requestedNctx)
                }
                check(actualNctx > 0) { "Context 创建失败" }

                _requestedNctx.value = requestedNctx
                // n_ctx 缩小时，输出预留也必须同步落到可用档位；否则直到下一次发送
                // 才会因为 maxTokens >= n_ctx 失败。
                val maxTokens = listOf(2048, 4096, 6000)
                    .last { it < actualNctx }
                _inferenceConfig.value = _inferenceConfig.value.copy(
                    maxTokens = _inferenceConfig.value.maxTokens.coerceAtMost(maxTokens)
                )
                _uiState.value = ChatUiState.Ready(
                    currentState.modelFile,
                    actualNctx,
                    currentState.modelDescription,
                    currentState.modelName
                )
            } catch (error: Throwable) {
                _uiState.value = ChatUiState.Error(
                    error.message ?: "Context 重建失败"
                )
            }
        }
    }

    /** 不重建 Context，直接更新 llama.cpp 的 Decode 与 Prefill 线程数。 */
    fun updateCpuThreads(threads: Int) {
        if (threads <= 0 || _isGenerating.value || _uiState.value is ChatUiState.Preparing) {
            return
        }
        _cpuThreads.value = threads
        if (_uiState.value is ChatUiState.Ready) {
            viewModelScope.launch {
                withContext(Dispatchers.Default) {
                    runtime.setCpuThreads(threads)
                }
            }
        }
    }

    fun updateInferenceConfig(config: InferenceConfig) {
        if (_isGenerating.value) {
            return
        }
        require(config.temperature >= 0f) { "temperature must not be negative" }
        require(config.topK >= 0) { "topK must not be negative" }
        require(config.topP in 0f..1f) { "topP must be between 0 and 1" }
        require(config.minP in 0f..1f) { "minP must be between 0 and 1" }
        require(config.repeatPenalty > 0f) { "repeatPenalty must be positive" }
        require(config.maxTokens > 0) { "maxTokens must be positive" }
        val actualNctx = (_uiState.value as? ChatUiState.Ready)?.nCtx
        require(actualNctx == null || config.maxTokens < actualNctx) {
            "maxTokens must be smaller than n_ctx"
        }
        _inferenceConfig.value = config
    }

    fun updateThinkingEnabled(enabled: Boolean) {
        if (!_isGenerating.value) {
            _thinkingEnabled.value = enabled
        }
    }

    /** Qwen3 的软开关必须跟随最新 user 消息；UI 中仍只保留用户原始问题。 */
    private fun String.withQwen3ThinkingInstruction(): String {
        val readyState = _uiState.value as? ChatUiState.Ready
            ?: return this
        if (!readyState.supportsQwen3ThinkingSwitch()) {
            return this
        }
        return "$this\n${if (_thinkingEnabled.value) "/think" else "/no_think"}"
    }

    private fun ChatUiState.Ready.supportsQwen3ThinkingSwitch(): Boolean {
        val identity = "$modelName $modelDescription"
        return identity.contains("qwen3", ignoreCase = true) &&
                !identity.contains("instruct-2507", ignoreCase = true)
    }

    private fun splitAssistantResponse(
        rawContent: String,
        templateStartsThinking: Boolean
    ): AssistantResponseParts {
        val openingTag = "<think>"
        val closingTag = "</think>"
        val openingIndex = rawContent.indexOf(openingTag)
        if (openingIndex >= 0) {
            val reasoningStart = openingIndex + openingTag.length
            val closingIndex = rawContent.indexOf(closingTag, reasoningStart)
            return if (closingIndex >= 0) {
                AssistantResponseParts(
                    reasoning = rawContent.substring(reasoningStart, closingIndex).trim(),
                    answer = rawContent.substring(closingIndex + closingTag.length).trimStart()
                )
            } else {
                AssistantResponseParts(
                    reasoning = rawContent.substring(reasoningStart).trimStart(),
                    answer = rawContent.substring(0, openingIndex).trim()
                )
            }
        }

        // 部分 Qwen3 模板已在 generation prompt 中写入 <think>，模型只回传正文和 </think>。
        if (templateStartsThinking) {
            val closingIndex = rawContent.indexOf(closingTag)
            return if (closingIndex >= 0) {
                AssistantResponseParts(
                    reasoning = rawContent.substring(0, closingIndex).trim(),
                    answer = rawContent.substring(closingIndex + closingTag.length).trimStart()
                )
            } else {
                AssistantResponseParts(reasoning = rawContent.trimStart(), answer = "")
            }
        }
        return AssistantResponseParts(reasoning = "", answer = rawContent)
    }

    private data class AssistantResponseParts(
        val reasoning: String,
        val answer: String
    )

    private fun ToolCall.traceDescription(): String = when (name) {
        "get_weather" -> "调用天气工具：${arguments.optString("city").trim()}"
        "web_search" -> "调用网页搜索：${arguments.optString("query").trim()}"
        "open_settings" -> "调用系统设置：${arguments.optString("page").trim()}"
        else -> "调用工具：$name"
    }

    fun stopGenerating() {
        if (!_isGenerating.value || _isStopRequested.value) {
            return
        }

        _isStopRequested.value = true
        // 不能只取消协程：llama.cpp 此时仍在 C++ 循环中。这里通过 JNI 写入
        // Native atomic 标记，由 generate() 在下一个安全边界自行退出。
        runtime.requestStop()
    }

    /** 只清 UI 聊天记录；Native Context 与已加载的模型均保持不变。 */
    fun clearConversation() {
        if (_isGenerating.value || _uiState.value !is ChatUiState.Ready) {
            return
        }

        _messages.value = emptyList()
        clearLastPromptState()
        _generationError.value = null
    }

    /**
     * 清 Native 的 KV Cache，但保留 Kotlin 消息记录和模型权重。
     * 当前基础版本每次发送都会重新 Prefill，因此它主要用于观察资源边界；
     * 未来改成增量 KV Cache 后，这个操作会直接影响下一轮是否需要重新 Prefill。
     */
    fun resetContext() {
        if (_isGenerating.value || _uiState.value !is ChatUiState.Ready) {
            return
        }

        viewModelScope.launch {
            withContext(Dispatchers.Default) {
                runtime.resetContext()
            }
            clearLastPromptState()
            _generationError.value = null
        }
    }

    fun releaseModel() {
        if (_uiState.value is ChatUiState.Preparing || _isGenerating.value) {
            return
        }

        viewModelScope.launch {
            _uiState.value = ChatUiState.Preparing("正在释放模型…")
            withContext(Dispatchers.Default) {
                // 释放顺序由 NativeRuntime 保证：Context -> Model -> llama backend。
                runtime.close()
            }
            _memorySnapshots.value = _memorySnapshots.value.copy(
                afterModelReleased = takeMemorySnapshot()
            )
            selectedModelStore.clear()
            _messages.value = emptyList()
            clearLastPromptState()
            _generationError.value = null
            _uiState.value = ChatUiState.Idle
        }
    }

    private fun clearLastPromptState() {
        _finalPrompt.value = ""
        _promptTokenCount.value = 0
        _trimmedHistoryTurnCount.value = 0
        _generationProgress.value = null
        _generationMetrics.value = null
    }

    private fun takeMemorySnapshot(): MemorySnapshot {
        val runtime = Runtime.getRuntime()
        val memoryInfo = Debug.MemoryInfo()
        Debug.getMemoryInfo(memoryInfo)
        return MemorySnapshot(
            javaUsedBytes = runtime.totalMemory() - runtime.freeMemory(),
            javaMaxBytes = runtime.maxMemory(),
            nativeAllocatedBytes = Debug.getNativeHeapAllocatedSize(),
            totalPssKb = memoryInfo.totalPss,
            nativePssKb = memoryInfo.nativePss
        )
    }

    override fun onCleared() {
        // ViewModel 真正销毁时释放 Native 堆内存；不能依赖 Kotlin GC 自动处理。
        runtime.close()
    }
}
