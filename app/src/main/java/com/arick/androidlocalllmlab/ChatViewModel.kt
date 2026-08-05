package com.arick.androidlocalllmlab

import android.app.Application
import android.net.Uri
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
        val nCtx: Int
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
    val content: String
)

class ChatViewModel(application: Application) : AndroidViewModel(application) {
    private companion object {
        // 生成区必须提前留出空间；否则 Prompt 即使刚好填满 n_ctx，Decode 也无法继续。
        const val MAX_GENERATION_TOKENS = 128
    }

    // Runtime 持有 JNI 返回的 nativeHandle，生命周期与 ViewModel 一致。
    private val runtime = LocalLlmRuntime()
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

    private val _requestedNctx = MutableStateFlow(2048)
    val requestedNctx: StateFlow<Int> = _requestedNctx.asStateFlow()

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
        val actualNctx = withContext(Dispatchers.Default) {
            // Native 模型加载与 Context 分配可能耗时，不能阻塞 Compose 主线程。
            runtime.prepareModel(
                modelPath = modelFile.absolutePath,
                requestedNctx = _requestedNctx.value
            )
        }

        check(actualNctx > 0) { "模型或 Context 创建失败" }
        _messages.value = emptyList()
        _finalPrompt.value = ""
        _promptTokenCount.value = 0
        _trimmedHistoryTurnCount.value = 0
        _generationError.value = null
        _uiState.value = ChatUiState.Ready(modelFile, actualNctx)
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
        val systemPrompt = _systemPrompt.value.trim().takeIf { it.isNotEmpty() }
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
        // 必须在启动协程前清除上次的停止标记，否则下一轮会立即被旧请求取消。
        runtime.resetStopRequest()
        _isStopRequested.value = false
        _isGenerating.value = true

        viewModelScope.launch {
            try {
                val answer = withContext(Dispatchers.Default) {
                    val trimmedHistory = buildHistoryWithinContextBudget(
                        systemPrompt = systemPrompt,
                        visibleHistory = visibleHistory,
                        currentUserPrompt = prompt,
                        nCtx = actualNctx
                    )
                    _trimmedHistoryTurnCount.value = trimmedHistory.droppedTurnCount

                    runtime.generate(
                        messages = trimmedHistory.messages,
                        maxTokens = MAX_GENERATION_TOKENS,
                        onPrompt = { finalPrompt, tokenCount ->
                            _finalPrompt.value = finalPrompt
                            _promptTokenCount.value = tokenCount
                        }
                    ) { piece ->
                        val currentMessages = _messages.value
                        _messages.value = currentMessages.mapIndexed { index, message ->
                            if (index == currentMessages.lastIndex) {
                                message.copy(content = message.content + piece)
                            } else {
                                message
                            }
                        }
                    }
                }
                if (answer.isBlank() && !_isStopRequested.value) {
                    error("模型没有返回内容")
                }
            } catch (error: Throwable) {
                _generationError.value = error.message ?: "生成失败"
            } finally {
                _isGenerating.value = false
                _isStopRequested.value = false
            }
        }
    }

    /**
     * 从最近的完整 user/assistant 轮次开始向前保留。每次尝试加入一轮时，都让
     * Native 用真实 Chat Template 重新计数；超预算后不再加入更早的历史。
     */
    private fun buildHistoryWithinContextBudget(
        systemPrompt: String?,
        visibleHistory: List<ChatMessage>,
        currentUserPrompt: String,
        nCtx: Int
    ): TrimmedHistory {
        val promptBudget = nCtx - MAX_GENERATION_TOKENS
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
                _uiState.value = ChatUiState.Ready(currentState.modelFile, actualNctx)
            } catch (error: Throwable) {
                _uiState.value = ChatUiState.Error(
                    error.message ?: "Context 重建失败"
                )
            }
        }
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
    }

    override fun onCleared() {
        // ViewModel 真正销毁时释放 Native 堆内存；不能依赖 Kotlin GC 自动处理。
        runtime.close()
    }
}
