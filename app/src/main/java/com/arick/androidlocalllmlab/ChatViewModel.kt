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
                requestedNctx = 2048
            )
        }

        check(actualNctx > 0) { "模型或 Context 创建失败" }
        _messages.value = emptyList()
        _finalPrompt.value = ""
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
        val historyForNative = buildList {
            _systemPrompt.value.trim().takeIf { it.isNotEmpty() }?.let { systemPrompt ->
                add(ChatMessage(MessageRole.System, systemPrompt))
            }
            addAll(visibleHistory)
            add(ChatMessage(MessageRole.User, prompt))
        }
        _messages.value = visibleHistory + ChatMessage(
            MessageRole.User,
            prompt
        ) + listOf(
            ChatMessage(MessageRole.Assistant, "")
        )
        _generationError.value = null
        _finalPrompt.value = ""
        // 必须在启动协程前清除上次的停止标记，否则下一轮会立即被旧请求取消。
        runtime.resetStopRequest()
        _isStopRequested.value = false
        _isGenerating.value = true

        viewModelScope.launch {
            try {
                val answer = withContext(Dispatchers.Default) {
                    runtime.generate(
                        messages = historyForNative,
                        maxTokens = 128,
                        onPrompt = { finalPrompt ->
                            _finalPrompt.value = finalPrompt
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

    fun updateSystemPrompt(value: String) {
        if (!_isGenerating.value) {
            _systemPrompt.value = value
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
            _uiState.value = ChatUiState.Idle
        }
    }

    override fun onCleared() {
        // ViewModel 真正销毁时释放 Native 堆内存；不能依赖 Kotlin GC 自动处理。
        runtime.close()
    }
}
