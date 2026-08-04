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

    private val _isGenerating = MutableStateFlow(false)
    val isGenerating: StateFlow<Boolean> = _isGenerating.asStateFlow()

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

        // 当前先完成单轮问答：每次发送都用这一句作为全新 prompt，不携带历史消息。
        _messages.value = listOf(ChatMessage(MessageRole.User, prompt))
        _generationError.value = null
        _isGenerating.value = true

        viewModelScope.launch {
            try {
                val answer = withContext(Dispatchers.Default) {
                    runtime.generate(prompt, maxTokens = 128)
                }
                check(answer.isNotBlank()) { "模型没有返回内容" }
                _messages.value = _messages.value + ChatMessage(MessageRole.Assistant, answer)
            } catch (error: Throwable) {
                _generationError.value = error.message ?: "生成失败"
            } finally {
                _isGenerating.value = false
            }
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
            _uiState.value = ChatUiState.Idle
        }
    }

    override fun onCleared() {
        // ViewModel 真正销毁时释放 Native 堆内存；不能依赖 Kotlin GC 自动处理。
        runtime.close()
    }
}
