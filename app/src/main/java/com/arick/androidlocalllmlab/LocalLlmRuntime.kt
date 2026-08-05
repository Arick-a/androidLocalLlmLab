package com.arick.androidlocalllmlab

class LocalLlmRuntime : AutoCloseable {
    private var nativeHandle: Long = 0L
    val isCreated: Boolean
        get() = nativeHandle != 0L

    fun create() {
        check(nativeHandle == 0L) { "Runtime already exists" }
        nativeHandle = NativeLlmBridge.nativeCreateRuntime()
    }

    override fun close() {
        if (nativeHandle != 0L) {
            NativeLlmBridge.nativeReleaseRuntime(nativeHandle)
            nativeHandle = 0L
        }
    }

    fun loadModel(modelPath: String): Boolean {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        return NativeLlmBridge.nativeLoadModel(nativeHandle, modelPath)
    }

    fun unloadModel() {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        NativeLlmBridge.nativeUnloadModel(nativeHandle)
    }

    fun createContext(requestedNctx: Int): Int {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        check(requestedNctx > 0) { "n_ctx must be positive" }

        return NativeLlmBridge.nativeCreateContext(
            nativeHandle,
            requestedNctx
        )
    }

    fun releaseContext() {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        NativeLlmBridge.nativeReleaseContext(nativeHandle)
    }

    fun resetContext() {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        NativeLlmBridge.nativeResetContext(nativeHandle)
    }

    fun requestStop() {
        if (nativeHandle != 0L) {
            NativeLlmBridge.nativeRequestStop(nativeHandle)
        }
    }

    fun resetStopRequest() {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        NativeLlmBridge.nativeResetStopRequest(nativeHandle)
    }

    fun createIfNeeded() {
        if (nativeHandle == 0L) {
            nativeHandle = NativeLlmBridge.nativeCreateRuntime()
        }
    }

    fun prepareModel(
        modelPath: String,
        requestedNctx: Int
    ): Int {
        createIfNeeded()

        // 切换模型前先释放旧 Context 和旧模型，避免旧 KV Cache 和权重继续占用 Native 内存。
        unloadModel()

        if (!loadModel(modelPath)) {
            return 0
        }

        val actualNctx = createContext(requestedNctx)
        if (actualNctx == 0) {
            // Context 创建失败时同步卸载模型，防止 UI 误以为模型已经可推理。
            unloadModel()
        }

        return actualNctx
    }

    fun recreateContext(requestedNctx: Int): Int {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        check(requestedNctx > 0) { "n_ctx must be positive" }

        // 只重建推理现场：模型权重仍留在 Native 内存中，旧 KV Cache 会随 Context 释放。
        releaseContext()
        return createContext(requestedNctx)
    }

    fun tokenize(text: String): IntArray {
        check(nativeHandle != 0L) { "Runtime has not been created" }

        return NativeLlmBridge.nativeTokenize(
            handle = nativeHandle,
            text = text
        )
    }

    /**
     * 先应用当前 GGUF 的 Chat Template，再计算真实 Prompt Token 数。
     * 这个计数用于 Kotlin 的历史裁剪，不能用字符串长度或普通 tokenize() 替代。
     */
    fun countChatPromptTokens(messages: List<ChatMessage>): Int {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        check(messages.isNotEmpty()) { "messages must not be empty" }

        return NativeLlmBridge.nativeCountChatPromptTokens(
            handle = nativeHandle,
            roles = messages.map { message ->
                when (message.role) {
                    MessageRole.System -> "system"
                    MessageRole.User -> "user"
                    MessageRole.Assistant -> "assistant"
                }
            }.toTypedArray(),
            contents = messages.map(ChatMessage::content).toTypedArray()
        )
    }

    fun prefill(tokenIds: IntArray): Int {
        check(nativeHandle != 0L) { "Runtime has not been created" }

        return NativeLlmBridge.nativePrefill(
            handle = nativeHandle,
            tokenIds = tokenIds
        )
    }

    fun sampleNextToken(): Int {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        return NativeLlmBridge.nativeSampleNextToken(nativeHandle)
    }

    fun tokenToPiece(tokenId: Int): String {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        return NativeLlmBridge.nativeTokenToPiece(nativeHandle, tokenId)
    }

    fun generate(
        messages: List<ChatMessage>,
        maxTokens: Int,
        onPrompt: (String, Int) -> Unit,
        onToken: (String) -> Unit
    ): String {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        check(messages.isNotEmpty()) { "messages must not be empty" }
        check(maxTokens > 0) { "maxTokens must be positive" }

        return NativeLlmBridge.nativeGenerate(
            handle = nativeHandle,
            roles = messages.map { message ->
                when (message.role) {
                    MessageRole.System -> "system"
                    MessageRole.User -> "user"
                    MessageRole.Assistant -> "assistant"
                }
            }.toTypedArray(),
            contents = messages.map(ChatMessage::content).toTypedArray(),
            maxTokens = maxTokens,
            generationCallback = object : NativeGenerationCallback {
                override fun onPrompt(prompt: String, tokenCount: Int) =
                    onPrompt(prompt, tokenCount)

                override fun onToken(piece: String) = onToken(piece)
            }
        )
    }
}
