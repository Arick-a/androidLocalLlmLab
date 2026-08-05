package com.arick.androidlocalllmlab

class LocalLlmRuntime : AutoCloseable {
    private var nativeHandle: Long = 0L
    private var cpuThreads: Int = 1
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

    fun modelDescription(): String {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        return NativeLlmBridge.nativeModelDescription(nativeHandle)
    }

    fun modelMetadata(key: String): String {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        return NativeLlmBridge.nativeModelMetadata(nativeHandle, key)
    }

    fun createContext(requestedNctx: Int): Int {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        check(requestedNctx > 0) { "n_ctx must be positive" }

        val actualNctx = NativeLlmBridge.nativeCreateContext(
            nativeHandle,
            requestedNctx
        )
        if (actualNctx > 0) {
            NativeLlmBridge.nativeSetThreads(nativeHandle, cpuThreads, cpuThreads)
        }
        return actualNctx
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
        requestedNctx: Int,
        cpuThreads: Int
    ): Int {
        this.cpuThreads = cpuThreads
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

    /**
     * llama.cpp 区分单 Token Decode 和批量 Prefill 的线程数。
     * 当前实验先让两者使用同一个值，避免一次引入两个变量。
     */
    fun setCpuThreads(threads: Int): Int {
        check(threads > 0) { "cpu threads must be positive" }
        cpuThreads = threads
        if (nativeHandle == 0L) {
            return cpuThreads
        }
        return NativeLlmBridge.nativeSetThreads(nativeHandle, threads, threads)
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
            contents = messages.map(ChatMessage::toNativeContent).toTypedArray()
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
        inferenceConfig: InferenceConfig,
        onPrompt: (String, Int) -> Unit,
        onToken: (String) -> Unit,
        onMetrics: (GenerationMetrics) -> Unit
    ): String {
        check(nativeHandle != 0L) { "Runtime has not been created" }
        check(messages.isNotEmpty()) { "messages must not be empty" }
        check(inferenceConfig.maxTokens > 0) { "maxTokens must be positive" }

        return NativeLlmBridge.nativeGenerate(
            handle = nativeHandle,
            roles = messages.map { message ->
                when (message.role) {
                    MessageRole.System -> "system"
                    MessageRole.User -> "user"
                    MessageRole.Assistant -> "assistant"
                }
            }.toTypedArray(),
            contents = messages.map(ChatMessage::toNativeContent).toTypedArray(),
            maxTokens = inferenceConfig.maxTokens,
            temperature = inferenceConfig.temperature,
            topK = inferenceConfig.topK,
            topP = inferenceConfig.topP,
            minP = inferenceConfig.minP,
            repeatPenalty = inferenceConfig.repeatPenalty,
            seed = inferenceConfig.seed,
            generationCallback = object : NativeGenerationCallback {
                override fun onPrompt(prompt: String, tokenCount: Int) =
                    onPrompt(prompt, tokenCount)

                override fun onToken(piece: String) = onToken(piece)

                override fun onMetrics(
                    generatedTokenCount: Int,
                    prefillMillis: Long,
                    firstTokenMillis: Long,
                    decodeMillis: Long,
                    totalMillis: Long
                ) = onMetrics(
                    GenerationMetrics(
                        generatedTokenCount = generatedTokenCount,
                        prefillMillis = prefillMillis,
                        firstTokenMillis = firstTokenMillis,
                        decodeMillis = decodeMillis,
                        totalMillis = totalMillis
                    )
                )
            }
        )
    }
}

/** Native 仍需看到完整 assistant 轨迹；UI 只是把 reasoning 从普通回答中分栏展示。 */
private fun ChatMessage.toNativeContent(): String {
    if (role != MessageRole.Assistant || reasoningContent.isBlank()) {
        return content
    }
    return "<think>\n$reasoningContent\n</think>\n\n$content"
}

/**
 * 采样器配置。temperature 为 0 时 Native 使用 greedy，其他随机采样参数不会参与选词；
 * 这样可以保留当前稳定、可复现的默认行为。
 */
data class InferenceConfig(
    val temperature: Float = 0f,
    val topK: Int = 40,
    val topP: Float = 0.95f,
    val minP: Float = 0.05f,
    val repeatPenalty: Float = 1.10f,
    val seed: Int = 42,
    val maxTokens: Int = 256
) {
    val isGreedy: Boolean
        get() = temperature <= 0f
}

/** 一轮 Native 推理的真实耗时；单位统一为毫秒，避免 Kotlin 侧猜测。 */
data class GenerationMetrics(
    val generatedTokenCount: Int,
    val prefillMillis: Long,
    val firstTokenMillis: Long,
    val decodeMillis: Long,
    val totalMillis: Long
) {
    val decodeTokensPerSecond: Double
        get() = if (decodeMillis <= 0L) 0.0 else generatedTokenCount * 1_000.0 / decodeMillis
}
