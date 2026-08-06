package com.arick.androidlocalllmlab

interface NativeGenerationCallback {
    fun onPrompt(prompt: String, tokenCount: Int)

    fun onToken(piece: String)

    fun onMetrics(
        generatedTokenCount: Int,
        reusedPromptTokenCount: Int,
        prefillMillis: Long,
        firstTokenMillis: Long,
        decodeMillis: Long,
        totalMillis: Long
    )
}

object NativeLlmBridge {
    init {
        System.loadLibrary("native_llm")
    }

    external fun nativeCreateRuntime(): Long

    external fun nativeReleaseRuntime(handle: Long)

    external fun nativeLoadModel(
        handle: Long,
        modelPath: String
    ): Boolean

    external fun nativeUnloadModel(handle: Long)

    external fun nativeModelDescription(handle: Long): String

    external fun nativeModelMetadata(handle: Long, key: String): String

    external fun nativeCreateContext(
        handle: Long,
        requestedNctx: Int
    ): Int

    external fun nativeReleaseContext(handle: Long)

    external fun nativeResetContext(handle: Long)

    external fun nativeSetThreads(
        handle: Long,
        generationThreads: Int,
        batchThreads: Int
    ): Int

    external fun nativeRequestStop(handle: Long)

    external fun nativeResetStopRequest(handle: Long)

    external fun nativeTokenize(
        handle: Long,
        text: String
    ): IntArray

    external fun nativeCountChatPromptTokens(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>
    ): Int

    external fun nativePrefill(
        handle: Long,
        tokenIds: IntArray
    ): Int

    external fun nativeSampleNextToken(handle: Long): Int

    external fun nativeTokenToPiece(
        handle: Long,
        tokenId: Int
    ): String

    external fun nativeGenerate(
        handle: Long,
        roles: Array<String>,
        contents: Array<String>,
        maxTokens: Int,
        temperature: Float,
        topK: Int,
        topP: Float,
        minP: Float,
        repeatPenalty: Float,
        seed: Int,
        generationCallback: NativeGenerationCallback
    ): String
}
