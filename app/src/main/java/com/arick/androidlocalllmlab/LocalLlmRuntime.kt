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
}
