package com.arick.androidlocalllmlab

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

    external fun nativeCreateContext(
        handle: Long,
        requestedNctx: Int
    ): Int

    external fun nativeReleaseContext(handle: Long)
}