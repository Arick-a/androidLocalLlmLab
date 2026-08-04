package com.arick.androidlocalllmlab

import android.content.Context
import java.io.File

class SelectedModelStore(context: Context) {
    private val preferences = context.getSharedPreferences(
        "local_llm_preferences",
        Context.MODE_PRIVATE
    )

    fun save(modelFile: File) {
        // 保存的是导入到 filesDir 的真实路径，不是临时的 content:// URI。
        preferences.edit()
            .putString(KEY_SELECTED_MODEL_PATH, modelFile.absolutePath)
            .apply()
    }

    fun selectedModelFile(): File? {
        val path = preferences.getString(KEY_SELECTED_MODEL_PATH, null)
            ?: return null
        return File(path)
    }

    fun clear() {
        preferences.edit()
            .remove(KEY_SELECTED_MODEL_PATH)
            .apply()
    }

    private companion object {
        const val KEY_SELECTED_MODEL_PATH = "selected_model_path"
    }
}
