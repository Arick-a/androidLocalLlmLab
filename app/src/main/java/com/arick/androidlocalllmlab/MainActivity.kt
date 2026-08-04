package com.arick.androidlocalllmlab

import android.os.Bundle
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import com.arick.androidlocalllmlab.ui.theme.AndroidLocalLlmLabTheme
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.platform.LocalContext
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

class MainActivity : ComponentActivity() {
    val localLlmRuntime: LocalLlmRuntime by lazy { LocalLlmRuntime() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroidLocalLlmLabTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    Greeting(
                        name = "Android",
                        modifier = Modifier.padding(innerPadding),
                        localLlmRuntime = localLlmRuntime
                    )
                }
            }
        }
    }
}

@Composable
fun Greeting(name: String, modifier: Modifier = Modifier, localLlmRuntime: LocalLlmRuntime) {
    val context = LocalContext.current
    val coroutineScope = rememberCoroutineScope()
    val importer = remember(context) {
        ModelFileImporter(context.applicationContext)
    }

    var modelStatus by remember {
        mutableStateOf("尚未选择模型")
    }

    val selectModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri == null) {
            modelStatus = "已取消选择"
            return@rememberLauncherForActivityResult
        }

        if (!localLlmRuntime.isCreated) {
            modelStatus = "请先创建 Runtime"
            return@rememberLauncherForActivityResult
        }

        coroutineScope.launch {
            modelStatus = "正在导入模型文件…"

            runCatching {
                val localFile = importer.importModel(uri)

                modelStatus = "正在加载模型…"
                val loaded = withContext(Dispatchers.Default) {
                    localLlmRuntime.loadModel(localFile.absolutePath)
                }

                check(loaded) { "llama.cpp 无法加载该 GGUF 文件" }
                localFile
            }.onSuccess { file ->
                modelStatus = "加载成功：${file.name}"
            }.onFailure { error ->
                modelStatus = "加载失败：${error.message}"
            }
        }
    }

    Column(modifier = modifier) {
        Button(onClick = {
            localLlmRuntime.create()
        }, content = { Text(text = "create") })
        Button(onClick = {
            localLlmRuntime.close()
        }, content = { Text(text = "release") })
        Button(onClick = {
            localLlmRuntime.loadModel("/not-found.gguf")
        }, content = { Text(text = "loadModel") })
        Button(
            onClick = {
                selectModelLauncher.launch(arrayOf("*/*"))
            }
        ) {
            Text("选择并加载 GGUF")
        }

        Text(modelStatus)
    }

}
