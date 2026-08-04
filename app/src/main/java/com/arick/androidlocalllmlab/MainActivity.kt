package com.arick.androidlocalllmlab

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import com.arick.androidlocalllmlab.ui.theme.AndroidLocalLlmLabTheme

class MainActivity : ComponentActivity() {
    private val chatViewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroidLocalLlmLabTheme {
                Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                    ChatScreen(
                        modifier = Modifier.padding(innerPadding),
                        viewModel = chatViewModel
                    )
                }
            }
        }
    }
}

@Composable
private fun ChatScreen(
    modifier: Modifier = Modifier,
    viewModel: ChatViewModel
) {
    val uiState by viewModel.uiState.collectAsState()
    val selectModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.loadModel(uri)
        }
    }

    val isPreparing = uiState is ChatUiState.Preparing

    Column(modifier = modifier) {
        Button(
            enabled = !isPreparing,
            onClick = {
                selectModelLauncher.launch(arrayOf("*/*"))
            }
        ) {
            Text("选择并加载 GGUF")
        }

        Button(
            enabled = uiState is ChatUiState.Ready,
            onClick = viewModel::releaseModel
        ) {
            Text("释放模型")
        }

        Text(
            text = when (val state = uiState) {
                ChatUiState.Idle -> "尚未加载模型"
                is ChatUiState.Preparing -> state.message
                is ChatUiState.Ready -> "模型已就绪：${state.modelFile.name}，n_ctx = ${state.nCtx}"
                is ChatUiState.Error -> "加载失败：${state.message}"
            }
        )
    }
}
