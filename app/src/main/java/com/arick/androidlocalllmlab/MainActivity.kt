package com.arick.androidlocalllmlab

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.text.BasicTextField
import com.arick.androidlocalllmlab.ui.theme.AndroidLocalLlmLabTheme

class MainActivity : ComponentActivity() {
    private val chatViewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroidLocalLlmLabTheme(dynamicColor = false) {
                ChatScreen(viewModel = chatViewModel)
            }
        }
    }
}

@Composable
private fun ChatScreen(viewModel: ChatViewModel) {
    val uiState by viewModel.uiState.collectAsState()
    val messages by viewModel.messages.collectAsState()
    val systemPrompt by viewModel.systemPrompt.collectAsState()
    val finalPrompt by viewModel.finalPrompt.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val isStopRequested by viewModel.isStopRequested.collectAsState()
    val generationError by viewModel.generationError.collectAsState()
    var prompt by remember { mutableStateOf("") }
    val selectModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.loadModel(uri)
        }
    }

    val isPreparing = uiState is ChatUiState.Preparing
    val isModelReady = uiState is ChatUiState.Ready

    Scaffold(
        containerColor = Color.White,
        topBar = {
            ChatHeader(
                onSelectModel = {
                    if (!isPreparing && !isGenerating) {
                        selectModelLauncher.launch(arrayOf("*/*"))
                    }
                }
            )
        },
        bottomBar = {
            ChatComposer(
                prompt = prompt,
                enabled = isModelReady,
                isGenerating = isGenerating,
                isStopRequested = isStopRequested,
                onPromptChange = { prompt = it },
                onSend = {
                    viewModel.send(prompt)
                    prompt = ""
                },
                onStop = viewModel::stopGenerating
            )
        }
    ) { innerPadding ->
        Box(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .background(Color.White)
        ) {
            when (val state = uiState) {
                ChatUiState.Idle -> {
                    EmptyState(
                        title = "选择一个 GGUF 模型开始对话",
                        actionText = "选择模型",
                        onAction = {
                            selectModelLauncher.launch(arrayOf("*/*"))
                        }
                    )
                }

                is ChatUiState.Preparing -> {
                    CenterMessage(state.message)
                }

                is ChatUiState.Ready -> {
                    ChatContent(
                        nCtx = state.nCtx,
                        messages = messages,
                        systemPrompt = systemPrompt,
                        finalPrompt = finalPrompt,
                        isGenerating = isGenerating,
                        isStopRequested = isStopRequested,
                        errorMessage = generationError,
                        onSystemPromptChange = viewModel::updateSystemPrompt
                    )
                }

                is ChatUiState.Error -> {
                    EmptyState(
                        title = "模型加载失败：${state.message}",
                        actionText = "重新选择模型",
                        onAction = {
                            selectModelLauncher.launch(arrayOf("*/*"))
                        }
                    )
                }
            }
        }
    }
}

@Composable
private fun ChatHeader(onSelectModel: () -> Unit) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .height(76.dp)
            .padding(horizontal = 12.dp)
    ) {
        IconButton(
            modifier = Modifier.align(Alignment.CenterStart),
            onClick = onSelectModel
        ) {
            Icon(
                imageVector = Icons.Default.Menu,
                contentDescription = "选择模型",
                tint = Color.Black
            )
        }

        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "新对话",
                color = Color.Black,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = "本地生成，请注意核实",
                color = Color(0xFF999999),
                fontSize = 12.sp
            )
        }
    }
}

@Composable
private fun ChatComposer(
    prompt: String,
    enabled: Boolean,
    isGenerating: Boolean,
    isStopRequested: Boolean,
    onPromptChange: (String) -> Unit,
    onSend: () -> Unit,
    onStop: () -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxWidth()
            .imePadding()
            .navigationBarsPadding()
            .padding(bottom = 12.dp)
    ) {
        Surface(
            modifier = Modifier
                .fillMaxWidth()
                .padding(horizontal = 18.dp),
            color = Color.White,
            shape = RoundedCornerShape(14.dp),
            shadowElevation = 2.dp
        ) {
            Row(
                modifier = Modifier
                    .fillMaxWidth()
                    .height(50.dp)
                    .padding(horizontal = 10.dp),
                verticalAlignment = Alignment.CenterVertically
            ) {
                Box(
                    modifier = Modifier
                        .weight(1f)
                        .padding(start = 14.dp, end = 8.dp),
                    contentAlignment = Alignment.CenterStart
                ) {
                    if (prompt.isBlank()) {
                        Text(
                            text = when {
                                isStopRequested -> "正在停止…"
                                isGenerating -> "正在生成…"
                                enabled -> "发消息"
                                else -> "正在准备本地模型…"
                            },
                            color = Color(0xFFAAAAAA),
                            fontSize = 16.sp
                        )
                    }

                    BasicTextField(
                        value = prompt,
                        enabled = enabled && !isGenerating,
                        onValueChange = onPromptChange,
                        singleLine = true,
                        textStyle = TextStyle(
                            color = Color.Black,
                            fontSize = 16.sp
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                Surface(
                    color = when {
                        isStopRequested -> Color(0xFF999999)
                        isGenerating -> Color(0xFFE85D5D)
                        enabled && prompt.isNotBlank() -> Color(0xFF1677FF)
                        else -> Color(0xFFE5E5E5)
                    },
                    shape = CircleShape
                ) {
                    IconButton(
                        modifier = Modifier.size(28.dp),
                        enabled = (isGenerating && !isStopRequested) ||
                                (!isGenerating && enabled && prompt.isNotBlank()),
                        onClick = if (isGenerating) onStop else onSend
                    ) {
                        if (isGenerating) {
                            Text("■", color = Color.White, fontSize = 12.sp)
                        } else {
                            Icon(
                                imageVector = Icons.Default.ArrowUpward,
                                contentDescription = "发送",
                                tint = Color.White
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatContent(
    nCtx: Int,
    messages: List<ChatMessage>,
    systemPrompt: String,
    finalPrompt: String,
    isGenerating: Boolean,
    isStopRequested: Boolean,
    errorMessage: String?,
    onSystemPromptChange: (String) -> Unit
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 18.dp, vertical = 12.dp)
    ) {
        Text(
            text = "本地模型已就绪 · n_ctx $nCtx",
            modifier = Modifier.align(Alignment.CenterHorizontally),
            color = Color(0xFF999999),
            fontSize = 12.sp
        )

        OutlinedTextField(
            value = systemPrompt,
            onValueChange = onSystemPromptChange,
            modifier = Modifier
                .fillMaxWidth()
                .padding(top = 12.dp),
            enabled = !isGenerating,
            label = { Text("System Prompt") },
            placeholder = { Text("例如：你是简洁的 Android 助手") },
            minLines = 2,
            maxLines = 3,
            textStyle = TextStyle(fontSize = 14.sp)
        )

        if (finalPrompt.isNotBlank()) {
            Text(
                text = "本轮最终 Prompt（Native 模板化后）",
                modifier = Modifier.padding(top = 16.dp),
                color = Color(0xFF777777),
                fontSize = 12.sp
            )
            Text(
                text = finalPrompt,
                modifier = Modifier.padding(top = 4.dp),
                color = Color(0xFF555555),
                fontSize = 12.sp
            )
        }

        messages.filter { it.content.isNotBlank() }.forEach { message ->
            Text(
                text = when (message.role) {
                    MessageRole.System -> "System"
                    MessageRole.User -> "我"
                    MessageRole.Assistant -> "AI"
                },
                modifier = Modifier.padding(top = 20.dp),
                color = Color(0xFF777777),
                fontSize = 12.sp
            )
            Text(
                text = message.content,
                modifier = Modifier.padding(top = 4.dp),
                color = Color.Black,
                fontSize = 16.sp
            )
        }

        if (isGenerating) {
            Text(
                text = if (isStopRequested) "正在停止…" else "正在生成…",
                modifier = Modifier.padding(top = 20.dp),
                color = Color(0xFF777777),
                fontSize = 12.sp
            )
        }

        if (errorMessage != null) {
            Text(
                text = "生成失败：$errorMessage",
                modifier = Modifier.padding(top = 20.dp),
                color = Color(0xFFD32F2F),
                fontSize = 12.sp
            )
        }
    }
}

@Composable
private fun EmptyState(
    title: String,
    actionText: String,
    onAction: () -> Unit
) {
    Column(
        modifier = Modifier.fillMaxSize(),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.Center
    ) {
        Text(title, color = Color(0xFF777777), fontSize = 14.sp)
        OutlinedButton(
            modifier = Modifier.padding(top = 12.dp),
            onClick = onAction
        ) {
            Text(actionText)
        }
    }
}

@Composable
private fun CenterMessage(message: String) {
    Text(
        text = message,
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        color = Color(0xFF777777),
        fontSize = 12.sp
    )
}
