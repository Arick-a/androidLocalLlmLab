package com.arick.androidlocalllmlab

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.background
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.layout.offset
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.layout.widthIn
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.text.BasicTextField
import com.arick.androidlocalllmlab.ui.theme.AppColors
import com.arick.androidlocalllmlab.ui.theme.AndroidLocalLlmLabTheme
import kotlinx.coroutines.launch

class MainActivity : ComponentActivity() {
    private val chatViewModel: ChatViewModel by viewModels()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        setContent {
            AndroidLocalLlmLabTheme {
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
    val promptTokenCount by viewModel.promptTokenCount.collectAsState()
    val trimmedHistoryTurnCount by viewModel.trimmedHistoryTurnCount.collectAsState()
    val requestedNctx by viewModel.requestedNctx.collectAsState()
    val isGenerating by viewModel.isGenerating.collectAsState()
    val isStopRequested by viewModel.isStopRequested.collectAsState()
    val generationError by viewModel.generationError.collectAsState()
    var prompt by remember { mutableStateOf("") }
    var isSettingsVisible by remember { mutableStateOf(false) }
    val drawerState = rememberDrawerState(initialValue = DrawerValue.Closed)
    val coroutineScope = rememberCoroutineScope()
    val selectModelLauncher = rememberLauncherForActivityResult(
        contract = ActivityResultContracts.OpenDocument()
    ) { uri ->
        if (uri != null) {
            viewModel.loadModel(uri)
        }
    }

    val isPreparing = uiState is ChatUiState.Preparing
    val isModelReady = uiState is ChatUiState.Ready

    // 设置页是当前 Composable 的内部页面状态，不在 Android Navigation 返回栈中。
    // 因此需要显式消费系统返回，优先回到聊天页而不是退出整个 App。
    BackHandler(enabled = isSettingsVisible) {
        isSettingsVisible = false
    }

    if (isSettingsVisible) {
        ConversationSettingsScreen(
            systemPrompt = systemPrompt,
            finalPrompt = finalPrompt,
            promptTokenCount = promptTokenCount,
            trimmedHistoryTurnCount = trimmedHistoryTurnCount,
            nCtx = (uiState as? ChatUiState.Ready)?.nCtx,
            requestedNctx = requestedNctx,
            isGenerating = isGenerating,
            onSystemPromptChange = viewModel::updateSystemPrompt,
            onContextWindowChange = viewModel::updateContextWindow,
            onClearConversation = viewModel::clearConversation,
            onResetContext = viewModel::resetContext,
            onSelectModel = {
                isSettingsVisible = false
                selectModelLauncher.launch(arrayOf("*/*"))
            },
            onReleaseModel = {
                viewModel.releaseModel()
                isSettingsVisible = false
            },
            onBack = { isSettingsVisible = false }
        )
    } else {
        // 模型状态放在标题下方：无论正在加载还是已经就绪，用户都能第一眼看到。
        val modelStatus = when (val state = uiState) {
            ChatUiState.Idle -> "请选择本地模型"
            is ChatUiState.Preparing -> state.message
            is ChatUiState.Ready -> "模型已就绪 · n_ctx ${state.nCtx}"
            is ChatUiState.Error -> "模型加载失败"
        }

        ModalNavigationDrawer(
            drawerState = drawerState,
            drawerContent = {
                HistoryDrawer(
                    onClose = { coroutineScope.launch { drawerState.close() } }
                )
            }
        ) {
            Scaffold(
                containerColor = AppColors.PageBackground,
                topBar = {
                    ChatHeader(
                        modelStatus = modelStatus,
                        settingsEnabled = !isGenerating,
                        onOpenHistory = { coroutineScope.launch { drawerState.open() } },
                        onOpenSettings = { isSettingsVisible = true }
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
                        .background(AppColors.PageBackground)
                ) {
                    when (val state = uiState) {
                        ChatUiState.Idle -> EmptyState(
                            title = "选择一个 GGUF 模型开始对话",
                            actionText = "选择模型",
                            onAction = { selectModelLauncher.launch(arrayOf("*/*")) }
                        )

                        // 加载过程的文字已放到标题下方，内容区保持干净。
                        is ChatUiState.Preparing -> Box(modifier = Modifier.fillMaxSize())

                        is ChatUiState.Ready -> ChatContent(
                            messages = messages,
                            isGenerating = isGenerating,
                            isStopRequested = isStopRequested,
                            errorMessage = generationError
                        )

                        is ChatUiState.Error -> EmptyState(
                            title = "模型加载失败：${state.message}",
                            actionText = "重新选择模型",
                            onAction = { selectModelLauncher.launch(arrayOf("*/*")) }
                        )
                    }
                }
            }
        }
    }
}

@Composable
private fun ChatHeader(
    modelStatus: String,
    settingsEnabled: Boolean,
    onOpenHistory: () -> Unit,
    onOpenSettings: () -> Unit
) {
    Box(
        modifier = Modifier
            .fillMaxWidth()
            .statusBarsPadding()
            .height(76.dp)
            .padding(horizontal = 12.dp)
    ) {
        IconButton(
            modifier = Modifier.align(Alignment.CenterStart),
            onClick = onOpenHistory
        ) {
            Icon(
                imageVector = Icons.Default.Menu,
                contentDescription = "历史对话",
                tint = AppColors.TextPrimary
            )
        }

        Column(
            modifier = Modifier.align(Alignment.Center),
            horizontalAlignment = Alignment.CenterHorizontally
        ) {
            Text(
                text = "新对话",
                color = AppColors.TextPrimary,
                fontSize = 16.sp,
                fontWeight = FontWeight.Medium
            )
            Text(
                text = modelStatus,
                color = AppColors.TextHint,
                fontSize = 12.sp
            )
        }

        IconButton(
            modifier = Modifier.align(Alignment.CenterEnd),
            enabled = settingsEnabled,
            onClick = onOpenSettings
        ) {
            Text("⋮", color = AppColors.TextPrimary, fontSize = 28.sp)
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
            color = AppColors.Surface,
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
                            color = AppColors.TextDisabled,
                            fontSize = 16.sp
                        )
                    }

                    BasicTextField(
                        value = prompt,
                        enabled = enabled && !isGenerating,
                        onValueChange = onPromptChange,
                        singleLine = true,
                        textStyle = TextStyle(
                            color = AppColors.TextPrimary,
                            fontSize = 16.sp
                        ),
                        modifier = Modifier.fillMaxWidth()
                    )
                }

                Surface(
                    color = when {
                        isStopRequested -> AppColors.TextHint
                        isGenerating -> AppColors.StopAction
                        enabled && prompt.isNotBlank() -> AppColors.PrimaryAction
                        else -> AppColors.DisabledAction
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
                            Text("■", color = AppColors.Surface, fontSize = 12.sp)
                        } else {
                            Icon(
                                imageVector = Icons.Default.ArrowUpward,
                                contentDescription = "发送",
                                tint = AppColors.Surface
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
    messages: List<ChatMessage>,
    isGenerating: Boolean,
    isStopRequested: Boolean,
    errorMessage: String?
) {
    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(rememberScrollState())
            .padding(horizontal = 18.dp, vertical = 12.dp)
    ) {
        messages.filter { it.content.isNotBlank() }.forEach { message ->
            ChatMessageItem(message)
        }

        if (isGenerating) {
            Text(
                text = if (isStopRequested) "正在停止…" else "正在生成…",
                modifier = Modifier.padding(top = 20.dp),
                color = AppColors.TextTertiary,
                fontSize = 12.sp
            )
        }

        if (errorMessage != null) {
            Text(
                text = "生成失败：$errorMessage",
                modifier = Modifier.padding(top = 20.dp),
                color = AppColors.Error,
                fontSize = 12.sp
            )
        }
    }
}

@Composable
private fun ChatMessageItem(message: ChatMessage) {
    val isUserMessage = message.role == MessageRole.User

    Column(
        modifier = Modifier
            .fillMaxWidth()
            .padding(top = 20.dp),
        horizontalAlignment = if (isUserMessage) Alignment.End else Alignment.Start
    ) {
        if (isUserMessage) {
            // 用户消息固定靠右，并限制最大宽度，避免短消息的气泡显得过宽。
            Surface(
                modifier = Modifier.widthIn(max = 280.dp),
                shape = RoundedCornerShape(14.dp),
                color = AppColors.UserBubble
            ) {
                Text(
                    text = message.content,
                    modifier = Modifier.padding(horizontal = 12.dp, vertical = 8.dp),
                    color = AppColors.TextPrimary,
                    fontSize = 16.sp
                )
            }
        } else {
            Text(
                text = message.content,
                color = AppColors.TextPrimary,
                fontSize = 16.sp
            )
        }
    }
}

@Composable
private fun HistoryDrawer(onClose: () -> Unit) {
    ModalDrawerSheet(
        modifier = Modifier.fillMaxWidth(0.82f),
        drawerContainerColor = AppColors.Surface
    ) {
        Text(
            text = "历史对话",
            modifier = Modifier
                .statusBarsPadding()
                .padding(horizontal = 24.dp, vertical = 20.dp),
            color = AppColors.TextPrimary,
            fontSize = 20.sp,
            fontWeight = FontWeight.Medium
        )
        Text(
            text = "历史会话将在后续阶段加入。",
            modifier = Modifier.padding(horizontal = 24.dp),
            color = AppColors.TextTertiary,
            fontSize = 14.sp
        )
        OutlinedButton(
            modifier = Modifier.padding(horizontal = 24.dp, vertical = 16.dp),
            onClick = onClose
        ) {
            Text("关闭")
        }
    }
}

@Composable
private fun ConversationSettingsScreen(
    systemPrompt: String,
    finalPrompt: String,
    promptTokenCount: Int,
    trimmedHistoryTurnCount: Int,
    nCtx: Int?,
    requestedNctx: Int,
    isGenerating: Boolean,
    onSystemPromptChange: (String) -> Unit,
    onContextWindowChange: (Int) -> Unit,
    onClearConversation: () -> Unit,
    onResetContext: () -> Unit,
    onSelectModel: () -> Unit,
    onReleaseModel: () -> Unit,
    onBack: () -> Unit
) {
    var isSystemPromptDialogVisible by remember { mutableStateOf(false) }
    var isFinalPromptDialogVisible by remember { mutableStateOf(false) }
    var isContextWindowDialogVisible by remember { mutableStateOf(false) }
    var isClearConversationDialogVisible by remember { mutableStateOf(false) }
    var isResetContextDialogVisible by remember { mutableStateOf(false) }
    var isReleaseModelDialogVisible by remember { mutableStateOf(false) }
    var systemPromptDraft by remember { mutableStateOf(systemPrompt) }
    var contextWindowDraft by remember { mutableStateOf(requestedNctx) }

    Scaffold(
        containerColor = AppColors.SettingsBackground,
        topBar = {
            Box(
                modifier = Modifier
                    .fillMaxWidth()
                    .statusBarsPadding()
                    .height(64.dp)
                    .padding(horizontal = 12.dp)
            ) {
                IconButton(
                    modifier = Modifier.align(Alignment.CenterStart),
                    onClick = onBack
                ) {
                    Text("‹", color = AppColors.TextPrimary, fontSize = 30.sp)
                }
                Text(
                    text = "设置",
                    modifier = Modifier.align(Alignment.Center),
                    color = AppColors.TextPrimary,
                    fontSize = 16.sp,
                    fontWeight = FontWeight.Medium
                )
            }
        }
    ) { innerPadding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(innerPadding)
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 18.dp, vertical = 12.dp)
        ) {
            SettingsSection(title = "对话") {
                SettingsRow(
                    title = "System Prompt",
                    summary = if (systemPrompt.isBlank()) "未设置" else "已设置",
                    enabled = !isGenerating,
                    onClick = {
                        systemPromptDraft = systemPrompt
                        isSystemPromptDialogVisible = true
                    }
                )
                SettingsRow(
                    title = "最终 Prompt",
                    summary = if (finalPrompt.isBlank()) "暂无" else "查看",
                    enabled = finalPrompt.isNotBlank(),
                    onClick = { isFinalPromptDialogVisible = true }
                )
                SettingsRow(
                    title = "清空聊天消息",
                    summary = "保留模型",
                    enabled = !isGenerating && nCtx != null,
                    onClick = { isClearConversationDialogVisible = true }
                )
            }

            SettingsSection(
                title = "推理",
                modifier = Modifier.padding(top = 28.dp)
            ) {
                SettingsRow(
                    title = "上下文窗口",
                    summary = "n_ctx ${nCtx ?: requestedNctx}",
                    enabled = !isGenerating,
                    onClick = {
                        contextWindowDraft = nCtx ?: requestedNctx
                        isContextWindowDialogVisible = true
                    }
                )
                SettingsRow(
                    title = "上下文占用",
                    // Native 回调的 token 数包含 Chat Template 和特殊 Token，
                    // 即本轮真正 Prefill 到 Context 中的 Prompt 占用。
                    summary = if (promptTokenCount == 0) {
                        "暂无"
                    } else {
                        "$promptTokenCount / ${nCtx ?: requestedNctx} tokens"
                    },
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "历史裁剪",
                    summary = if (trimmedHistoryTurnCount == 0) {
                        "未触发"
                    } else {
                        "本轮省略 $trimmedHistoryTurnCount 轮"
                    },
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "重置 Context",
                    summary = "清空 KV Cache",
                    enabled = !isGenerating && nCtx != null,
                    onClick = { isResetContextDialogVisible = true }
                )
            }

            SettingsSection(
                title = "模型",
                modifier = Modifier.padding(top = 28.dp)
            ) {
                SettingsRow(
                    title = "切换本地模型",
                    enabled = !isGenerating,
                    onClick = onSelectModel
                )
                SettingsRow(
                    title = "卸载本地模型",
                    summary = "释放内存",
                    enabled = !isGenerating && nCtx != null,
                    onClick = { isReleaseModelDialogVisible = true }
                )
            }
        }
    }

    if (isSystemPromptDialogVisible) {
        AlertDialog(
            onDismissRequest = { isSystemPromptDialogVisible = false },
            containerColor = AppColors.Surface,
            title = { Text("System Prompt", fontSize = 16.sp) },
            text = {
                OutlinedTextField(
                    value = systemPromptDraft,
                    onValueChange = { systemPromptDraft = it },
                    modifier = Modifier.fillMaxWidth(),
                    placeholder = {
                        Text("例如：你是简洁的 Android 助手", fontSize = 14.sp)
                    },
                    minLines = 4,
                    maxLines = 8,
                    textStyle = TextStyle(fontSize = 14.sp)
                )
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        onSystemPromptChange(systemPromptDraft)
                        isSystemPromptDialogVisible = false
                    }
                ) {
                    Text("保存")
                }
            },
            dismissButton = {
                TextButton(onClick = { isSystemPromptDialogVisible = false }) {
                    Text("取消")
                }
            }
        )
    }

    if (isFinalPromptDialogVisible) {
        AlertDialog(
            onDismissRequest = { isFinalPromptDialogVisible = false },
            containerColor = AppColors.Surface,
            title = { Text("最终 Prompt", fontSize = 16.sp) },
            text = {
                Text(
                    text = finalPrompt,
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(280.dp)
                        .verticalScroll(rememberScrollState()),
                    color = AppColors.TextSecondary,
                    fontSize = 12.sp
                )
            },
            confirmButton = {
                TextButton(onClick = { isFinalPromptDialogVisible = false }) {
                    Text("关闭")
                }
            }
        )
    }

    if (isContextWindowDialogVisible) {
        AlertDialog(
            onDismissRequest = { isContextWindowDialogVisible = false },
            containerColor = AppColors.Surface,
            title = { Text("上下文窗口", fontSize = 16.sp) },
            text = {
                Column {
                    Text(
                        text = "修改会重建 Native Context 并清空 KV Cache；聊天消息不会删除。",
                        color = AppColors.TextTertiary,
                        fontSize = 12.sp
                    )
                    listOf(512, 1024, 2048, 4096).forEach { option ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable { contextWindowDraft = option }
                                .padding(top = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = contextWindowDraft == option,
                                onClick = { contextWindowDraft = option }
                            )
                            Text(
                                text = "n_ctx $option",
                                modifier = Modifier.padding(start = 8.dp),
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        onContextWindowChange(contextWindowDraft)
                        isContextWindowDialogVisible = false
                    }
                ) {
                    Text("确认")
                }
            },
            dismissButton = {
                TextButton(onClick = { isContextWindowDialogVisible = false }) {
                    Text("取消")
                }
            }
        )
    }

    if (isClearConversationDialogVisible) {
        ConfirmActionDialog(
            title = "清空聊天消息？",
            message = "只清空当前页面的聊天记录；模型和 Native Context 会继续保留。",
            confirmText = "清空",
            onConfirm = {
                onClearConversation()
                isClearConversationDialogVisible = false
            },
            onDismiss = { isClearConversationDialogVisible = false }
        )
    }

    if (isResetContextDialogVisible) {
        ConfirmActionDialog(
            title = "重置 Context？",
            message = "这会清空 Native KV Cache，但不会删除当前聊天消息或卸载模型。",
            confirmText = "重置",
            onConfirm = {
                onResetContext()
                isResetContextDialogVisible = false
            },
            onDismiss = { isResetContextDialogVisible = false }
        )
    }

    if (isReleaseModelDialogVisible) {
        ConfirmActionDialog(
            title = "卸载本地模型？",
            message = "这会释放 Native Context、模型权重和 Runtime，并取消下次启动时自动恢复该模型。",
            confirmText = "卸载",
            onConfirm = {
                onReleaseModel()
                isReleaseModelDialogVisible = false
            },
            onDismiss = { isReleaseModelDialogVisible = false }
        )
    }
}

@Composable
private fun ConfirmActionDialog(
    title: String,
    message: String,
    confirmText: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = AppColors.Surface,
        title = { Text(title, fontSize = 16.sp) },
        text = {
            Text(
                text = message,
                color = AppColors.TextTertiary,
                fontSize = 12.sp
            )
        },
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text(confirmText)
            }
        },
        dismissButton = {
            TextButton(onClick = onDismiss) {
                Text("取消")
            }
        }
    )
}

@Composable
private fun SettingsSection(
    title: String,
    modifier: Modifier = Modifier,
    content: @Composable () -> Unit
) {
    Column(modifier = modifier.fillMaxWidth()) {
        Text(
            text = title,
            modifier = Modifier.padding(start = 12.dp, bottom = 12.dp),
            color = AppColors.TextHint,
            fontSize = 14.sp
        )
        Surface(
            modifier = Modifier.fillMaxWidth(),
            color = AppColors.Surface,
            shape = RoundedCornerShape(24.dp)
        ) {
            Column {
                content()
            }
        }
    }
}

@Composable
private fun SettingsRow(
    title: String,
    summary: String? = null,
    enabled: Boolean = true,
    showChevron: Boolean = true,
    clickable: Boolean = true,
    onClick: () -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .then(
                if (clickable) {
                    Modifier.clickable(enabled = enabled, onClick = onClick)
                } else {
                    Modifier
                }
            )
            .padding(horizontal = 20.dp, vertical = 13.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Text(
            text = title,
            modifier = Modifier.weight(1f),
            color = if (enabled) AppColors.TextPrimary else AppColors.TextDisabled,
            fontSize = 16.sp
        )
        if (summary != null) {
            Text(
            text = summary,
            modifier = Modifier.align(Alignment.CenterVertically),
            color = AppColors.TextHint,
            fontSize = 14.sp,
            lineHeight = 20.sp
            )
        }
        if (showChevron) {
            Text(
                text = "›",
                modifier = Modifier
                    .padding(start = 10.dp)
                    .offset(y = (-2).dp),
                color = AppColors.Chevron,
                fontSize = 22.sp,
                lineHeight = 22.sp
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
        Text(title, color = AppColors.TextTertiary, fontSize = 14.sp)
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
        color = AppColors.TextTertiary,
        fontSize = 12.sp
    )
}
