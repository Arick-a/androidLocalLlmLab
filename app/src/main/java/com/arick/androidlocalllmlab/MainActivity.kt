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
import androidx.compose.foundation.interaction.MutableInteractionSource
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
import androidx.compose.material.icons.filled.KeyboardArrowDown
import androidx.compose.material.icons.filled.KeyboardArrowUp
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.DrawerValue
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberDrawerState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.text.SpanStyle
import androidx.compose.ui.text.buildAnnotatedString
import androidx.compose.ui.text.withStyle
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.foundation.text.BasicTextField
import com.arick.androidlocalllmlab.ui.theme.AppColors
import com.arick.androidlocalllmlab.ui.theme.AndroidLocalLlmLabTheme
import kotlinx.coroutines.launch
import kotlinx.coroutines.flow.collect
import java.util.Locale
import kotlin.math.roundToInt

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
    val cpuThreads by viewModel.cpuThreads.collectAsState()
    val inferenceConfig by viewModel.inferenceConfig.collectAsState()
    val thinkingEnabled by viewModel.thinkingEnabled.collectAsState()
    val generationProgress by viewModel.generationProgress.collectAsState()
    val generationMetrics by viewModel.generationMetrics.collectAsState()
    val modelPreparationMetrics by viewModel.modelPreparationMetrics.collectAsState()
    val memorySnapshots by viewModel.memorySnapshots.collectAsState()
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
    val supportsQwen3ThinkingSwitch = (uiState as? ChatUiState.Ready)?.let { state ->
        val identity = "${state.modelName} ${state.modelDescription}"
        identity.contains("qwen3", ignoreCase = true) &&
                !identity.contains("instruct-2507", ignoreCase = true)
    } == true

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
            cpuThreads = cpuThreads,
            inferenceConfig = inferenceConfig,
            supportsQwen3ThinkingSwitch = supportsQwen3ThinkingSwitch,
            thinkingEnabled = thinkingEnabled,
            generationMetrics = generationMetrics,
            modelPreparationMetrics = modelPreparationMetrics,
            memorySnapshots = memorySnapshots,
            isGenerating = isGenerating,
            onSystemPromptChange = viewModel::updateSystemPrompt,
            onContextWindowChange = viewModel::updateContextWindow,
            onCpuThreadsChange = viewModel::updateCpuThreads,
            onInferenceConfigChange = viewModel::updateInferenceConfig,
            onThinkingEnabledChange = viewModel::updateThinkingEnabled,
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
                            generationProgress = generationProgress,
                            errorMessage = generationError,
                            onToggleReasoning = viewModel::toggleReasoning
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
    generationProgress: String?,
    errorMessage: String?,
    onToggleReasoning: (Int) -> Unit
) {
    val scrollState = rememberScrollState()
    val coroutineScope = rememberCoroutineScope()
    var shouldFollowBottom by remember { mutableStateOf(true) }
    var isAutoScrolling by remember { mutableStateOf(false) }

    // 用户手势向上查看历史时暂停跟随；自己滚到底部后自动恢复。
    LaunchedEffect(scrollState) {
        snapshotFlow { scrollState.isScrollInProgress to scrollState.canScrollForward }
            .collect { (isScrolling, canScrollForward) ->
                if (isScrolling && canScrollForward && !isAutoScrolling) {
                    shouldFollowBottom = false
                }
                if (!canScrollForward) {
                    shouldFollowBottom = true
                }
            }
    }

    // messages 在每个流式 Token 到来时更新。仅在用户仍停留底部时，才跟随新内容。
    LaunchedEffect(messages, generationProgress, shouldFollowBottom) {
        if (shouldFollowBottom) {
            isAutoScrolling = true
            scrollState.scrollTo(scrollState.maxValue)
            isAutoScrolling = false
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(scrollState)
            .padding(horizontal = 18.dp, vertical = 12.dp)
    ) {
        messages.forEachIndexed { index, message ->
            if (message.content.isNotBlank() || message.reasoningContent.isNotBlank()) {
                ChatMessageItem(
                    message = message,
                    onToggleReasoning = { onToggleReasoning(index) }
                )
            }
        }

        if (isGenerating) {
            Text(
                text = if (isStopRequested) "正在停止…" else generationProgress ?: "正在生成…",
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

        if (!shouldFollowBottom && scrollState.canScrollForward) {
            Surface(
                modifier = Modifier
                    .align(Alignment.BottomEnd)
                    .padding(end = 20.dp, bottom = 12.dp),
                color = AppColors.Surface,
                shape = CircleShape,
                shadowElevation = 4.dp
            ) {
                IconButton(
                    modifier = Modifier.size(40.dp),
                    onClick = {
                        coroutineScope.launch {
                            shouldFollowBottom = true
                            isAutoScrolling = true
                            scrollState.animateScrollTo(scrollState.maxValue)
                            isAutoScrolling = false
                        }
                    }
                ) {
                    Icon(
                        imageVector = Icons.Default.KeyboardArrowDown,
                        contentDescription = "回到最新消息",
                        tint = AppColors.TextPrimary
                    )
                }
            }
        }
    }
}

@Composable
private fun ChatMessageItem(
    message: ChatMessage,
    onToggleReasoning: () -> Unit
) {
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
            if (message.reasoningContent.isNotBlank()) {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .clickable(
                            interactionSource = remember { MutableInteractionSource() },
                            indication = null,
                            onClick = onToggleReasoning
                        )
                ) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            text = "思考过程",
                            color = AppColors.TextTertiary,
                            fontSize = 12.sp
                        )
                        message.reasoningDurationMillis?.let { durationMillis ->
                            Text(
                                text = " ${formatDuration(durationMillis)}",
                                color = AppColors.TextHint,
                                fontSize = 12.sp
                            )
                        }
                        Icon(
                            imageVector = if (message.isReasoningExpanded) {
                                Icons.Default.KeyboardArrowUp
                            } else {
                                Icons.Default.KeyboardArrowDown
                            },
                            contentDescription = if (message.isReasoningExpanded) "折叠思考过程" else "展开思考过程",
                            tint = AppColors.TextHint,
                            modifier = Modifier
                                .padding(start = 2.dp)
                                .size(16.dp)
                        )
                    }
                    if (message.isReasoningExpanded) {
                        Text(
                            text = message.reasoningContent,
                            modifier = Modifier.padding(top = 6.dp),
                            color = AppColors.TextSecondary,
                            fontSize = 12.sp
                        )
                    }
                }
            }
            if (message.content.isNotBlank()) {
                MarkdownText(
                    markdown = message.content,
                    modifier = Modifier.padding(top = if (message.reasoningContent.isBlank()) 0.dp else 10.dp),
                )
            }
        }
    }
}

/**
 * 轻量 Markdown：块级处理标题、列表和分割线；行内处理粗体、斜体和代码。
 * 不引入第三方库，避免把模型输出的 `---` 当作普通正文显示。
 */
@Composable
private fun MarkdownText(markdown: String, modifier: Modifier = Modifier) {
    Column(modifier = modifier) {
        markdown.lines().forEach { line ->
            val trimmedLine = line.trim()
            when {
                trimmedLine.isHorizontalRule() -> {
                    HorizontalDivider(
                        modifier = Modifier.padding(vertical = 12.dp),
                        color = AppColors.TextHint
                    )
                }

                trimmedLine.isMarkdownHeading() -> {
                    MarkdownInlineText(
                        markdown = trimmedLine.dropWhile { it == '#' }.trimStart(),
                        modifier = Modifier.padding(top = 8.dp, bottom = 4.dp),
                        fontWeight = FontWeight.SemiBold
                    )
                }

                trimmedLine.isMarkdownBullet() -> {
                    Row(modifier = Modifier.padding(top = 2.dp)) {
                        Text("•", color = AppColors.TextPrimary, fontSize = 16.sp)
                        MarkdownInlineText(
                            markdown = trimmedLine.drop(2),
                            modifier = Modifier.padding(start = 8.dp)
                        )
                    }
                }

                trimmedLine.isEmpty() -> Text("", modifier = Modifier.height(8.dp))
                else -> MarkdownInlineText(markdown = line)
            }
        }
    }
}

@Composable
private fun MarkdownInlineText(
    markdown: String,
    modifier: Modifier = Modifier,
    fontWeight: FontWeight? = null
) {
    val text = remember(markdown) { markdown.toMarkdownAnnotatedString() }
    Text(
        text = text,
        modifier = modifier,
        color = AppColors.TextPrimary,
        fontSize = 16.sp,
        fontWeight = fontWeight
    )
}

private fun String.isHorizontalRule(): Boolean =
    length >= 3 && all { it == '-' || it == '*' || it == '_' }

private fun String.isMarkdownHeading(): Boolean =
    startsWith("#") && dropWhile { it == '#' }.startsWith(" ")

private fun String.isMarkdownBullet(): Boolean =
    (startsWith("- ") || startsWith("* ")) && length > 2

private fun String.toMarkdownAnnotatedString(): AnnotatedString = buildAnnotatedString {
    val source = this@toMarkdownAnnotatedString
    var index = 0
    while (index < source.length) {
        val marker = when {
            source.startsWith("**", index) -> "**"
            source.startsWith("`", index) -> "`"
            source.startsWith("*", index) -> "*"
            else -> null
        }
        if (marker == null) {
            append(this@toMarkdownAnnotatedString[index++])
            continue
        }

        val contentStart = index + marker.length
        val contentEnd = source.indexOf(marker, contentStart)
        if (contentEnd < 0 || contentEnd == contentStart) {
            append(this@toMarkdownAnnotatedString[index++])
            continue
        }
        val style = when (marker) {
            "**" -> SpanStyle(fontWeight = FontWeight.SemiBold)
            "`" -> SpanStyle(fontFamily = FontFamily.Monospace)
            else -> SpanStyle(fontStyle = androidx.compose.ui.text.font.FontStyle.Italic)
        }
        withStyle(style) {
            append(source.substring(contentStart, contentEnd))
        }
        index = contentEnd + marker.length
    }
}

private fun formatDuration(durationMillis: Long): String = when {
    durationMillis < 1_000L -> "${durationMillis}ms"
    else -> String.format(Locale.US, "%.1fs", durationMillis / 1_000.0)
}

private fun formatBytes(bytes: Long): String = when {
    bytes < 1024L * 1024L -> "${bytes / 1024} KB"
    else -> String.format(Locale.US, "%.1f MB", bytes / (1024.0 * 1024.0))
}

private fun formatKb(kb: Int): String = formatBytes(kb.toLong() * 1024L)

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
    cpuThreads: Int,
    inferenceConfig: InferenceConfig,
    supportsQwen3ThinkingSwitch: Boolean,
    thinkingEnabled: Boolean,
    generationMetrics: GenerationMetrics?,
    modelPreparationMetrics: ModelPreparationMetrics?,
    memorySnapshots: RuntimeMemorySnapshots,
    isGenerating: Boolean,
    onSystemPromptChange: (String) -> Unit,
    onContextWindowChange: (Int) -> Unit,
    onCpuThreadsChange: (Int) -> Unit,
    onInferenceConfigChange: (InferenceConfig) -> Unit,
    onThinkingEnabledChange: (Boolean) -> Unit,
    onClearConversation: () -> Unit,
    onResetContext: () -> Unit,
    onSelectModel: () -> Unit,
    onReleaseModel: () -> Unit,
    onBack: () -> Unit
) {
    var isSystemPromptDialogVisible by remember { mutableStateOf(false) }
    var isFinalPromptDialogVisible by remember { mutableStateOf(false) }
    var isContextWindowDialogVisible by remember { mutableStateOf(false) }
    var isCpuThreadsDialogVisible by remember { mutableStateOf(false) }
    var isSamplingDialogVisible by remember { mutableStateOf(false) }
    var isClearConversationDialogVisible by remember { mutableStateOf(false) }
    var isResetContextDialogVisible by remember { mutableStateOf(false) }
    var isReleaseModelDialogVisible by remember { mutableStateOf(false) }
    var systemPromptDraft by remember { mutableStateOf(systemPrompt) }
    var contextWindowDraft by remember { mutableStateOf(requestedNctx) }
    var samplingDraft by remember { mutableStateOf(inferenceConfig) }

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
                    title = "CPU 线程",
                    summary = "$cpuThreads 线程",
                    enabled = !isGenerating && nCtx != null,
                    onClick = { isCpuThreadsDialogVisible = true }
                )
                SettingsSwitchRow(
                    title = "Qwen3 思考",
                    summary = if (supportsQwen3ThinkingSwitch) {
                        "开启后先生成推理过程"
                    } else {
                        "当前模型固定非思考"
                    },
                    checked = thinkingEnabled,
                    enabled = !isGenerating && supportsQwen3ThinkingSwitch,
                    onCheckedChange = onThinkingEnabledChange
                )
                SettingsRow(
                    title = "采样参数",
                    summary = if (inferenceConfig.isGreedy) {
                        "稳定 · ${inferenceConfig.maxTokens} tokens"
                    } else {
                        "采样 · ${inferenceConfig.maxTokens} tokens"
                    },
                    enabled = !isGenerating && nCtx != null,
                    onClick = {
                        samplingDraft = inferenceConfig
                        isSamplingDialogVisible = true
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
                title = "性能",
                modifier = Modifier.padding(top = 28.dp)
            ) {
                SettingsRow(
                    title = "模型加载",
                    summary = modelPreparationMetrics?.modelLoadMillis?.let { "$it ms" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Context 创建",
                    summary = modelPreparationMetrics?.contextCreateMillis?.let { "$it ms" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Java Heap",
                    summary = memorySnapshots.current?.let {
                        "${formatBytes(it.javaUsedBytes)} / ${formatBytes(it.javaMaxBytes)}"
                    } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Native Heap",
                    summary = memorySnapshots.current?.nativeAllocatedBytes?.let(::formatBytes) ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Native PSS",
                    summary = memorySnapshots.current?.nativePssKb?.let(::formatKb) ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "模型后 Native PSS",
                    summary = memorySnapshots.afterModelLoaded?.nativePssKb?.let(::formatKb) ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Context 后 Native PSS",
                    summary = memorySnapshots.afterContextCreated?.nativePssKb?.let(::formatKb) ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "首 Token 等待",
                    summary = generationMetrics?.firstTokenMillis?.let { "$it ms" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Prefill",
                    summary = generationMetrics?.prefillMillis?.let { "$it ms" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "KV Cache 复用",
                    summary = generationMetrics?.let { "${it.reusedPromptTokenCount} tokens" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "Decode",
                    summary = generationMetrics?.let {
                        "${formatTokensPerSecond(it.decodeTokensPerSecond)} tok/s"
                    } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "本轮生成",
                    summary = generationMetrics?.generatedTokenCount?.let { "$it tokens" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
                )
                SettingsRow(
                    title = "本轮总耗时",
                    summary = generationMetrics?.totalMillis?.let { "$it ms" } ?: "暂无",
                    showChevron = false,
                    clickable = false,
                    onClick = {}
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
                        text = "修改会重建 Native Context 并清空 KV Cache；8K 适合普通多轮，16K 仅建议内存充足的设备。",
                        color = AppColors.TextTertiary,
                        fontSize = 12.sp
                    )
                    listOf(4096, 8192, 12288, 16384).forEach { option ->
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

    if (isCpuThreadsDialogVisible) {
        AlertDialog(
            onDismissRequest = { isCpuThreadsDialogVisible = false },
            containerColor = AppColors.Surface,
            title = { Text("CPU 线程", fontSize = 16.sp) },
            text = {
                Column {
                    Text(
                        text = "同时影响 Prompt 的 Prefill 和逐 Token Decode。线程更多不一定更快，请用性能数据对照。",
                        color = AppColors.TextTertiary,
                        fontSize = 12.sp
                    )
                    listOf(1, 2, 4, 6, 8).forEach { option ->
                        Row(
                            modifier = Modifier
                                .fillMaxWidth()
                                .clickable {
                                    onCpuThreadsChange(option)
                                    isCpuThreadsDialogVisible = false
                                }
                                .padding(top = 8.dp),
                            verticalAlignment = Alignment.CenterVertically
                        ) {
                            RadioButton(
                                selected = cpuThreads == option,
                                onClick = {
                                    onCpuThreadsChange(option)
                                    isCpuThreadsDialogVisible = false
                                }
                            )
                            Text(
                                text = "$option 线程",
                                modifier = Modifier.padding(start = 8.dp),
                                fontSize = 14.sp
                            )
                        }
                    }
                }
            },
            confirmButton = {
                TextButton(onClick = { isCpuThreadsDialogVisible = false }) {
                    Text("关闭")
                }
            }
        )
    }

    if (isSamplingDialogVisible) {
        AlertDialog(
            onDismissRequest = { isSamplingDialogVisible = false },
            containerColor = AppColors.Surface,
            title = { Text("采样参数", fontSize = 16.sp) },
            text = {
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .height(390.dp)
                        .verticalScroll(rememberScrollState())
                ) {
                    Text(
                        text = "Temperature 为 0 时使用稳定的 greedy 模式；此时 Top-K、Top-P、Min-P、Repeat Penalty 与 Seed 不参与选词。",
                        color = AppColors.TextTertiary,
                        fontSize = 12.sp
                    )
                    Text(
                        text = "预设",
                        modifier = Modifier.padding(top = 12.dp),
                        color = AppColors.TextPrimary,
                        fontSize = 14.sp
                    )
                    SamplingPreset.entries.chunked(2).forEach { row ->
                        Row(modifier = Modifier.fillMaxWidth()) {
                            row.forEach { preset ->
                                OutlinedButton(
                                    modifier = Modifier
                                        .weight(1f)
                                        .padding(end = 6.dp, top = 4.dp),
                                    onClick = {
                                        samplingDraft = preset.configFor(nCtx ?: requestedNctx)
                                    }
                                ) {
                                    Text(preset.label, fontSize = 12.sp)
                                }
                            }
                            if (row.size == 1) {
                                Box(modifier = Modifier.weight(1f))
                            }
                        }
                    }
                    SamplingSlider("Temperature", listOf("0（稳定）" to 0f, "0.2" to 0.2f, "0.7" to 0.7f, "1.0" to 1.0f), samplingDraft.temperature) {
                        samplingDraft = samplingDraft.copy(temperature = it)
                    }
                    SamplingSlider("Top-K", listOf("20" to 20, "40" to 40, "80" to 80, "不限制" to 0), samplingDraft.topK) {
                        samplingDraft = samplingDraft.copy(topK = it)
                    }
                    SamplingSlider("Top-P", listOf("0.8" to 0.8f, "0.9" to 0.9f, "0.95" to 0.95f, "1.0" to 1.0f), samplingDraft.topP) {
                        samplingDraft = samplingDraft.copy(topP = it)
                    }
                    SamplingSlider("Min-P", listOf("0" to 0f, "0.05" to 0.05f, "0.1" to 0.1f), samplingDraft.minP) {
                        samplingDraft = samplingDraft.copy(minP = it)
                    }
                    SamplingSlider("Repeat Penalty", listOf("1.0" to 1.0f, "1.05" to 1.05f, "1.1" to 1.1f, "1.2" to 1.2f), samplingDraft.repeatPenalty) {
                        samplingDraft = samplingDraft.copy(repeatPenalty = it)
                    }
                    SamplingSlider("Seed", listOf("42" to 42, "2026" to 2026, "随机" to -1), samplingDraft.seed) {
                        samplingDraft = samplingDraft.copy(seed = it)
                    }
                    SamplingSlider(
                        title = "Max Tokens",
                        options = listOf(2048, 4096, 6000)
                            .filter { it < (nCtx ?: requestedNctx) }
                            .map { "$it" to it },
                        selected = samplingDraft.maxTokens
                    ) {
                        samplingDraft = samplingDraft.copy(maxTokens = it)
                    }
                }
            },
            confirmButton = {
                TextButton(
                    onClick = {
                        onInferenceConfigChange(samplingDraft)
                        isSamplingDialogVisible = false
                    }
                ) {
                    Text("保存")
                }
            },
            dismissButton = {
                TextButton(onClick = { isSamplingDialogVisible = false }) {
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
private fun <T> SamplingSlider(
    title: String,
    options: List<Pair<String, T>>,
    selected: T,
    onSelect: (T) -> Unit
) {
    val selectedIndex = options.indexOfFirst { (_, value) -> value == selected }
        .coerceAtLeast(0)
    val currentLabel = options[selectedIndex].first

    Column(modifier = Modifier.padding(top = 14.dp)) {
        Row(modifier = Modifier.fillMaxWidth()) {
            Text(
                text = title,
                modifier = Modifier.weight(1f),
                color = AppColors.TextSecondary,
                fontSize = 14.sp
            )
            Text(
                text = currentLabel,
                color = AppColors.TextHint,
                fontSize = 14.sp
            )
        }
        Slider(
            value = selectedIndex.toFloat(),
            onValueChange = { position -> onSelect(options[position.roundToInt()].second) },
            valueRange = 0f..(options.lastIndex.toFloat()),
            // 不显示 Material 默认的离散刻度线；值仍会在 onValueChange 时吸附到预设档位。
            steps = 0,
            modifier = Modifier.fillMaxWidth()
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

private fun formatTokensPerSecond(value: Double): String =
    String.format(Locale.US, "%.1f", value)

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
private fun SettingsSwitchRow(
    title: String,
    summary: String,
    checked: Boolean,
    enabled: Boolean,
    onCheckedChange: (Boolean) -> Unit
) {
    Row(
        modifier = Modifier
            .fillMaxWidth()
            .clickable(enabled = enabled) { onCheckedChange(!checked) }
            .padding(horizontal = 20.dp, vertical = 10.dp),
        verticalAlignment = Alignment.CenterVertically
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(
                text = title,
                color = if (enabled) AppColors.TextPrimary else AppColors.TextDisabled,
                fontSize = 16.sp
            )
            Text(
                text = summary,
                modifier = Modifier.padding(top = 2.dp),
                color = AppColors.TextHint,
                fontSize = 12.sp
            )
        }
        Switch(
            checked = checked,
            enabled = enabled,
            onCheckedChange = onCheckedChange
        )
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
