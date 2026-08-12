package com.arick.androidlocalllmlab

import android.content.Context
import android.content.Intent
import android.provider.Settings
import org.json.JSONObject
import java.net.HttpURLConnection
import java.net.URL
import java.net.URLEncoder
import java.nio.charset.StandardCharsets

data class ToolCall(
    val name: String,
    val arguments: JSONObject
)

data class SourceLink(
    val title: String,
    val url: String
)

sealed interface ToolResult {
    data class Success(
        val content: String,
        val sources: List<SourceLink> = emptyList()
    ) : ToolResult

    data class Failure(val message: String) : ToolResult
}

/**
 * 工具定义同时描述模型可见的协议和 Kotlin 侧实际执行入口。
 * 这里的 schema 是轻量元数据：用于生成提示词和让新增工具的参数边界集中可见；
 * 每个工具仍自行完成与业务语义有关的校验。
 */
data class ToolParameterSchema(
    val name: String,
    val description: String,
    val allowedValues: List<String> = emptyList(),
    val maxLength: Int? = null
)

interface ToolDefinition {
    val name: String
    val description: String
    val parameterSchema: List<ToolParameterSchema>

    fun execute(arguments: JSONObject, onProgress: (String) -> Unit): ToolResult
}

/** 显式白名单注册表：模型请求的名字只能命中这里已经注册的工具。 */
class ToolRegistry(definitions: List<ToolDefinition>) {
    private val definitionsByName = definitions.associateBy(ToolDefinition::name)

    init {
        require(definitionsByName.size == definitions.size) { "工具名称不能重复" }
    }

    fun execute(call: ToolCall, onProgress: (String) -> Unit = {}): ToolResult =
        definitionsByName[call.name]?.execute(call.arguments, onProgress)
            ?: ToolResult.Failure("不允许调用工具：${call.name}")

    fun modelInstruction(): String = definitionsByName.values.joinToString("\n") { tool ->
        val parameters = tool.parameterSchema.joinToString("；") { parameter ->
            buildString {
                append("${parameter.name}：${parameter.description}")
                parameter.allowedValues.takeIf { it.isNotEmpty() }?.let {
                    append("，仅允许 ${it.joinToString(" 或 ")}")
                }
                parameter.maxLength?.let { append("，最多 $it 个字符") }
            }
        }
        "- ${tool.name}：${tool.description}。参数：$parameters"
    }
}

/** Kotlin 是工具权限边界：模型只能请求，不能直接执行 Intent。 */
class AndroidToolExecutor(private val context: Context) {
    private val registry = ToolRegistry(
        listOf(
            object : ToolDefinition {
                override val name = "open_settings"
                override val description = "打开 App 无法直接修改的 Android 设置页面"
                override val parameterSchema = listOf(
                    ToolParameterSchema("page", "要打开的设置页", allowedValues = listOf("wifi", "system"))
                )

                override fun execute(arguments: JSONObject, onProgress: (String) -> Unit) = openSettings(arguments)
            },
            object : ToolDefinition {
                override val name = "get_weather"
                override val description = "查询城市当前天气和未来三天预报；不能用于活动或场馆状态"
                override val parameterSchema = listOf(
                    ToolParameterSchema("city", "明确的城市名", maxLength = 40)
                )

                override fun execute(arguments: JSONObject, onProgress: (String) -> Unit) = getWeather(arguments, onProgress)
            },
            object : ToolDefinition {
                override val name = "web_search"
                override val description = "搜索实时或外部可验证的网页信息"
                override val parameterSchema = listOf(
                    ToolParameterSchema("query", "用户问题对应的简洁检索词", maxLength = 240)
                )

                override fun execute(arguments: JSONObject, onProgress: (String) -> Unit) = webSearch(arguments, onProgress)
            }
        )
    )

    fun execute(call: ToolCall, onProgress: (String) -> Unit = {}): ToolResult =
        registry.execute(call, onProgress)

    fun modelInstruction(): String = registry.modelInstruction()

    private fun openSettings(arguments: JSONObject): ToolResult {
        val page = arguments.optString("page")
        val action = when (page) {
            "wifi" -> Settings.ACTION_WIFI_SETTINGS
            "system" -> Settings.ACTION_SETTINGS
            else -> return ToolResult.Failure("open_settings 的 page 仅允许 wifi 或 system")
        }

        return runCatching {
            context.startActivity(Intent(action).addFlags(Intent.FLAG_ACTIVITY_NEW_TASK))
            ToolResult.Success("已打开 ${if (page == "wifi") "Wi-Fi" else "系统"} 设置页")
        }.getOrElse { error ->
            ToolResult.Failure("无法打开设置页：${error.message ?: error::class.java.simpleName}")
        }
    }

    /**
     * 仅在客户端请求 Open-Meteo 的公开接口：城市名先转坐标，再查天气。
     * 网络结果被压缩为短文本后再交给模型，避免整段 API JSON 挤占 Context。
     */
    private fun getWeather(arguments: JSONObject, onProgress: (String) -> Unit): ToolResult {
        val city = arguments.optString("city").trim()
        if (city.isEmpty() || city.length > 40) {
            return ToolResult.Failure("get_weather 的 city 必须是 1 到 40 个字符的城市名")
        }
        return runCatching {
            onProgress("正在查询 $city 的位置…")
            val encodedCity = URLEncoder.encode(city, StandardCharsets.UTF_8.name())
            val locationJson = requestJson(
                "https://geocoding-api.open-meteo.com/v1/search?name=$encodedCity&count=1&language=zh&format=json"
            )
            val location = locationJson.optJSONArray("results")?.optJSONObject(0)
                ?: return ToolResult.Failure("未找到“$city”，请换成更明确的城市名")
            val latitude = location.getDouble("latitude")
            val longitude = location.getDouble("longitude")
            val locationName = listOfNotNull(
                location.optString("name").takeIf(String::isNotBlank),
                location.optString("admin1").takeIf(String::isNotBlank),
                location.optString("country").takeIf(String::isNotBlank)
            ).distinct().joinToString("，")

            onProgress("正在获取 $locationName 的天气…")
            val weather = requestJson(
                "https://api.open-meteo.com/v1/forecast" +
                    "?latitude=$latitude&longitude=$longitude" +
                    "&current=temperature_2m,apparent_temperature,weather_code,wind_speed_10m" +
                    "&daily=weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max" +
                    "&timezone=auto&forecast_days=3"
            )
            val current = weather.getJSONObject("current")
            val daily = weather.getJSONObject("daily")
            val dates = daily.getJSONArray("time")
            val codes = daily.getJSONArray("weather_code")
            val highs = daily.getJSONArray("temperature_2m_max")
            val lows = daily.getJSONArray("temperature_2m_min")
            val precipitationProbabilities = daily.getJSONArray("precipitation_probability_max")
            val forecast = buildString {
                for (index in 0 until dates.length()) {
                    if (index > 0) append("；")
                    append(dates.getString(index))
                    append(' ')
                    append(weatherDescription(codes.getInt(index)))
                    append("，")
                    append(formatTemperature(lows.getDouble(index)))
                    append("～")
                    append(formatTemperature(highs.getDouble(index)))
                    append("，降水概率 ")
                    append(precipitationProbabilities.getInt(index))
                    append('%')
                }
            }
            ToolResult.Success(
                "天气查询成功。地点：$locationName。当前：${formatTemperature(current.getDouble("temperature_2m"))}" +
                    "，体感 ${formatTemperature(current.getDouble("apparent_temperature"))}" +
                    "，${weatherDescription(current.getInt("weather_code"))}" +
                    "，风速 ${current.getDouble("wind_speed_10m").toInt()} km/h。" +
                    "未来三天：$forecast。数据来源：Open-Meteo，仅供参考。"
            )
        }.getOrElse { error ->
            ToolResult.Failure("天气查询失败：${error.message ?: "网络请求异常"}")
        }
    }

    /**
     * 等价于 Tavily 的 curl POST 示例。只把前五条的标题、摘要、链接回填给模型，
     * 防止完整网页内容耗尽本地模型的 Context。
     */
    private fun webSearch(arguments: JSONObject, onProgress: (String) -> Unit): ToolResult {
        val query = arguments.optString("query").trim()
        if (query.isEmpty() || query.length > 240) {
            return ToolResult.Failure("web_search 的 query 必须是 1 到 240 个字符")
        }
        val apiKey = BuildConfig.TAVILY_API_KEY.trim()
        if (apiKey.isEmpty()) {
            return ToolResult.Failure("未配置 Tavily Key，请在本机 local.properties 设置 tavily.api.key 后重新构建")
        }

        return runCatching {
            onProgress("正在联网搜索：$query")
            val response = postJson(
                url = "https://api.tavily.com/search",
                body = JSONObject()
                    .put("query", query)
                    // 与你验证成功的 curl 示例保持一致；advanced 每次会消耗 2 个 Tavily Credits。
                    .put("search_depth", "advanced")
                    .put("max_results", 3)
                    .put("include_answer", false)
                    .toString(),
                bearerToken = apiKey
            )
            val results = response.optJSONArray("results")
            if (results == null || results.length() == 0) {
                return ToolResult.Success("网页搜索完成，但没有找到可用结果。")
            }
            val resultCount = minOf(results.length(), 3)
            onProgress("已获得 $resultCount 条搜索结果")
            val sources = buildList {
                for (index in 0 until resultCount) {
                    val result = results.optJSONObject(index) ?: continue
                    val url = result.optString("url").trim()
                    if (url.isNotEmpty()) {
                        add(SourceLink(result.optString("title", "搜索结果 ${index + 1}"), url))
                    }
                }
            }
            val compactResults = buildString {
                for (index in 0 until resultCount) {
                    val result = results.optJSONObject(index) ?: continue
                    if (isNotEmpty()) append("\n\n")
                    append("[${index + 1}] ")
                    append(result.optString("title", "无标题"))
                    append("\n摘要：")
                    append(result.optString("content").replace(Regex("\\s+"), " ").take(240))
                }
            }
            ToolResult.Success(
                content = "网页搜索成功，查询：$query。以下是可引用的搜索结果：\n$compactResults",
                sources = sources
            )
        }.getOrElse { error ->
            ToolResult.Failure("网页搜索失败：${error.message ?: "网络请求异常"}")
        }
    }

    private fun requestJson(url: String): JSONObject {
        val connection = (URL(url).openConnection() as HttpURLConnection)
        return try {
            connection.requestMethod = "GET"
            connection.connectTimeout = 10_000
            connection.readTimeout = 10_000
            connection.setRequestProperty("Accept", "application/json")
            check(connection.responseCode in 200..299) { "天气服务返回 HTTP ${connection.responseCode}" }
            JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
        } finally {
            connection.disconnect()
        }
    }

    private fun postJson(url: String, body: String, bearerToken: String): JSONObject {
        val connection = (URL(url).openConnection() as HttpURLConnection)
        return try {
            connection.requestMethod = "POST"
            connection.connectTimeout = 10_000
            connection.readTimeout = 20_000
            connection.doOutput = true
            connection.setRequestProperty("Content-Type", "application/json")
            connection.setRequestProperty("Accept", "application/json")
            connection.setRequestProperty("Authorization", "Bearer $bearerToken")
            connection.outputStream.bufferedWriter(StandardCharsets.UTF_8).use { writer ->
                writer.write(body)
            }
            check(connection.responseCode in 200..299) { "Tavily 返回 HTTP ${connection.responseCode}" }
            JSONObject(connection.inputStream.bufferedReader().use { it.readText() })
        } finally {
            connection.disconnect()
        }
    }

    private fun formatTemperature(value: Double): String = "${value.toInt()}°C"

    private fun weatherDescription(code: Int): String = when (code) {
        0 -> "晴"
        1, 2 -> "少云"
        3 -> "阴"
        45, 48 -> "雾"
        51, 53, 55, 56, 57 -> "毛毛雨"
        61, 63, 65, 66, 67 -> "雨"
        71, 73, 75, 77 -> "雪"
        80, 81, 82 -> "阵雨"
        85, 86 -> "阵雪"
        95, 96, 99 -> "雷暴"
        else -> "天气代码 $code"
    }
}

object FunctionCalling {
    const val maxToolCalls = 3

    val systemInstruction = """
        必要时可调用受控工具；是否调用由你根据用户问题决定。调用时只能输出完整 JSON：
        {"tool":"工具名","arguments":{...}}
        不调用工具时直接正常回答。工具结果返回后必须基于结果回答，不能重复调用同一工具。
        工具：
        %s
    """.trimIndent()

    fun systemInstruction(toolInstruction: String): String =
        systemInstruction.format(toolInstruction)

    fun parseToolCall(modelText: String): ToolCall? {
        val rawText = modelText.substringAfterLast("</think>", modelText)
        val text = rawText
            .trim()
            .removePrefix("```json")
            .removePrefix("```")
            .removeSuffix("```")
            .trim()
        if (!text.startsWith("{") || !text.endsWith("}")) {
            return null
        }
        return runCatching {
            val json = JSONObject(text)
            ToolCall(
                name = json.getString("tool"),
                arguments = json.optJSONObject("arguments") ?: JSONObject()
            )
        }.getOrNull()
    }

    /**
     * 流式阶段只能看到半截内容。工具协议固定从 {"tool 开始，先缓冲这些内容，
     * 直到确认它是普通回答或完整工具调用，避免 JSON 闪进聊天气泡。
     */
    fun shouldDeferToolRendering(streamedText: String): Boolean {
        val text = streamedText
            .substringAfterLast("</think>", streamedText)
            .trimStart()
        val toolPrefix = "{\"tool"
        return text.isEmpty() || toolPrefix.startsWith(text) || text.startsWith(toolPrefix)
    }
}
