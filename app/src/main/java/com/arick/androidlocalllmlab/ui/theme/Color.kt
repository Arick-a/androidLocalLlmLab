package com.arick.androidlocalllmlab.ui.theme

import androidx.compose.ui.graphics.Color

/**
 * 应用固定视觉色板。
 *
 * 聊天页采用浅色方案，不跟随系统壁纸或深色模式变化；页面、卡片和弹窗可以保持一致。
 */
object AppColors {
    val PageBackground = Color.White
    val SettingsBackground = Color(0xFFF7F7F7)
    val Surface = Color.White

    val TextPrimary = Color.Black
    val TextSecondary = Color(0xFF555555)
    val TextTertiary = Color(0xFF777777)
    val TextHint = Color(0xFF999999)
    val TextDisabled = Color(0xFFAAAAAA)
    val Chevron = Color(0xFFBBBBBB)

    val UserBubble = Color(0xFFEAF3FF)
    val PrimaryAction = Color(0xFF0B57D0)
    val StopAction = Color(0xFFE85D5D)
    val DisabledAction = Color(0xFFE5E5E5)
    val Error = Color(0xFFD32F2F)
}
