package com.arick.androidlocalllmlab.ui.theme

import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable

private val AppColorScheme = lightColorScheme(
    primary = AppColors.PrimaryAction,
    onPrimary = AppColors.Surface,
    background = AppColors.PageBackground,
    onBackground = AppColors.TextPrimary,
    surface = AppColors.Surface,
    onSurface = AppColors.TextPrimary,
    surfaceVariant = AppColors.SettingsBackground,
    onSurfaceVariant = AppColors.TextSecondary,
    error = AppColors.Error
)

@Composable
fun AndroidLocalLlmLabTheme(content: @Composable () -> Unit) {
    MaterialTheme(
        colorScheme = AppColorScheme,
        typography = Typography,
        content = content
    )
}
