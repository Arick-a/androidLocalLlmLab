package com.arick.androidlocalllmlab

import android.os.Bundle
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
import androidx.compose.ui.Modifier
import com.arick.androidlocalllmlab.ui.theme.AndroidLocalLlmLabTheme

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
    Column(modifier = modifier) {
        Button(onClick = {
            localLlmRuntime.create()
        }, content = { Text(text = "create") })
        Button(onClick = {
            localLlmRuntime.close()
        }, content = { Text(text = "release") })
    }

}
