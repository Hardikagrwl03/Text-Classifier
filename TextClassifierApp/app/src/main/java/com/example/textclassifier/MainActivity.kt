package com.example.textclassifier

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.widget.*
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.content.ContextCompat
import com.example.textclassifier.classifier.ClassifierFactory

class MainActivity : ComponentActivity() {

    private lateinit var inputText: EditText
    private lateinit var modelSpinner: Spinner
    private lateinit var outputText: TextView
    private lateinit var classifyButton: Button

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestPermission()) { isGranted ->
            if (isGranted) {
                Toast.makeText(this, "Permission granted", Toast.LENGTH_SHORT).show()
            } else {
                Toast.makeText(this, "Permission denied", Toast.LENGTH_SHORT).show()
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Set XML layout
        setContentView(R.layout.activity_main)

        checkAndRequestPermission()

        inputText = findViewById(R.id.inputText)
        modelSpinner = findViewById(R.id.modelSpinner)
        outputText = findViewById(R.id.outputText)
        classifyButton = findViewById(R.id.classifyButton)

        val models = listOf("Gemma", "BERT", "TinyML")

        val adapter = ArrayAdapter(this, android.R.layout.simple_spinner_item, models)
        adapter.setDropDownViewResource(android.R.layout.simple_spinner_dropdown_item)
        modelSpinner.adapter = adapter

        classifyButton.setOnClickListener {
            val text = inputText.text.toString()
            val selectedModel = modelSpinner.selectedItem.toString()

            if (text.isEmpty()) {
                Toast.makeText(this, "Enter some text", Toast.LENGTH_SHORT).show()
                return@setOnClickListener
            }

            val category = classifyText(text, selectedModel)
            outputText.text = "Category: $category"
        }
    }

    private fun classifyText(text: String, model: String): String {
        val classifier = ClassifierFactory.getClassifier(model)
        val result = classifier.classify(text)
        return result
    }

    private fun checkAndRequestPermission() {
        val permission = when {
            Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU -> {
                Manifest.permission.READ_MEDIA_IMAGES
            }
            else -> {
                Manifest.permission.READ_EXTERNAL_STORAGE
            }
        }

        when {
            ContextCompat.checkSelfPermission(this, permission)
                    == PackageManager.PERMISSION_GRANTED -> {
                // Already granted
            }

            else -> {
                requestPermissionLauncher.launch(permission)
            }
        }
    }
}