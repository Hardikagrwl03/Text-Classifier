package com.example.textclassifier.classifier

interface TextClassifier {
    fun classify(text: String): String
}