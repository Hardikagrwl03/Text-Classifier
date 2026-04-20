package com.example.textclassifier.classifier

class TinyMLClassifier : TextClassifier {
    override fun classify(text: String): String {
        return "TinyML → Event"
    }
}