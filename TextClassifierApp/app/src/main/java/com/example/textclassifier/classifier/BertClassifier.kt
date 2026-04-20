package com.example.textclassifier.classifier

class BertClassifier : TextClassifier {
    override fun classify(text: String): String {
        return "BERT → Academics"
    }
}