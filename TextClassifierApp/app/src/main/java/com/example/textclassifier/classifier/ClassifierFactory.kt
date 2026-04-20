package com.example.textclassifier.classifier

object ClassifierFactory {

    fun getClassifier(modelName: String): TextClassifier {
        return when (modelName) {
            "Gemma" -> GemmaClassifier()
            "BERT" -> BertClassifier()
            "TinyML" -> TinyMLClassifier()
            else -> throw IllegalArgumentException("Unknown model: $modelName")
        }
    }
}