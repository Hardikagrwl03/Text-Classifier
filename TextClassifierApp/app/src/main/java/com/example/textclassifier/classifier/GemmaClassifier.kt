package com.example.textclassifier.classifier

import com.google.ai.edge.litertlm.*
class GemmaClassifier : TextClassifier {

    var engine: Engine
    var conversationConfig: ConversationConfig

    init{
        val engineConfig = EngineConfig(
            modelPath = "/sdcard/slm/gemma-4-E2B-it.litertlm", // Replace with your model path
            backend = Backend.CPU(),
             cacheDir = "/tmp/"
        )
        engine = Engine(engineConfig)
        engine.initialize()
        conversationConfig = ConversationConfig(
            systemInstruction = Contents.of("You are a Text Classifier and are supposed to classify text into [Health, Academics, Sports, Finance, Event] Categories.")
//            samplerConfig = SamplerConfig(topK = 10, topP = 0.95, temperature = 0.8),
        )
    }

    override fun classify(text: String): String {
        val conversation = engine.createConversation(conversationConfig)
        val output = conversation.sendMessage("My grandfather has Diabetes.")

        return output.toString()
    }
}