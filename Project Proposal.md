![[스크린샷 2025-03-03 오후 5.50.07 1.png]]
## **1. Introduction**

### **1.1. Overview**

With the increasing need for AI-driven mental health support, this project aims to develop an **empathetic Large Language Model (LLM)** tailored for **art therapy**. While there are several existing **empathetic LLMs**, they primarily focus on **general emotional intelligence** and **mental health conversations**. However, **no AI system has been specifically developed to facilitate art therapy discussions**—a unique and essential aspect of emotional expression. This project will bridge that gap by combining conversational AI with an understanding of **art therapy and emotional intelligence**.

Additionally, traditional **art therapy requires time and financial resources**, as individuals must visit professional therapists. This project seeks to **democratize access to art therapy**, making it available to anyone with a **tablet or smartphone**, allowing users to engage in therapeutic conversations from the comfort of their homes.

This AI model aspires to be a **modern revival of Bob Ross**, helping people unwind and find peace after a stressful day, much like the beloved painter did through his instructional art sessions.

([https://www.youtube.com/watch?v=_CpdwTcvBU8](https://www.youtube.com/watch?v=_CpdwTcvBU8)) 
([https://www.youtube.com/watch?v=z7r0DIjRGd4](https://www.youtube.com/watch?v=z7r0DIjRGd4))

### **1.2. Motivation & Importance**

- **Art therapy** is an effective form of self-expression and emotional healing, yet there are **no decent AI systems** designed to facilitate these conversations.
    
- Traditional **art therapy requires time and financial resources**, as individuals must visit professional therapists. This limits accessibility for many people.

- This project will create an **emotionally aware AI** that enhances user experience and emotional well-being by integrating **art therapy techniques into conversational AI**.
    
- This project aims to **make art therapy accessible from home**—anyone with a **tablet or smartphone** can engage in therapeutic conversations without requiring in-person sessions.
---
## **2. Objectives**

1. **Develop a fine-tuned LLM** for empathetic conversations related to art and emotions.
2. **Incorporate psychological principles** from mental health datasets into the model.
3. **Optimize for efficient deployment** on **Intel hardwares (IPEX-LLM)**.
4. **Evaluate AI-generated responses** based on coherence, empathy, and relevance.
---
## **3. Methodology**

### **3.1. Data Preparation**

We curated multiple datasets to ensure the model understands both **empathy and art therapy dialogues**:

| **Dataset**                                   | **Description**                               | **Use Case**                                         | Detailed description                                                  |
| --------------------------------------------- | --------------------------------------------- | ---------------------------------------------------- | --------------------------------------------------------------------- |
| <b style="color:red;">EmpatheticDialogues</b> | 24K dialogues with empathy-based responses    | Helps model learn **empathetic responses**           | https://huggingface.co/datasets/Estwld/empathetic_dialogues_llm       |
| <b style="color:red;">CounselChat Datase</b>  | Q&A from professional therapists              | Provides structured therapy-based advice             | https://huggingface.co/datasets/loaiabdalslam/counselchat/viewer      |
| **Mental Health Reddit Dataset**              | Conversations from mental health forums       | Trains model on **real mental health concerns**      | https://www.kaggle.com/datasets/neelghoshal/reddit-mental-health-data |
| <b style="color:red;">DailyDialog Datase</b>  | General human conversations with emotion tags | Enhances **emotion recognition** in conversations    | https://huggingface.co/datasets/li2017dailydialog/daily_dialog        |
| **Artemis Dataset**                           | Text descriptions of paintings                | Fine-tune the model specifically for **art therapy** | https://github.com/optas/artemis                                      |

To ensure consistency, we structured the data into **JSONL format** with clear input-response structures. Now, I have preprocessed three datasets in red (EmpatheticDialogues, CounselChat, and DailyDialog). 

1. <b style="color:red;">EmpatheticDialogues</b>
```jsonl
# 1.empathetic_dialogs_finetune_data.jsonl

# Turn 1:
{"instruction": "Respond empathetically to the following conversation.", 

"input": "Overall Emotion: sentimental\nUser: i remember going to see the fireworks with my best friend. it was the first time we ever spent time alone together. although there was a lot of people, we felt like the only people in the world.",

"output": "was this a friend you were in love with, or just a best friend?"}

# Turn 2:
{"instruction": "Respond empathetically to the following conversation.", 

"input": "Overall Emotion: sentimental\nUser: i remember going to see the fireworks with my best friend. it was the first time we ever spent time alone together. although there was a lot of people, we felt like the only people in the world.\nAssistant: was this a friend you were in love with, or just a best friend?\nUser: this was a best friend. i miss her.",

"output": "where has she gone?"}

# Turn 3:
{"instruction": "Respond empathetically to the following conversation.", 

"input": "Overall Emotion: sentimental\nUser: i remember going to see the fireworks with my best friend. it was the first time we ever spent time alone together. although there was a lot of people, we felt like the only people in the world.\nAssistant: was this a friend you were in love with, or just a best friend?\nUser: this was a best friend. i miss her.\nAssistant: where has she gone?\nUser: we no longer talk.",

"output": "oh was this something that happened because of an argument?"}
```

2. <b style="color:red;">CounselChat Datase</b>:
```jsonl
# 2.counsel_chats_instruct_finetune_data.jsonl

# Instruct (Brief description of situation of user)
{"instruction": "Do I have too many issues for counseling?",

# Input with emotion label
"input": "[Emotion: depression] Do I have too many issues for counseling? I have so many issues to address. I have a history of sexual abuse, I\u2019m a breast cancer survivor and I am a lifetime insomniac. I have a long history of depression and I\u2019m beginning to have anxiety. I have low self esteem but I\u2019ve been happily married for almost 35 years. I\u2019ve never had counseling about any of this. Do I have too many issues to address in counseling?",

# Counseling answer from profession
"output": "It is very common for people to have multiple issues that they want to (and need to) address in counseling. I have had clients ask that same question and through more exploration, there is often an underlying fear that they \"can't be helped\" or that they will \"be too much for their therapist.\" I don't know if any of this rings true for you. But, most people have more than one problem in their lives and more often than not, people have numerous significant stressors in their lives. Let's face it, life can be complicated! Therapists are completely ready and equipped to handle all of the issues small or large that a client presents in session. Most therapists over the first couple of sessions will help you prioritize the issues you are facing so that you start addressing the issues that are causing you the most distress. You can never have too many issues to address in counseling. All of the issues you mention above can be successfully worked through in counseling."}
```

3. <b style="color:red;">DailyDialog Datase</b>:
```jsonl
# 3.standardized_daily_dialog_finetune_data.jsonl
# This model has error for dividing the role (user & assistant)

# Trun 1:
{"instruction": "Respond empathetically to the following conversation.",

"input": "Overall Emotion: surprise\nUser [neutral]: Can you do push-ups?", 

"output": "Of course I can. It's a piece of cake! Believe it or not, I can do 30 push-ups a minute."}

# Turn 2:
{"instruction": "Respond empathetically to the following conversation.", 

"input": "Overall Emotion: surprise\nUser [neutral]: Can you do push-ups?\nAssistant [neutral]: Of course I can. It's a piece of cake! Believe it or not, I can do 30 push-ups a minute.\nUser [surprise]: Really? I think that's impossible!",

"output": "You mean 30 push-ups?"}

# Turn 3:
{"instruction": "Respond empathetically to the following conversation.", 

"input": "Overall Emotion: surprise\nUser [neutral]: Can you do push-ups?\nAssistant [neutral]: Of course I can. It's a piece of cake! Believe it or not, I can do 30 push-ups a minute.\nUser [surprise]: Really? I think that's impossible!\nAssistant [neutral]: You mean 30 push-ups?\nUser [neutral]: Yeah!",

"output": "It's easy. If you do exercise everyday, you can make it, too."}
```

However, we still need more datasets which can help our model  to analyze user's state of mind and emotions. Also, we need dataset to train our model about artistic features and knowledges so it can actively help user's drawing (ex. recommend color or brush according to the dialogue between user and system). 

**EX. Artemis Dataset:**

![[스크린샷 2025-03-03 오후 5.36.22.png]]

---
### **3.2 Fine-Tuning Process**

#### **3.2.1. Model Selection**

| **Model**                    | **Parameters** | **Why Choose This?**                                            |
| ---------------------------- | -------------- | --------------------------------------------------------------- |
| **Mistral 7B Instruct V0.1** | 7B             | Efficient for **empathetic & Interactive LLM** (MLX, llama.cpp) |
| **Phi-2**                    | 2.7B           | Small, fast, and great for fine-tuning                          |
| **LLaMA 3** (Future)         | TBD            | Future-proof, optimized for research                            |
Now, I have chose Mistral 7B Instruct V0.1 model.
#### **3.2.2. Training Approach**

Each dataset is trained in **phases** to prevent interference between conversation types:

1. **Fine-Tune on Dataset 1** (<b style="color:red;">EmpatheticDialogues</b>) → Empathy & Mental Health Conversations
    
2. **Fine-Tune on Dataset 2** (<b style="color:red;">CounselChat Datase</b>) → Structured Q&A for Therapy
    
3. **Fine-Tune on Dataset 3** (<b style="color:red;">DailyDialog Datase</b>)→ Casual Multi-Turn Dialogues
    

For efficient and powerful training, we need to use these techniques:

1. **IPEX-LLM (Intel Extension for PyTorch - Large Language Models)**: optimization framework designed by **Intel** to accelerate inference and training of large language models (LLMs) on **Intel hardware** (such as Xeon CPUs and upcoming AI accelerators).

2. **Quantization**: reduces the precision of model weights (e.g., from 16-bit to 8-bit or 4-bit) to **save GPU/CPU memory** and **speed up inference**. This is particularly beneficial when fine-tuning **large models like LLaMA, Mistral, or GPT-based models** on limited hardware.

3. **LoRA (Low-Rank Adaptation)**: fine-tuning technique designed to make training large language models (LLMs) **faster, more memory-efficient, and scalable**. Instead of updating all model weights, LoRA introduces **small trainable matrices** (low-rank adapters) that modify only a subset of parameters during fine-tuning.

If we find more helpful techniques, we can apply them to our model.
#### **3.2.3. Deployment on Intel IPEX-LLM**

If we can start with Intel GPU and CPU, it would be best but if not:
	Once fine-tuning is completed, the model is converted for **Intel IPEX-LLM optimization** to ensure high efficiency in deployment.

---
### 3.3. Hybrid Method for Model release

#### **💡 Why Hugging Face + ONNX/OpenVINO is the Best Choice for ArtiTech**

ArtiTech is developing an **LLM-based emotion analysis and art therapic AI**. To ensure optimal **model deployment and performance**, a combination of **Hugging Face and ONNX/OpenVINO** is the most effective approach. Below is a detailed explanation of why this hybrid method is ideal.

---

#### **3.3.1. Why Use Hugging Face (`save_pretrained()`)**

##### ✅ **1) Easy LLM Deployment & Model Updates**

- ArtiTech needs an **LLM for emotion analysis and art style recommendation**.
- Hugging Face provides a **centralized platform for managing, sharing, and updating models**.
- Fine-tuned models can be updated regularly, allowing for continuous improvements.

##### ✅ **2) Seamless Integration with Data Pipelines & APIs**

- The `transformers` library from Hugging Face allows **easy model loading and deployment**.
- It enables smooth integration with **web apps, chatbots, and API services**.
- For example, ArtiTech's **emotion analysis feature can be deployed as an API** using FastAPI or Flask.

##### ✅ **3) GPU Support & Cloud Scalability**

- Hugging Face supports **accelerators like `bitsandbytes` and `accelerate`**, enabling efficient model inference even on low-resource environments.
- The platform is **cloud-friendly**, making it easy to scale inference workloads on **AWS, GCP, or Azure**.

---

#### **3.3.2. Why Use ONNX + OpenVINO**

##### ✅ **1) Performance Optimization (Speed & Lightweight Models)**

- ONNX optimizes PyTorch/TensorFlow models for **faster inference speeds**.
- OpenVINO further optimizes inference for **Intel CPUs and GPUs**, ensuring **maximum efficiency**.
- Since ArtiTech needs **real-time emotion analysis and art style recommendations**, **low-latency inference is crucial**.

##### ✅ **2) Intel IPEX Integration (Hardware Optimization)**

- ArtiTech plans to integrate **Intel IPEX LLM models**.
- OpenVINO is Intel’s dedicated inference engine, making it the **best choice for Intel-based AI acceleration**.
- **Supports both Apple Silicon (MPS) and Intel IPEX**, allowing for development and testing on Mac.

##### ✅ **3) Reduced Deployment Costs**

- Cloud-based GPU inference is **expensive**, but OpenVINO allows **high-performance inference on CPUs**.
- This approach enables **cost-effective AI deployment on local servers and edge devices**.
- Since art style recommendations require real-time inference, **a highly optimized model is essential**.

---

#### **3.3.3 ArtiTech’s Fine-tuning & Deployment Workflow**

##### **📌 1) Fine-tuning Phase (Offline/Online)**

1. Download a **pretrained model from Hugging Face**
2. Fine-tune the model using PyTorch
3. Save the model using `save_pretrained()`

##### **📌 2) Model Optimization for Deployment (ONNX/OpenVINO Conversion)**

1. Convert the fine-tuned model to **ONNX format**
2. Optimize the ONNX model using **OpenVINO (Intel IPEX acceleration)**
3. Deploy the optimized model to **local or cloud-based servers**

##### **📌 3) Real-time AI Service Operation**

1. **Emotion Analysis API** → Uses Hugging Face for quick deployment
2. **Art Style Recommendation AI** → Uses OpenVINO for optimized inference
3. **Regular fine-tuning updates based on user feedback**

---

#### 3.3.4. **💡 Conclusion: Hugging Face + ONNX/OpenVINO is the Optimal Approach**

##### **🎯 Why Hugging Face?**

- **Efficient LLM management and model updates**
- **Easy integration with APIs and data pipelines**
- **GPU support and cloud scalability**

##### **🎯 Why ONNX + OpenVINO?**

- **Performance optimization and lightweight models**
- **Intel IPEX integration for hardware acceleration**
- **Lower deployment costs with CPU inference**

##### **🎯 Final Strategy**

- **Use Hugging Face for LLM model management and API deployment**
- **Use ONNX/OpenVINO for optimized inference on Intel hardware and local/edge AI deployment**
- **Leverage both methods for maximum performance and scalability!**

### **3.4. Evaluation Metrics**

| **Metric**              | **Evaluation Method**                           |
| ----------------------- | ----------------------------------------------- |
| **Response Coherence**  | Manual Review of Conversations                  |
| **Emotion Recognition** | Compare Model vs. Human Labeled Data            |
| **Inference Speed**     | Benchmark other hardwares vs. Intel Performance |


---
## **4. Expected Outcomes**

- A **fine-tuned AI model** capable of engaging in **empathetic art therapy conversations**.
    
- **Deployment-ready LLM** optimized for **Intel (IPEX-LLM)**.
    
- An **evaluation framework** for measuring emotional intelligence in AI-generated responses.
    
---
## **5. Conclusion & Next Steps**

This project introduces an innovative **empathetic conversational AI** that bridges the gap between **mental health and art therapy**. Moreover, to ensure **seamless interaction and a more natural user experience**, the model should be **speech-based**, allowing users to communicate through voice rather than just text. This will enhance accessibility and improve engagement, especially for individuals who find verbal expression more therapeutic than typing.

To enable **speech-based interaction** with your model, you need a **pretrained speech-to-text (STT) and text-to-speech (TTS) model**. Here are the best options:

### **1. Speech-to-Text (STT) Models** (Convert user speech into text for LLM input)

✅ **Whisper (by OpenAI)** – Best in class for multilingual transcription  
✅ **DeepSpeech (Mozilla)** – Lightweight, open-source STT  
✅ **Wav2Vec2 (Facebook AI)** – Great for noisy environments

**Recommended for your project:**  
📌 **Whisper** because it provides high-quality transcriptions and supports many languages.

### **2. Text-to-Speech (TTS) Models** (Convert LLM-generated responses into speech)

✅ **VITS (Vocoder + TTS from NVIDIA)** – High-quality voice synthesis  
✅ **Tortoise-TTS** – Realistic, expressive speech synthesis  
✅ **Coqui TTS** – Open-source with many pre-trained models

**Recommended for your project:**  
📌 **VITS or Coqui-TTS** because they support fine-tuning, allowing you to create a unique **calm, Bob Ross-like voice** for the AI.

---

### **How to Integrate?**

- **Step 1:** Use **Whisper** to convert spoken words into text.
- **Step 2:** Send the transcribed text to your **fine-tuned LLM**.
- **Step 3:** Convert the AI-generated response into speech using **VITS or Coqui-TTS**.
- **Step 4:** Play the generated speech to the user.
---