# Viva Preparation - Emotion-Aware Mental Health Chatbot (MCS 3204)

## Likely Questions Your Supervisor May Ask

### 1. Research Design & Motivation
- **Why did you build an emotion-aware chatbot?**  
  Mental health support benefits from context-aware responses. Recognizing user emotion (sad, anxious, stressed, etc.) lets the system tailor empathetic replies instead of generic ones.

- **What is the novelty of your approach?**  
  Combining IEMOCAP for emotion detection (voice) and dataset-driven response selection with topic/emotion-based curated responses for mental health support. Dynamic responses vary by situation (exam, work, family, crisis, etc.).

- **What problem does your system solve?**  
  Provides 24/7 emotional support that adapts to the user's emotional state and situation, with crisis handling and referral to professional help when needed.

### 2. Emotion Detection
- **How do you detect user emotions?**  
  (1) **Voice:** SER model trained on IEMOCAP predicts emotion from audio.  
  (2) **Text:** Keyword-based detection (sad, anxious, stress, angry, lonely, happy, tired, fear) when voice is not used.

- **Why IEMOCAP?**  
  Widely used benchmark for emotion recognition, has labeled emotions (neutral, happy, angry, sad, excited, frustrated). We use it for training the SER model and for response selection in suitable scenarios.

- **What are IEMOCAP’s limitations?**  
  Scripted/acted speech; no topic labels (exam, work, family); Western context. We combine it with curated mental-health responses for topics IEMOCAP does not cover.

### 3. Response Generation
- **How do you select appropriate responses?**  
  Multi-stage pipeline: Crisis → Greeting → Topic (exam, work, family, etc.) → Emotion → IEMOCAP (only when strong match and suitable) → Default. Each stage uses curated empathetic responses or filtered IEMOCAP replies.

- **When do you use IEMOCAP for responses?**  
  Only when (a) user input strongly matches a dataset utterance (0.55+ content-word overlap) and (b) the response passes a suitability filter (excludes drama/bureaucracy scripts).

- **How do you handle irrelevant IEMOCAP lines?**  
  A blocklist filters out scripted phrases (e.g. “brandy”, “forms”, “bank account”). Only responses suitable for mental health support are used.

### 4. Evaluation & Limitations
- **How did you evaluate the system?**  
  [You can mention: emotion detection accuracy on IEMOCAP test set; qualitative user feedback; demonstration of scenarios with expected responses.]

- **What are the limitations?**  
  Keyword-based text emotion detection is limited; IEMOCAP is acted; no long-term memory; not a substitute for professional care.

- **How do you handle crisis situations?**  
  Detection of crisis-related keywords (suicide, self-harm) triggers immediate crisis responses and referral to helplines (e.g. Sumithrayo 1926).

### 5. Ethical & Practical
- **Is this a replacement for therapists?**  
  No. It is a supportive companion that provides emotional validation and encourages professional help when needed. The disclaimer states this clearly.

- **How do you ensure user safety?**  
  Crisis detection, helpline referral, and clear disclaimer that the system is not professional care.

---

## Demonstration Scenarios for Viva

Prepare to demo these in order. Type exactly as shown for reliable behavior.

| # | Scenario | Example User Input | Expected Behavior |
|---|----------|-------------------|-------------------|
| 1 | **Greeting** | "Hi, I'm Imesha" | Warm welcome, invites to share |
| 2 | **Crisis** | "I sometimes think about ending it all" | Crisis response, helpline (1926) |
| 3 | **Exam stress** | "I have exams next week and I'm so stressed" | Exam-specific empathetic response |
| 4 | **Work stress** | "My boss is making work unbearable" | Work stress response |
| 5 | **Family problems** | "I'm having family issues" | Family-support response |
| 6 | **Relationship** | "My boyfriend broke up with me" | Relationship support |
| 7 | **Loneliness** | "I feel so alone" | Loneliness response |
| 8 | **Anxiety** | "I can't stop worrying" | Anxiety/grounding response |
| 9 | **Sadness** | "I feel really down today" | Sadness support |
| 10 | **Anger** | "I'm so frustrated with everything" | Anger validation |
| 11 | **Tired/Burnout** | "I'm exhausted all the time" | Tired/burnout response |
| 12 | **Pet loss** | "My cat went missing" | Pet-related support |
| 13 | **Health/Sleep** | "I haven't been able to sleep" | Health/sleep response |
| 14 | **Money stress** | "I'm worried about my bills" | Financial stress response |
| 15 | **Vague distress** | "I don't know what's wrong, something feels off" | General support, invites to elaborate |
| 16 | **Self-doubt** | "I feel like I'm not good enough" | Self-worth validation |
| 17 | **Grief/Loss** | "I lost someone close to me" | Grief support |
| 18 | **Positive** | "I'm feeling better today" | Positive validation |
| 19 | **Fear/Future** | "I'm scared about what the future holds" | Fear/future anxiety support |
| 20 | **Guilt** | "I feel guilty about something I did" | Guilt validation |
| 21 | **Rejection/Social** | "Nobody likes me" / "I was rejected" | Rejection support |
| 22 | **Job loss** | "I lost my job" | Job loss support |
| 23 | **Bullying** | "People are bullying me" | Bullying support |
| 24 | **Conflict/Fight** | "I had a big fight with my friend" | Conflict support |
| 25 | **Disappointment** | "I'm so disappointed things didn't work out" | Disappointment support |
| 26 | **Overthinking** | "I can't stop overthinking" | Overthinking/grounding |
| 27 | **Irritable** | "Everything annoys me lately" | Irritability support |
| 28 | **Numb/Empty** | "I feel numb / empty inside" | Numbness support |
| 29 | **Missing someone** | "I really miss them" | Missing someone support |
| 30 | **Life change** | "I'm graduating / moved / everything changed" | Transition support |
| 31 | **Parenting stress** | "I'm overwhelmed with my kids" | Parenting support |
| 32 | **Academic failure** | "I failed my exam" | Failure support |

---

## IEMOCAP vs Added Scenarios

**IEMOCAP covers:** Bureaucracy/frustration (DMV), deployment/sacrifice, happiness, grief/death, relationship tension (scripted drama). Many IEMOCAP lines are filtered out for mental health support.

**Added for chatbot:** Exam, work, family, relationship, health, money, pet, grief, self-doubt, vague distress, guilt, future anxiety, rejection, job loss, bullying, conflict, disappointment, overthinking, irritable, numb/empty, missing someone, life change, parenting, academic failure.

---

## Response Pipeline Summary (for quick recall)

1. **Crisis** → Helpline, safety
2. **Greeting** → Welcome, invite to share
3. **Topic** → exam, work, family, relationship, health, money, pet, grief, self-doubt, vague
4. **Emotion** → sad, anxious, stress, angry, lonely, happy, tired, fear
5. **IEMOCAP** → Only when strong match + suitable response
6. **Default** → General empathetic support
