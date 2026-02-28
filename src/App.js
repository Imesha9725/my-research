import React, { useState } from 'react';
import Chat from './components/Chat';
import './App.css';

const NAV_SECTIONS = [
  { id: 'overview', label: 'Overview' },
  { id: 'problem', label: 'Research Problem' },
  { id: 'questions', label: 'Research Questions' },
  { id: 'objectives', label: 'Objectives' },
  { id: 'scope', label: 'Scope' },
  { id: 'related', label: 'Related Work' },
  { id: 'methodology', label: 'Methodology' },
  { id: 'evaluation', label: 'Evaluation' },
  { id: 'novelty', label: 'Novelty' },
  { id: 'timeline', label: 'Timeline' },
  { id: 'deliverables', label: 'Deliverables' },
  { id: 'references', label: 'References' },
];

const TIMELINE = [
  { months: 'Mar–Apr', task: 'Supervisor Selection, Title & Proposal Submission' },
  { months: 'Apr–May', task: 'Literature Review' },
  { months: 'May–Jun', task: 'Data Collection' },
  { months: 'Jun', task: 'Database Design' },
  { months: 'Jun–Jul', task: 'Draft Introduction + Literature Review Chapters' },
  { months: 'Jul–Oct', task: 'Chatbot Implementation' },
  { months: 'Sep', task: 'Interim Report Submission' },
  { months: 'Oct', task: 'Evaluation Plan Submission' },
  { months: 'Oct', task: 'Supervisor Demo Submission' },
  { months: 'Nov–Dec', task: 'Draft Thesis - Supervisor Version' },
  { months: 'Dec–Jan', task: 'Evaluation and Validation' },
  { months: 'Jan–Feb', task: 'Final Thesis & Final Defense' },
];

const COMPARISON_ROWS = [
  { name: 'Woebot', approach: 'CBT', emotion: 'No real-time', ai: 'Rule-based NLP', metrics: 'User engagement', limitation: 'Lacks adaptive responses, No deep learning' },
  { name: 'Wysa', approach: 'CBT, DBT, Mindfulness', emotion: 'No real-time', ai: 'ML-based sentiment', metrics: 'App ratings, user feedback', limitation: 'Does not recognize emotional intensity, No multimodal' },
  { name: 'Replika', approach: 'Conversational AI', emotion: 'No real-time', ai: 'Transformer-based', metrics: 'User retention', limitation: 'Lacks therapeutic grounding, No crisis intervention' },
  { name: 'Tess', approach: 'Psychological AI', emotion: 'No real-time', ai: 'Pre-trained NLP', metrics: 'Expert validation', limitation: 'No multimodal recognition, No personalized adaptation' },
  { name: 'Youper', approach: 'CBT, ACT', emotion: 'No real-time', ai: 'Deep learning', metrics: 'Clinical trials', limitation: 'Lacks emotional history tracking, Limited real-time adaptation' },
];

function App() {
  const [view, setView] = useState('chat');
  const [navOpen, setNavOpen] = useState(false);

  const scrollTo = (id) => {
    document.getElementById(id)?.scrollIntoView({ behavior: 'smooth' });
    setNavOpen(false);
  };

  if (view === 'chat') {
    return (
      <div className="app app--chat">
        <Chat />
      </div>
    );
  }

  return (
    <div className="app">
      <header className="header">
        <div className="header-inner">
          <span className="badge">UCSC-PG-MCS3204-F001 · MCS 3204</span>
          <button
            className="nav-toggle"
            onClick={() => setNavOpen(!navOpen)}
            aria-label="Toggle menu"
          >
            <span></span><span></span><span></span>
          </button>
          <nav className={`nav ${navOpen ? 'nav-open' : ''}`}>
            <button className="nav-link nav-tab" onClick={() => setView('chat')}>
              Chat
            </button>
            {NAV_SECTIONS.map((s) => (
              <button key={s.id} className="nav-link" onClick={() => scrollTo(s.id)}>
                {s.label}
              </button>
            ))}
          </nav>
        </div>
      </header>

      <main>
        <section id="hero" className="hero">
          <div className="hero-content">
            <h1 className="hero-title">
              Emotion-Aware Question Answering Chatbot for Mental Health Support
            </h1>
            <p className="hero-subtitle">
              Individual Project Proposal · Natural Language Processing & Machine Learning
            </p>
            <div className="hero-meta">
              <div className="meta-card">
                <span className="meta-label">Candidate</span>
                <strong>Kaludura Imesha Rishani De Silva</strong>
                <span>K.I.R. De Silva</span>
              </div>
              <div className="meta-card">
                <span className="meta-label">Registration</span>
                <span>2023/MCS/014 · Index 23440147</span>
              </div>
              <div className="meta-card">
                <span className="meta-label">Supervisor</span>
                <strong>Prof. G.D.S.P. Wimalaratne</strong>
                <span>Professor, UCSC</span>
              </div>
            </div>
          </div>
        </section>

        <section id="overview" className="section">
          <div className="container">
            <h2 className="section-title">Project Overview</h2>
            <div className="cards">
              <div className="card">
                <h3>Project Type</h3>
                <p>Individual Project in MCS Degree Program (MCS 3204) · 1st Attempt</p>
              </div>
              <div className="card">
                <h3>Contact</h3>
                <p>imesharishani9725@gmail.com · 0716811496 / 0718348272</p>
              </div>
            </div>
          </div>
        </section>

        <section id="problem" className="section section-alt">
          <div className="container">
            <h2 className="section-title">Research Problem</h2>
            <div className="prose">
              <p>
                Millions of individuals face mental health challenges such as depression, anxiety, stress, and other emotional difficulties, often in silence. Due to stigma, limited access to professionals, and overwhelming emotions, many find it difficult to seek help. The fear of judgment and lack of accessible resources prevent people from reaching out, especially when they cannot express their feelings adequately.
              </p>
              <p>
                Despite existing mental health chatbots (Woebot, Wysa, Replika), these systems fail to address <strong>real-time emotional awareness</strong>, <strong>personalized support</strong>, and <strong>dynamic response adaptation</strong>. Most rely on basic sentiment analysis without detecting emotions in real-time or adapting to emotional intensity. Traditional chatbots only process text, missing rich emotional cues from voice tone, and lack the ability to dynamically adjust conversation style to the user's emotional state and context.
              </p>
              <p className="highlight">
                This research aims to bridge these gaps by developing an emotion-aware chatbot that uses text and voice-based emotion recognition to detect emotions and adjust responses dynamically—ensuring more personalized, effective, and empathetic mental health support, potentially contributing to suicide prevention and psychological well-being.
              </p>
            </div>
          </div>
        </section>

        <section id="questions" className="section">
          <div className="container">
            <h2 className="section-title">Research Questions</h2>
            <ol className="numbered-list">
              <li>
                <strong>How can an NLP-based chatbot accurately detect emotions from both text and voice inputs?</strong>
                <ul>
                  <li>What techniques can be used to generate context-aware and empathetic responses for users experiencing mental health challenges?</li>
                  <li>How can a chatbot dynamically adapt its response style based on the user's emotional intensity and context?</li>
                </ul>
              </li>
            </ol>
            <p className="prose">
              The study explores NLP, speech emotion recognition (SER), and deep learning to understand and respond effectively. It involves implementing strategies to modulate responses by detected emotions and leveraging reinforcement learning to enhance personalization over time.
            </p>
          </div>
        </section>

        <section id="objectives" className="section section-alt">
          <div className="container">
            <h2 className="section-title">Research Objectives</h2>
            <p className="lead">
              To develop an NLP-based emotion-aware chatbot for mental health support that can detect emotions from both text and voice inputs and generate empathetic, adaptive responses using a Large Language Model (LLM).
            </p>
            <ul className="objectives-list">
              <li>To accurately develop an emotion detection model that analyzes both textual and voice inputs using NLP and speech emotion recognition techniques.</li>
              <li>To implement an LLM-based machine learning model that generates contextually appropriate and empathetic responses based on detected emotions.</li>
              <li>To integrate emotion-based response adaptation that dynamically adjusts the chatbot's conversational style based on the user's emotional state, and to implement response personalization using reinforcement learning.</li>
              <li>To evaluate the effectiveness of the chatbot through expert feedback.</li>
            </ul>
          </div>
        </section>

        <section id="scope" className="section">
          <div className="container">
            <h2 className="section-title">Scope of the Study</h2>
            <div className="scope-grid">
              <div className="card">
                <h3>Emotion Detection from Multiple Modalities</h3>
                <p><strong>Text:</strong> Transformer-based models (BERT, RoBERTa, GPT variants) for emotion classification.</p>
                <p><strong>Voice:</strong> Speech emotion recognition (SER) with acoustic features (pitch, tone, energy, MFCCs) via openSMILE or librosa; classification using CNNs, RNNs, or Wav2Vec-based architectures.</p>
              </div>
              <div className="card">
                <h3>Adaptive Response Mechanism</h3>
                <p>Dynamic response generation based on emotional intensity and context. LLMs (GPT/T5) fine-tuned for empathetic response generation; reinforcement learning integrated to adapt responses to individual user needs.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="related" className="section section-alt">
          <div className="container">
            <h2 className="section-title">Related Work & Research Gaps</h2>
            <p className="prose">
              Current mental health chatbots (Woebot, Wysa, Replika, Tess, Youper) provide general emotional support but lack real-time emotion detection and personalized, context-aware responses. Most focus on text-based communication with limited voice-based emotion recognition. NLP techniques (BERT, GPT, T5) have advanced, but their application in mental health chatbots remains underexplored.
            </p>
            <div className="table-wrap">
              <table className="comparison-table">
                <thead>
                  <tr>
                    <th>Chatbot</th>
                    <th>Therapeutic Approach</th>
                    <th>Emotion Detection</th>
                    <th>AI/NLP Techniques</th>
                    <th>Key Limitation</th>
                  </tr>
                </thead>
                <tbody>
                  {COMPARISON_ROWS.map((row, i) => (
                    <tr key={i}>
                      <td><strong>{row.name}</strong></td>
                      <td>{row.approach}</td>
                      <td>{row.emotion}</td>
                      <td>{row.ai}</td>
                      <td>{row.limitation}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="gaps">
              <h3>Identified Research Gaps</h3>
              <ul>
                <li><strong>Lack of real-time emotion detection</strong> using both text and voice inputs. This research proposes an emotional memory model that tracks user emotions over time.</li>
                <li><strong>Limited adaptive, context-aware responses</strong> based on emotional intensity. This research will develop an emotion-based response adaptation mechanism.</li>
              </ul>
            </div>
          </div>
        </section>

        <section id="methodology" className="section">
          <div className="container">
            <h2 className="section-title">Methodology</h2>
            <div className="method-steps">
              <div className="method-step">
                <span className="step-num">1</span>
                <h3>Data Collection</h3>
                <p>Gather emotion-labeled datasets: IEMOCAP, MELD, MOSEI for emotion recognition across speech and text.</p>
              </div>
              <div className="method-step">
                <span className="step-num">2</span>
                <h3>Preprocessing</h3>
                <p>Text tokenization (BERT/GPT-based); speech feature extraction (openSMILE, librosa); emotion classification into predefined classes.</p>
              </div>
              <div className="method-step">
                <span className="step-num">3</span>
                <h3>Model Development</h3>
                <p>Emotion recognition (BERT/RoBERTa for text; Wav2Vec/SpeechBERT for speech); response generation (GPT/T5/BERT-based); reinforcement learning for personalization.</p>
              </div>
              <div className="method-step">
                <span className="step-num">4</span>
                <h3>Chatbot Development</h3>
                <p>Integration of emotion recognition and response generation; personalization mechanisms based on prior interactions.</p>
              </div>
              <div className="method-step">
                <span className="step-num">5</span>
                <h3>Testing & Optimization</h3>
                <p>Expert evaluation by mental health professionals; standard metrics (ROUGE, BERTScore, user satisfaction); A/B testing against existing chatbots.</p>
              </div>
            </div>
          </div>
        </section>

        <section id="evaluation" className="section section-alt">
          <div className="container">
            <h2 className="section-title">Evaluation</h2>
            <div className="eval-grid">
              <div className="card">
                <h3>Quantitative</h3>
                <ul>
                  <li>Emotion detection accuracy (precision, recall, accuracy)</li>
                  <li>Response generation: ROUGE, BERTScore; RL-based optimization with expert feedback</li>
                </ul>
              </div>
              <div className="card">
                <h3>Qualitative</h3>
                <ul>
                  <li>Expert evaluation by mental health professionals</li>
                  <li>A/B testing vs. existing mental health chatbots</li>
                </ul>
              </div>
              <div className="card">
                <h3>Data</h3>
                <ul>
                  <li>Pre-existing datasets (IEMOCAP, MELD, MOSEI)</li>
                  <li>Anonymized real-world conversations from voluntary user testing</li>
                </ul>
              </div>
            </div>
          </div>
        </section>

        <section id="novelty" className="section">
          <div className="container">
            <h2 className="section-title">Novelty & Expected Contributions</h2>
            <ul className="contrib-list">
              <li><strong>Multimodal Emotion Recognition:</strong> Integration of text and voice-based emotion analysis; detection of high-risk states (e.g., suicidal ideation) from both modalities.</li>
              <li><strong>Contextually-Aware Psychological Support:</strong> Dynamic response adjustment based on emotional history and conversation context for more personalized, empathetic responses.</li>
            </ul>
            <div className="contrib-outcomes">
              <h3>Expected Contributions</h3>
              <ul>
                <li>Development of an emotion-aware chatbot for mental health support.</li>
                <li>New insights into emotion recognition and NLP applications in mental health.</li>
              </ul>
            </div>
          </div>
        </section>

        <section id="timeline" className="section section-alt">
          <div className="container">
            <h2 className="section-title">Project Plan & Timeline</h2>
            <div className="timeline">
              {TIMELINE.map((item, i) => (
                <div key={i} className="timeline-item">
                  <span className="timeline-marker"></span>
                  <div className="timeline-content">
                    <span className="timeline-months">{item.months}</span>
                    <span className="timeline-task">{item.task}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </section>

        <section id="deliverables" className="section">
          <div className="container">
            <h2 className="section-title">Deliverables</h2>
            <div className="card card-wide">
              <h3>NLP-based Emotion-Aware Chatbot</h3>
              <p>A fully functional chatbot that integrates multimodal emotion recognition (text and voice) and provides emotion-aware, personalized responses.</p>
              <ul>
                <li>Emotion detection based on text and voice inputs.</li>
                <li>Dynamic response adaptation based on emotional state.</li>
              </ul>
            </div>
          </div>
        </section>

        <section id="references" className="section section-alt">
          <div className="container">
            <h2 className="section-title">References</h2>
            <ol className="ref-list">
              <li>National Eating Disorders Association (2023). AI chatbot offline after complaints. CNN.</li>
              <li>YouPer Review | AI Mental Health App & CBT Chatbot features (2024). valueabletrends.com</li>
              <li>admin (2023). Why Generative AI Is Not Yet Ready for Mental Healthcare. Woebot Health.</li>
              <li>Akram, H. (2024). 10 Best Emotional AI Chatbot for 2024. JenAI Chat Blog.</li>
              <li>Bilquise, G., Ibrahim, S. & Shaalan, K. (2022). Emotionally Intelligent Chatbots: A Systematic Literature review. Human Behavior and Emerging Technologies.</li>
              <li>Devaram, S. (2020). Empathic Chatbot: Emotional Intelligence for Mental Health Well-being. arXiv.</li>
              <li>Startmotionmedia (2024). Replika AI: Is It Worth It? StartMotionMedia.</li>
              <li>Taiwo, O. & Al‐Bander, B. (2025). Emotion‐aware psychological first aid: Integrating BERT‐based emotional distress detection with PFA-GPT. Cognitive Computation and Systems.</li>
              <li>Chaudhry et al. (2024). User perceptions of an AI-driven conversational agent for mental health support. mHealth.</li>
              <li>White, N. (2024). AI chatbots for Mental Health: Opportunities and Limitations. eMHIC.</li>
            </ol>
            <div className="additional-info">
              <h3>Additional Information</h3>
              <p><strong>Potential Collaborations:</strong> Partnerships with mental health professionals to validate the chatbot's psychological support mechanisms and effectiveness.</p>
              <p><strong>Project Work Related:</strong> No. This is an independent academic project for the Master's degree.</p>
            </div>
          </div>
        </section>
      </main>

      <footer className="footer">
        <div className="container">
          <p>UCSC-PG-MCS3204-F001 · Proposal Submission Form 2025</p>
          <p>Kaludura Imesha Rishani De Silva · 2023/MCS/014</p>
        </div>
      </footer>
    </div>
  );
}

export default App;
