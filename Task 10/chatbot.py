import nltk
import random
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download required NLTK data
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# ─────────────────────────────────────────────
# Knowledge Base: intents with patterns + responses
# ─────────────────────────────────────────────
INTENTS = [
    {
        "tag": "greeting",
        "patterns": [
            "hello", "hi", "hey", "good morning", "good evening",
            "how are you", "whats up", "sup", "howdy"
        ],
        "responses": [
            "Hello! I'm Serenity, your mental wellness companion. How are you feeling today?",
            "Hi there! I'm here to listen and support you. What's on your mind?",
            "Hey! It's great to see you. How are you doing today?"
        ]
    },
    {
        "tag": "anxiety",
        "patterns": [
            "i feel anxious", "i have anxiety", "i am nervous", "panic attack",
            "i am stressed out", "i feel panicked", "my heart is racing",
            "i am worried", "i cant stop worrying", "overthinking",
            "feeling overwhelmed", "i am overwhelmed", "so much pressure"
        ],
        "responses": [
            "I hear you — anxiety can feel really overwhelming. Let's try a quick breathing exercise:\n\n🌬️ **4-7-8 Breathing:**\n• Inhale for 4 seconds\n• Hold for 7 seconds\n• Exhale slowly for 8 seconds\n\nRepeat 3 times. This activates your nervous system's calm response. How does that feel?",
            "Anxiety is tough, but you're not alone. Try grounding yourself with the **5-4-3-2-1 technique:**\n\n👁️ 5 things you can see\n✋ 4 things you can touch\n👂 3 things you can hear\n👃 2 things you can smell\n👅 1 thing you can taste\n\nThis brings your focus back to the present moment.",
            "I understand. When anxiety peaks, try **box breathing:**\n\n• Breathe in for 4 counts\n• Hold for 4 counts\n• Breathe out for 4 counts\n• Hold for 4 counts\n\nYou're safe. This feeling will pass. Would you like to talk about what's causing the worry?"
        ]
    },
    {
        "tag": "depression",
        "patterns": [
            "i feel depressed", "i am sad", "i feel empty", "i feel hopeless",
            "nothing makes me happy", "i dont want to do anything", "i feel worthless",
            "life feels pointless", "i feel low", "i have been crying", "i feel numb",
            "feeling blue", "everything is dark", "i feel lost"
        ],
        "responses": [
            "I'm really sorry you're feeling this way. Depression can make everything feel heavy and distant. You showed courage by reaching out.\n\n💙 A small step for today: try going outside for just 5 minutes of sunlight, or drink a glass of water. Small acts of self-care matter.\n\nWould you like to talk about what's been going on?",
            "Those feelings are valid, and I'm here for you. Depression lies — it tells you things won't get better, but that's not true.\n\n✨ One gentle exercise: write down **3 small things** you did today, even tiny ones like 'I got out of bed.' That counts.\n\nYou are not alone in this. Is there someone in your life you trust to talk to?",
            "I hear you, and I'm glad you're talking about it. Feeling low is exhausting. Please know that what you're feeling is a health condition — not a character flaw.\n\n🌱 Try the **behavioral activation** technique: do one small enjoyable activity today, even if you don't feel like it. It could be listening to a song you like.\n\nWould you like some resources for professional support?"
        ]
    },
    {
        "tag": "stress",
        "patterns": [
            "i am stressed", "too much work", "i cant handle this", "burnout",
            "i am exhausted", "work stress", "study stress", "exam stress",
            "deadlines", "so much to do", "no time", "pressure at work",
            "feeling burnt out", "i am tired of everything"
        ],
        "responses": [
            "Stress is your body's signal that it needs a break. Let's address it:\n\n📋 **Priority Reset:**\n1. Write down everything stressing you\n2. Circle only what you can control\n3. Pick ONE thing to tackle first\n\nBreaking it down makes it manageable. What's the biggest stressor right now?",
            "Burnout is real and it matters. Your wellbeing comes before productivity.\n\n🧘 **Progressive Muscle Relaxation:**\nTense each muscle group for 5 seconds, then release — start from your toes up to your face. This releases physical tension stored from stress.\n\nAlso: when did you last take a proper break?",
            "You're carrying a lot. Remember — you don't have to do everything at once.\n\n⏱️ Try the **Pomodoro method:** work for 25 minutes, then take a 5-minute break. It prevents burnout and boosts focus.\n\nAlso, hydration and sleep are non-negotiable for stress management. Are you sleeping enough?"
        ]
    },
    {
        "tag": "sleep",
        "patterns": [
            "i cant sleep", "insomnia", "trouble sleeping", "i wake up at night",
            "bad sleep", "i sleep too much", "nightmares", "i am not sleeping well",
            "sleep problems", "i stay up all night", "sleep anxiety"
        ],
        "responses": [
            "Poor sleep affects everything — mood, focus, and health. Here are some tips:\n\n🌙 **Sleep Hygiene Checklist:**\n• No screens 30 min before bed\n• Keep your room cool and dark\n• Same sleep/wake time every day\n• Avoid caffeine after 2 PM\n• Try chamomile tea before bed\n\nWould you like a bedtime relaxation routine?",
            "Sleep struggles are very common, especially when you're stressed. Try **4-7-8 breathing** as you lie in bed — inhale 4s, hold 7s, exhale 8s.\n\nAlso try **body scan meditation:** mentally scan from your head to toes, consciously relaxing each part.\n\nHow long has this been going on?"
        ]
    },
    {
        "tag": "loneliness",
        "patterns": [
            "i feel lonely", "i have no friends", "nobody cares", "i feel isolated",
            "no one understands me", "i feel alone", "i am lonely",
            "i miss people", "i feel disconnected", "no one to talk to"
        ],
        "responses": [
            "Feeling lonely is one of the most painful human experiences, and I want you to know — reaching out here took courage. You matter.\n\n💬 **Connection tips:**\n• Join an online community around a hobby\n• Volunteer — giving time creates bonds\n• Reconnect with one old friend via a message\n\nSometimes loneliness is a sign to invest in new connections. What are your interests?",
            "You're not alone — I'm here. Loneliness doesn't reflect your worth. It's a circumstance, not a verdict.\n\n🤝 Even small interactions count — a smile at a cashier, a comment in an online forum. Connection builds gradually.\n\nIs there anything specific making you feel disconnected right now?"
        ]
    },
    {
        "tag": "self_care",
        "patterns": [
            "how to feel better", "tips to improve mood", "self care",
            "how to be happy", "wellness tips", "improve mental health",
            "how to take care of myself", "positive habits", "daily routine"
        ],
        "responses": [
            "Here are evidence-based self-care habits:\n\n✅ **Daily Wellness Routine:**\n• 🌅 Morning: 5 min journaling + glass of water\n• 🏃 Move your body for 20 min (walk, dance, stretch)\n• 📵 Digital detox: 1 hour screen-free daily\n• 🙏 Gratitude: write 3 things you're thankful for\n• 😴 Sleep 7-8 hours consistently\n\nWhich one do you want to start with?",
            "Mental wellness is built daily through small choices:\n\n🧠 **Science-backed habits:**\n• Exercise releases endorphins — even a 15 min walk helps\n• Journaling processes emotions and reduces anxiety\n• Social connection is as vital as nutrition\n• Sunlight regulates mood-related hormones\n• Meditation for 10 min/day reshapes the brain over time\n\nWould you like a personalized routine?"
        ]
    },
    {
        "tag": "motivation",
        "patterns": [
            "i have no motivation", "i cant focus", "i am procrastinating",
            "i dont feel like doing anything", "i give up", "what is the point",
            "i feel stuck", "i cant move forward", "lack of energy"
        ],
        "responses": [
            "Loss of motivation often means you need rest or a new direction — not that you're failing.\n\n🔥 **Motivation reset:**\n1. Break your goal into the smallest possible step\n2. Commit to just 2 minutes of starting\n3. Momentum follows action — not the other way around\n\nWhat's one thing you've been putting off? Let's break it down together.",
            "Feeling stuck is part of growth. Even the most accomplished people face this.\n\n💡 Try **implementation intention:** instead of 'I'll study today,' say 'I'll study at 6 PM at my desk for 30 minutes.'\n\nSpecificity turns intention into action. What goal do you want to work toward?"
        ]
    },
    {
        "tag": "crisis",
        "patterns": [
            "i want to die", "i want to hurt myself", "suicidal", "end my life",
            "i cant go on", "i want to kill myself", "self harm", "no reason to live",
            "i don't want to exist", "i am going to hurt myself"
        ],
        "responses": [
            "🆘 I'm very concerned about you right now, and I want you to know — **your life has value.**\n\nPlease reach out to a crisis helpline immediately:\n\n📞 **Pakistan:** Umang helpline: 0317-4288665\n📞 **International:** Crisis Text Line — text HOME to 741741\n📞 **Befrienders Worldwide:** befrienders.org\n\nYou don't have to face this alone. Please talk to someone — a trusted person, doctor, or counselor. Are you safe right now?"
        ]
    },
    {
        "tag": "gratitude",
        "patterns": [
            "i feel good", "i am happy", "things are going well", "i feel great",
            "i am grateful", "feeling positive", "good day", "i feel better",
            "i am doing well", "life is good"
        ],
        "responses": [
            "That's wonderful to hear! 🌟 Holding onto good moments is itself a wellness practice.\n\nWould you like to explore ways to build on this positive energy — like a gratitude journal or a mood tracker?",
            "I love that! 😊 Happy moments are worth savoring. Research shows that mentally 'photographing' good moments helps your brain store them more vividly.\n\nWhat made today good for you?"
        ]
    },
    {
        "tag": "goodbye",
        "patterns": [
            "bye", "goodbye", "see you", "take care", "i have to go",
            "thanks", "thank you", "that helped", "i feel better now"
        ],
        "responses": [
            "Take care of yourself 💙 Remember — one day at a time. I'm always here if you need to talk.",
            "Goodbye! You did something good for yourself by reaching out today. Keep going. 🌱",
            "Thank you for sharing with me. Be kind to yourself — you deserve it. 💚"
        ]
    }
]

# ─────────────────────────────────────────────
# NLP Engine
# ─────────────────────────────────────────────
class MentalHealthBot:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            # Fallback minimal stopwords if NLTK data not downloaded
            self.stop_words = {
                'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves',
                'you', 'your', 'yours', 'he', 'him', 'his', 'she', 'her',
                'it', 'its', 'they', 'them', 'their', 'what', 'which', 'who',
                'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
                'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do',
                'does', 'did', 'will', 'would', 'could', 'should', 'may',
                'might', 'must', 'a', 'an', 'the', 'and', 'but', 'or', 'so',
                'for', 'in', 'on', 'at', 'to', 'of', 'with', 'about', 'from'
            }
        # Keep important negative words
        self.stop_words -= {'no', 'not', 'nor', 'neither', "don't", "can't", "won't"}

        self.intents = INTENTS
        self.vectorizer = TfidfVectorizer(tokenizer=self._preprocess, lowercase=True)

        # Build corpus: map each pattern to its intent tag
        self.corpus = []
        self.tags = []
        for intent in self.intents:
            for pattern in intent['patterns']:
                self.corpus.append(pattern)
                self.tags.append(intent['tag'])

        self.tfidf_matrix = self.vectorizer.fit_transform(self.corpus)
        self.conversation_history = []

    def _preprocess(self, text):
        """Tokenize, lemmatize, remove stopwords."""
        try:
            tokens = word_tokenize(text.lower())
        except LookupError:
            tokens = text.lower().split()
        try:
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens
                      if t.isalpha() and t not in self.stop_words]
        except LookupError:
            tokens = [t for t in tokens if t.isalpha() and t not in self.stop_words]
        return tokens if tokens else [text.lower()]

    def _get_intent(self, user_input):
        """Use TF-IDF + cosine similarity to find best matching intent."""
        user_vec = self.vectorizer.transform([user_input])
        similarities = cosine_similarity(user_vec, self.tfidf_matrix).flatten()
        best_idx = np.argmax(similarities)
        best_score = similarities[best_idx]

        if best_score < 0.15:
            return None, best_score

        return self.tags[best_idx], best_score

    def get_response(self, user_input):
        """Generate a response based on matched intent."""
        # Always prioritize crisis detection
        crisis_keywords = [
            'kill myself', 'want to die', 'end my life', 'suicide',
            'self harm', 'hurt myself', 'no reason to live', "don't want to exist"
        ]
        if any(kw in user_input.lower() for kw in crisis_keywords):
            for intent in self.intents:
                if intent['tag'] == 'crisis':
                    return intent['responses'][0]

        tag, score = self._get_intent(user_input)

        if tag is None:
            return (
                "I'm here to listen. Could you tell me more about how you're feeling? "
                "I want to make sure I understand and can support you properly."
            )

        for intent in self.intents:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])

        return "I'm here for you. Can you share a bit more about what's on your mind?"