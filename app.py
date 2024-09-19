import streamlit as st
import speech_recognition as sr
import pyttsx3
from groq import Groq

# Initialize the recognizer
r = sr.Recognizer()

# Initialize Groq client
client = Groq(api_key="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")


# Function to evaluate the transcribed text using the Groq API
def evaluate_text(text):
    prompt = """
    You are a highly experienced communication and soft skills trainer. Evaluate the input text based on the following criteria: clarity of speech, professionalism, tone, grammar, vocabulary, fluency, coherence, and overall effectiveness in a professional setting. Provide constructive feedback on how the user can improve their communication and English speaking skills, if necessary.
    """
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": text}
    ]

    completion = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=messages,
        temperature=0.2,
        max_tokens=8000,
        top_p=0.6,
        stream=False
    )

    output = completion.choices[0].message.content
    temp = [{"role": "assistant", "content": output}]
    context = messages + temp
    return {"messages": context, "code": output}


# Function to convert text to speech
def speech_to_text(command):
    # Initialize the engine
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()


# Streamlit app
st.title("Speech to Text with Communication Evaluation")

# Record button
if st.button("Record"):
    with st.spinner('Recording...'):
        try:
            # Use the microphone as source for input
            with sr.Microphone() as source2:
                # Adjust for ambient noise
                r.adjust_for_ambient_noise(source2, duration=0.2)

                # Listen for user's input
                st.info("Please speak now...")
                audio2 = r.listen(source2)

                MyText = r.recognize_google(audio2).lower()

                # Display recognized text
                st.write("You said:")
                st.write(MyText)

                # Speak the recognized text
                speech_to_text(MyText)

                # Evaluate the transcription
                st.write("Evaluating communication and professional speaking skills...")
                evaluation = evaluate_text(MyText)
                st.write("Evaluation:")
                st.write(evaluation["code"])

        except sr.RequestError as e:
            st.error(f"Could not request results; {e}")
        except sr.UnknownValueError:
            st.error("Sorry, I could not understand the audio.")
