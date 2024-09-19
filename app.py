import streamlit as st
import speech_recognition as sr
from io import BytesIO
from groq import Groq

# Initialize the recognizer
r = sr.Recognizer()

# Initialize Groq client
client = Groq(api_key="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")

# Function to evaluate the transcribed text using the Groq API
def evaluateText(text):
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

# Streamlit app
st.title("Speech to Transcription with Communication Evaluation")

# File uploader for audio files
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Display the file name
    st.write(f"Uploaded file: {uploaded_file.name}")

    # Let the user listen to the uploaded audio file
    st.audio(uploaded_file, format='audio/wav')

    # Process the audio file
    try:
        # Convert the uploaded file into an AudioFile instance for recognition
        audio_bytes = BytesIO(uploaded_file.read())
        with sr.AudioFile(audio_bytes) as source:
            r.adjust_for_ambient_noise(source, duration=0.2)
            audio_data = r.record(source)

            # Use Google API to recognize speech from the audio file
            text = r.recognize_google(audio_data).lower()
            st.write("Transcription:")
            st.write(text)

            # Evaluate the transcription
            st.write("Evaluating communication and professional speaking skills...")
            evaluation = evaluateText(text)
            st.write("Evaluation:")
            st.write(evaluation["code"])

    except sr.RequestError as e:
        st.error(f"Could not request results; {e}")
    except sr.UnknownValueError:
        st.error("Unknown error occurred while processing the audio.")
