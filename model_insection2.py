import speech_recognition as sr

# Initialize recognizer class (for recognizing the speech)
recognizer = sr.Recognizer()

# Function to convert speech to text
def speech_to_text():
    print("Say 'stop' to end the program.")
    while True:
        try:
            # Use the microphone as the audio source
            with sr.Microphone() as source:
                print("Listening...")
                
                # Adjust the recognizer sensitivity to ambient noise and record the audio
                recognizer.adjust_for_ambient_noise(source, duration=0.5)
                audio = recognizer.listen(source)

                # Using Google's Web Speech API to recognize the speech
                text = recognizer.recognize_google(audio).lower()
                print("You said: " + text)

                # Break the loop if 'stop' is said
                if "stop" in text:
                    print("Stopping speech recognition.")
                    break

        except sr.RequestError as e:
            print(f"Could not request results from Google Speech Recognition service; {e}")
        except sr.UnknownValueError:
            print("Google Speech Recognition could not understand the audio")
        except KeyboardInterrupt:
            print("Program manually stopped.")
            break

# Call the function to continuously listen and transcribe speech
speech_to_text()