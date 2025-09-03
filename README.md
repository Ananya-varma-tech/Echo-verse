# Echo-verse
Ai audio power tool
EchoVerse is a generative AI-based audiobook creation system that transforms user- provided text into expressive, downloadable audio content. Designed for accessibility, convenience, and content reusability, the tool empowers students, professionals, and visually impaired users to convert written material into natural-sounding narrations with customizable tone and voice.
The system accepts either pasted text or uploaded .txt files and displays the original content within the interface. Upon selection of a desired tone—Neutral, Suspenseful, or Inspiring— the text is rewritten using the IBM Watsonx Granite large language model. Prompt chaining is used to ensure the tone-specific rewrite remains faithful to the original meaning while enhancing stylistic quality.
Once rewritten, the text is passed to IBM Watson Text-to-Speech (TTS), where the user- selected voice—such as Lisa, Michael, or Allison—is used to synthesize high-quality narration. The final audio is streamed in-app using Streamlit’s audio playback component, and users are also given the option to download the file in .mp3 format.
A side-by-side layout showcases the original and rewritten text, supporting content verification and tone comparison. Additionally, a session-based “Past Narrations” panel displays all previously generated results, allowing users to replay or re-download audio within the same session. No login or persistent user tracking is implemented.
The tool is built using Python and Streamlit, with IBM Watsonx and TTS services integrated through API calls. Configuration variables such as API credentials and endpoints are secured using .env files. Though all voice styles worked successfully during testing, tone consistency was moderately affected by token limitations inherent in the current Watsonx deployment.
EchoVerse delivers a complete, low-friction workflow from text ingestion to expressive narration output—demonstrating how generative AI can enhance accessibility and user experience in content consumption.
Scenario 1: Study Notes to Audiobook.
Scenario 2: Blog to Podcast Adaptation.
Scenario 3: Accessibility for Visually Impaired Users.
Scenario 4: Exploring Voice and Tone Variations.
