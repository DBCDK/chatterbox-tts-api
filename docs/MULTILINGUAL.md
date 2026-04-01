# Multilingual Support Documentation

<p align="center">
  <img src="https://img.shields.io/badge/Languages-22%20Supported-brightgreen" alt="22 Languages Supported">
  <img src="https://img.shields.io/badge/chatterbox--tts-v0.1.4-blue" alt="chatterbox-tts v0.1.4">
  <img src="https://img.shields.io/badge/API-OpenAI%20Compatible-orange" alt="OpenAI Compatible">
</p>

## Overview

Chatterbox TTS API supports multilingual text-to-speech generation across **22 languages** using the enhanced `chatterbox-tts` v0.1.4 multilingual model. This feature enables high-quality voice cloning and speech synthesis in multiple languages while maintaining full OpenAI API compatibility.

The API can now also load compatible multilingual Chatterbox models from Hugging Face or from a local snapshot directory. For custom models, configure the supported languages explicitly so `/languages` and voice validation match the loaded model.

### Key Features

🌍 **22 Languages Supported** - Generate speech in Arabic, Chinese, English, French, German, Italian, Japanese, Spanish, and more  
🎭 **Language-Aware Voice Cloning** - Upload voices with specific language assignments  
🔄 **Automatic Language Detection** - Speech generation automatically uses the voice's assigned language  
🧠 **Smart Fallbacks** - Graceful handling of missing languages with English fallback  
📚 **Voice Library Integration** - Language metadata stored with each voice  
⚙️ **Configurable** - Enable/disable multilingual mode via environment variables  
🔗 **OpenAI Compatible** - No breaking changes to existing API endpoints  
📱 **Frontend Support** - Language selection UI with flags and native names

## Supported Languages

The multilingual model supports the following 22 languages:

| Code | Language   | Native Name   | Flag |
| ---- | ---------- | ------------- | ---- |
| `ar` | Arabic     | العربية       | 🇸🇦   |
| `da` | Danish     | Dansk         | 🇩🇰   |
| `de` | German     | Deutsch       | 🇩🇪   |
| `el` | Greek      | Ελληνικά      | 🇬🇷   |
| `en` | English    | English       | 🇺🇸   |
| `es` | Spanish    | Español       | 🇪🇸   |
| `fi` | Finnish    | Suomi         | 🇫🇮   |
| `fr` | French     | Français      | 🇫🇷   |
| `he` | Hebrew     | עברית         | 🇮🇱   |
| `hi` | Hindi      | हिन्दी        | 🇮🇳   |
| `it` | Italian    | Italiano      | 🇮🇹   |
| `ja` | Japanese   | 日本語        | 🇯🇵   |
| `ko` | Korean     | 한국어        | 🇰🇷   |
| `ms` | Malay      | Bahasa Melayu | 🇲🇾   |
| `nl` | Dutch      | Nederlands    | 🇳🇱   |
| `no` | Norwegian  | Norsk         | 🇳🇴   |
| `pl` | Polish     | Polski        | 🇵🇱   |
| `pt` | Portuguese | Português     | 🇵🇹   |
| `ru` | Russian    | Русский       | 🇷🇺   |
| `sv` | Swedish    | Svenska       | 🇸🇪   |
| `sw` | Swahili    | Kiswahili     | 🇹🇿   |
| `tr` | Turkish    | Türkçe        | 🇹🇷   |

> **Note**: Chinese (`zh`) support is available in the model but currently disabled. Contact support if you need Chinese language support.

## Configuration

### Enable/Disable Multilingual Mode

Multilingual support is controlled by the `USE_MULTILINGUAL_MODEL` environment variable:

```bash
# Enable multilingual support (default)
USE_MULTILINGUAL_MODEL=true

# Disable multilingual support (English only)
USE_MULTILINGUAL_MODEL=false
```

**Default Behavior:**

- Multilingual mode is **enabled by default** (`true`)
- When disabled, only English is supported
- Existing installations automatically get multilingual support

### Environment Variables

Add to your `.env` file:

```env
# Multilingual TTS Configuration
USE_MULTILINGUAL_MODEL=true   # Enable 23-language support (default: true)

# Custom Hugging Face or local multilingual model
# MODEL_SOURCE=hf_repo
# MODEL_CLASS=multilingual
# MODEL_REPO_ID=CoRal-project/roest-v3-chatterbox-500m
# MODEL_SUPPORTED_LANGUAGES=da,en
# DEFAULT_LANGUAGE=da
```

## API Usage

### 1. Get Supported Languages

Retrieve the list of languages supported by your current configuration:

```bash
curl http://localhost:4123/languages
```

**Response (Multilingual Mode):**

```json
{
  "languages": [
    { "code": "ar", "name": "Arabic" },
    { "code": "da", "name": "Danish" },
    { "code": "de", "name": "German" }
    // ... all 23 languages
  ],
  "count": 23,
  "model_type": "multilingual"
}
```

**Response (Standard Mode):**

```json
{
  "languages": [{ "code": "en", "name": "English" }],
  "count": 1,
  "model_type": "standard"
}
```

### 2. Upload Voice with Language

Upload a voice sample and assign a specific language:

```bash
curl -X POST http://localhost:4123/voices \
  -F "voice_name=french_speaker" \
  -F "language=fr" \
  -F "voice_file=@french_voice.wav"
```

**Parameters:**

- `voice_name`: Unique identifier for the voice
- `language`: ISO 639-1 language code (e.g., `fr`, `de`, `ja`)
- `voice_file`: Audio file in supported format

**Language Validation:**

- Language codes are validated against supported languages
- Invalid codes return a clear error message
- Defaults to `"en"` if not specified

### 3. Generate Multilingual Speech

Once a voice is uploaded with a language, speech generation automatically uses the correct language:

```bash
# Generate French speech using French voice
curl -X POST http://localhost:4123/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Bonjour, comment allez-vous?",
    "voice": "french_speaker"
  }' \
  --output french_speech.wav
```

**Key Points:**

- **No language parameter needed** in speech requests (OpenAI compatibility)
- Language is automatically determined from voice metadata
- Text can be in any language - the model handles cross-lingual synthesis
- All standard TTS parameters work with multilingual voices

### 4. Voice Library with Language Metadata

List voices to see language information:

```bash
curl http://localhost:4123/voices
```

**Response:**

```json
{
  "voices": [
    {
      "name": "french_speaker",
      "file_path": "/voices/french_speaker.wav",
      "aliases": [],
      "metadata": {
        "language": "fr",
        "created_at": "2024-01-15T10:30:00Z",
        "file_size": 2048576,
        "duration": 12.5
      }
    }
  ],
  "count": 1
}
```

## Advanced Usage Examples

### Python Examples

#### Upload and Use Multilingual Voice

```python
import requests

# Upload a German voice
with open("german_speaker.wav", "rb") as voice_file:
    response = requests.post(
        "http://localhost:4123/voices",
        data={
            "voice_name": "german_narrator",
            "language": "de"
        },
        files={
            "voice_file": ("german_speaker.wav", voice_file, "audio/wav")
        }
    )

print(f"Upload status: {response.status_code}")

# Generate German speech
response = requests.post(
    "http://localhost:4123/v1/audio/speech",
    json={
        "input": "Guten Tag! Wie geht es Ihnen heute?",
        "voice": "german_narrator",
        "exaggeration": 0.8
    }
)

with open("german_output.wav", "wb") as f:
    f.write(response.content)
```

#### Batch Upload Multiple Languages

```python
import requests
import os

voices = [
    {"file": "spanish_voice.wav", "name": "spanish_speaker", "lang": "es"},
    {"file": "italian_voice.wav", "name": "italian_speaker", "lang": "it"},
    {"file": "japanese_voice.wav", "name": "japanese_speaker", "lang": "ja"},
]

for voice in voices:
    with open(voice["file"], "rb") as f:
        response = requests.post(
            "http://localhost:4123/voices",
            data={
                "voice_name": voice["name"],
                "language": voice["lang"]
            },
            files={"voice_file": f}
        )
    print(f"Uploaded {voice['name']}: {response.status_code}")
```

#### Generate Speech in Multiple Languages

```python
import requests

texts = [
    {"text": "Hello, how are you today?", "voice": "english_speaker"},
    {"text": "Hola, ¿cómo estás hoy?", "voice": "spanish_speaker"},
    {"text": "Ciao, come stai oggi?", "voice": "italian_speaker"},
    {"text": "こんにちは、今日はいかがですか？", "voice": "japanese_speaker"},
]

for i, item in enumerate(texts):
    response = requests.post(
        "http://localhost:4123/v1/audio/speech",
        json={
            "input": item["text"],
            "voice": item["voice"]
        }
    )

    with open(f"multilingual_output_{i+1}.wav", "wb") as f:
        f.write(response.content)
```

### Streaming with Multilingual Voices

```bash
# Stream Japanese speech
curl -X POST http://localhost:4123/v1/audio/speech/stream \
  -H "Content-Type: application/json" \
  -d '{
    "input": "こんにちは。私の名前は田中です。よろしくお願いします。",
    "voice": "japanese_speaker",
    "chunk_strategy": "sentence"
  }' \
  --output japanese_stream.wav
```

### Voice Upload with Custom Parameters

```bash
# Upload with additional metadata and parameters
curl -X POST http://localhost:4123/voices \
  -F "voice_name=professional_german" \
  -F "language=de" \
  -F "voice_file=@professional_voice.wav"
```

## Frontend Integration

The web UI includes comprehensive multilingual support:

### Language Selection

- Dropdown with native language names and flag emojis
- Automatic validation against supported languages
- Default selection to English

### Voice Library Display

- Language badges next to each voice
- Flag emojis for visual identification
- Sorting and filtering by language

### Upload Interface

- Language selection integrated into voice upload modal
- Real-time validation and feedback
- Intuitive language picker with search

## Technical Implementation

### Architecture

The multilingual implementation consists of several key components:

1. **Model Loading**: Automatic detection and loading of multilingual vs standard TTS model
2. **Language Detection**: Voice metadata stores language information
3. **Speech Generation**: Automatic language parameter injection based on voice metadata
4. **API Compatibility**: Maintains OpenAI API format without breaking changes

### Model Switching

```python
# Automatic model selection based on configuration
if Config.USE_MULTILINGUAL_MODEL:
    model = ChatterboxMultilingualTTS(...)
    supported_languages = SUPPORTED_LANGUAGES
else:
    model = ChatterboxTTS(...)
    supported_languages = {"en": "English"}
```

### Language Resolution

```python
def resolve_voice_path_and_language(voice_name_or_path):
    """Resolve voice path and extract language metadata"""
    if voice_name_or_path in voice_library:
        voice_info = voice_library.get_voice_info(voice_name_or_path)
        return voice_info.path, voice_info.language
    else:
        return voice_name_or_path, "en"  # Default to English
```

### Backward Compatibility

- **Existing voices**: Automatically assigned English (`"en"`) language
- **Existing API calls**: Continue to work without modification
- **Configuration**: Multilingual mode can be disabled for compatibility
- **Graceful degradation**: Falls back to English for unsupported languages

## Performance Considerations

### Memory Usage

- Multilingual model requires slightly more memory than standard model
- Language switching doesn't require model reloading
- Voice library scales efficiently with multiple languages

### Generation Speed

- Multilingual generation performance is comparable to standard model
- Language-specific optimizations built into the model
- Streaming maintains low latency across all languages

### Storage

- Voice files stored with language metadata in JSON format
- No additional storage overhead for multilingual support
- Efficient indexing by language for large voice libraries

## Troubleshooting

### Common Issues

**Languages endpoint returns only English**

```bash
# Check multilingual configuration
curl http://localhost:4123/config | grep USE_MULTILINGUAL_MODEL
```

**Voice upload fails with language validation error**

```json
{
  "error": {
    "message": "Unsupported language code: xx. Supported: ar, da, de, ...",
    "type": "language_validation_error"
  }
}
```

**Speech generation ignores voice language**

- Ensure voice was uploaded with correct language parameter
- Check voice metadata: `curl http://localhost:4123/voices`
- Verify multilingual mode is enabled

### Debugging

Enable debug logging for multilingual operations:

```bash
# Check current configuration
curl http://localhost:4123/config

# Verify supported languages
curl http://localhost:4123/languages

# Check voice metadata
curl http://localhost:4123/voices
```

## Migration Guide

### From Standard to Multilingual

1. **Update dependencies** (already done in v0.1.4):

   ```bash
   uv sync  # or pip install -r requirements.txt
   ```

2. **Enable multilingual mode**:

   ```bash
   echo "USE_MULTILINGUAL_MODEL=true" >> .env
   ```

3. **Restart the API**:

   ```bash
   uv run main.py  # or python main.py
   ```

4. **Upload new voices with languages**:
   ```bash
   curl -X POST http://localhost:4123/voices \
     -F "voice_name=multilingual_voice" \
     -F "language=fr" \
     -F "voice_file=@voice.wav"
   ```

### Existing Voice Library

- Existing voices continue to work unchanged
- All existing voices default to English (`"en"`)
- Optionally re-upload voices with correct language assignments
- No data loss or corruption

## Best Practices

### Voice Quality Guidelines

1. **Language-Specific Recordings**:

   - Use native speakers for each language
   - Record in the target language for best results
   - Avoid mixing languages within a single voice sample

2. **Audio Quality**:

   - 10-30 seconds of clear speech
   - Consistent speaking pace and tone
   - Minimal background noise
   - High-quality audio format (WAV preferred)

3. **Voice Naming**:
   - Include language in voice names: `french_narrator`, `spanish_casual`
   - Use descriptive names for different styles: `german_formal`, `italian_cheerful`
   - Consider voice characteristics: `japanese_female_young`, `arabic_male_deep`

### Multilingual Workflows

1. **Development**:

   - Test with multiple languages during development
   - Validate language assignment for uploaded voices
   - Use streaming for better user experience with longer texts

2. **Production**:

   - Monitor memory usage with multiple language models
   - Implement proper error handling for unsupported languages
   - Consider caching frequently used voice/language combinations

3. **Content Management**:
   - Organize voices by language and use case
   - Document voice characteristics and appropriate use cases
   - Maintain consistent quality standards across languages

## API Reference

### Endpoints

| Endpoint                  | Method | Description                              |
| ------------------------- | ------ | ---------------------------------------- |
| `/languages`              | GET    | Get supported languages                  |
| `/voices`                 | POST   | Upload voice with language               |
| `/voices`                 | GET    | List voices with language metadata       |
| `/v1/audio/speech`        | POST   | Generate speech (language auto-detected) |
| `/v1/audio/speech/stream` | POST   | Stream speech generation                 |

### Request/Response Models

#### SupportedLanguageItem

```json
{
  "code": "fr",
  "name": "French"
}
```

#### SupportedLanguagesResponse

```json
{
  "languages": [SupportedLanguageItem],
  "count": 23,
  "model_type": "multilingual"
}
```

#### VoiceLibraryItem

```json
{
  "name": "french_speaker",
  "file_path": "/voices/french_speaker.wav",
  "aliases": [],
  "metadata": {
    "language": "fr",
    "created_at": "2024-01-15T10:30:00Z",
    "file_size": 2048576,
    "duration": 12.5
  }
}
```

## Examples Repository

For more examples and integration patterns, see:

- [Basic multilingual examples](../tests/test_api.py)
- [Frontend implementation](../frontend/src/components/VoiceUploadModal.tsx)
- [Testing guide](./MULTILINGUAL_TESTING_GUIDE.md)

## Support

- 📖 **Documentation**: [Main README](../README.md) | [API Documentation](./API_README.md)
- 💬 **Discord**: [Join the community](http://chatterboxtts.com/discord)

---

_Built with `chatterbox-tts` v0.1.4 • Supports 22 languages • OpenAI API Compatible_
