# Multilingual Models

The API can run either the default standard model or a multilingual model.

## Default multilingual loader

```env
MODEL_SOURCE=default
USE_MULTILINGUAL_MODEL=true
```

## Hugging Face or local multilingual models

For custom multilingual models, configure supported languages explicitly.

```env
MODEL_SOURCE=hf_repo
MODEL_CLASS=multilingual
MODEL_REPO_ID=CoRal-project/roest-v3-chatterbox-500m
MODEL_SUPPORTED_LANGUAGES=da,en
DEFAULT_LANGUAGE=da
```

Or:

```env
MODEL_SOURCE=local_dir
MODEL_CLASS=multilingual
MODEL_LOCAL_PATH=./models/custom-model
MODEL_SUPPORTED_LANGUAGES=da,en
DEFAULT_LANGUAGE=da
```

`MODEL_SUPPORTED_LANGUAGES` accepts:

- comma-separated codes: `da,en`
- code/name pairs: `da:Danish,en:English`
- JSON arrays or objects

When no language is provided at request time, the API uses `DEFAULT_LANGUAGE`.
