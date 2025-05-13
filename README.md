# Arabic Text Autocorrection

This project provides an Arabic text autocorrection system using a masked language model approach.

## Setup

1. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

2. Make sure the model files are in the `model` directory.

## Usage

### Command Line Interface
Run the autocorrection script from the command line:
```
python autocorrect.py
```

### Graphical User Interface
Run the GUI application for a more user-friendly experience:
```
python run_gui.py
```

The GUI provides:
- Text input area for Arabic text
- Correction button to process the text
- Output area showing the corrected text
- Details panel showing which words were corrected

### Programmatic Usage
You can also import the functions in your own code:
```python
from autocorrect import pipeline

corrected_text = pipeline("وززارة النربية والتعليم")
print(corrected_text)  # Should print "وزارة التربية والتعليم"
```

## Features

- Arabic text preprocessing and normalization
- Misspelling detection using vocabulary and language model
- Context-aware word correction
- User-friendly graphical interface