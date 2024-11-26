# Sentiment Analysis System using LLaMA 3.1

A robust sentiment analysis system that leverages LLaMA 3.1 through the Groq API, featuring in-context learning capabilities for analyzing text sentiment.

## Features

- Zero-shot and Few-shot sentiment analysis
- Integration with Groq's LLaMA 3.1 model
- Interactive Streamlit web interface
- Automatic PDF report generation
- Rate limiting handling with exponential backoff
- Batch processing capabilities

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sentiment-analyzer-icl.git
cd sentiment-analyzer-icl
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.streamlit/secrets.toml` file with your Groq API key:
```toml
GROQ_API_KEY = "your-api-key-here"
```

## Usage Guide

1. Start the Streamlit application:
```bash
streamlit run main.py
```

2. Enter your Groq API key in the sidebar
3. Select the prompt type (Zero-shot or Few-shot)
4. Click "Load Dataset & Run Analysis" to start the analysis
5. View results and download the generated PDF report

## Development

- The project uses Python 3.11
- Dependencies are managed through requirements.txt
- Code follows PEP 8 style guidelines

## Deployment on Streamlit Cloud

1. Push your code to GitHub
2. Visit [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add your Groq API key to Streamlit Cloud secrets
5. Deploy the application

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
